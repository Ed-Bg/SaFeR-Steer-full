"""
多轮对话推理模块
"""

import json
import base64
import os
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from io import BytesIO
from typing import Dict, List, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from datetime import datetime
import threading

# 线程本地存储，用于复用 HTTP Session
_thread_local = threading.local()

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.image_utils import check_and_resize_image
from utils.path_utils import fix_image_path
from .prompts import DEFAULT_INFER_SYSTEM_PROMPT


def load_benchmark_data(jsonl_path: str) -> List[Dict]:
    """
    加载单个Benchmark的JSONL数据
    
    Args:
        jsonl_path: JSONL文件完整路径
    
    Returns:
        样本列表，每个元素包含：
        - question_id: 唯一标识
        - image: 图像路径
        - messages_fixed: 预设的对话消息列表
        - category: 风险类别
        - risk_trajectory: Ground Truth风险轨迹（可选）
    """
    samples = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def get_system_prompt(config_prompt: Optional[str], data_messages: List[Dict]) -> str:
    """
    按优先级获取System Prompt
    
    优先级1: config_prompt（用户配置）
    优先级2: data_messages[0]["content"]（数据集中的system）
    
    Args:
        config_prompt: CONFIG中配置的system_prompt（可为None）
        data_messages: 数据集中的messages_fixed列表
    
    Returns:
        最终使用的system prompt字符串
    """
    # 优先级1: 用户配置
    if config_prompt:
        return config_prompt
    
    # 优先级2: 数据集中的system消息
    if data_messages and len(data_messages) > 0:
        if data_messages[0].get("role") == "system":
            return data_messages[0]["content"]
    
    # 默认
    return DEFAULT_INFER_SYSTEM_PROMPT


def extract_text_from_content(content: Union[str, List[Dict]]) -> str:
    """
    从消息content中提取纯文本
    
    Args:
        content: 可能是字符串或列表格式的content
    
    Returns:
        提取的纯文本内容
    """
    if isinstance(content, str):
        return content
    
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(item.get("text", ""))
        return " ".join(text_parts)
    
    return str(content)


def build_first_turn_message(
    user_content: List[Dict],
    image_path: str,
    max_pixels: int,
    min_pixels: int,
    path_mapping: Dict[str, str],
    benchmark_name: str = ""
) -> Dict:
    """
    构建第一轮用户消息（可能包含图像）
    
    处理流程：
    1. 检查图像路径是否存在
    2. 如果有图像：修复路径 → 处理大小 → 编码base64
    3. 如果无图像：只返回文本消息（仅mmsafetybench允许）
    4. 构建OpenAI兼容的消息格式
    
    Args:
        user_content: 原始用户消息的content列表
        image_path: 图像文件路径（可能为空，mmsafetybench存在无图数据）
        max_pixels: 最大像素数限制
        min_pixels: 最小像素数限制
        path_mapping: 图像路径映射
        benchmark_name: 数据集名称，用于判断是否允许图像缺失
    
    Returns:
        格式化的消息（有图或无图）
    
    Raises:
        ValueError: 非mmsafetybench数据集图像路径为空
        FileNotFoundError: 非mmsafetybench数据集图像文件不存在
    """
    # 提取文本
    text = extract_text_from_content(user_content)
    
    # 判断是否是 mmsafetybench 数据集（允许无图像）
    is_mmsafetybench = "mmsafetybench" in benchmark_name.lower() if benchmark_name else False
    
    # 检查是否有图像
    if not image_path or not image_path.strip():
        if is_mmsafetybench:
            # mmsafetybench 允许无图像数据
            return {
                "role": "user",
                "content": [{"type": "text", "text": text}]
            }
        else:
            # 其他数据集图像字段为空，终端提示并抛出异常
            raise ValueError(
                f"❌ 图像路径为空！\n"
                f"   数据集: {benchmark_name}\n"
                f"   该数据集要求必须有图像，请检查数据文件。"
            )
    
    try:
        # 修复图像路径
        fixed_image_path = fix_image_path(image_path, path_mapping)
        
        # 检查文件是否存在
        if not os.path.exists(fixed_image_path):
            if is_mmsafetybench:
                # mmsafetybench 允许图像文件不存在
                return {
                    "role": "user",
                    "content": [{"type": "text", "text": text}]
                }
            else:
                # 其他数据集图像文件不存在，终端提示并抛出异常
                raise FileNotFoundError(
                    f"❌ 图像文件不存在！\n"
                    f"   数据集: {benchmark_name}\n"
                    f"   原始路径: {image_path}\n"
                    f"   修复后路径: {fixed_image_path}\n"
                    f"   请检查图像文件是否存在或路径映射是否正确。"
                )
        
        # 处理图像
        img = check_and_resize_image(fixed_image_path, max_pixels, min_pixels)
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        # 构建包含图像的消息
        return {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}},
                {"type": "text", "text": text}
            ]
        }
    
    except Exception as e:
        # 图像处理失败，降级为纯文本（避免因图像问题导致整个样本失败）
        print(f"⚠️  图像处理失败 ({image_path}): {e}，降级为纯文本")
        return {
            "role": "user",
            "content": [{"type": "text", "text": text}]
        }


def build_subsequent_turn_message(user_content: Union[str, List[Dict]]) -> Dict:
    """
    构建后续轮用户消息（仅文本，无图像）
    
    Args:
        user_content: 原始用户消息的content列表或字符串
    
    Returns:
        格式化的消息：
        {
            "role": "user",
            "content": [{"type": "text", "text": "用户问题"}]
        }
    """
    text = extract_text_from_content(user_content)
    
    return {
        "role": "user",
        "content": [{"type": "text", "text": text}]
    }


def get_session() -> requests.Session:
    """
    获取线程本地的 HTTP Session，复用连接以提高性能
    
    Returns:
        requests.Session: 线程本地的 Session 对象
    """
    if not hasattr(_thread_local, "session"):
        session = requests.Session()
        # 配置连接池和重试策略
        retry_strategy = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[502, 503, 504],
        )
        adapter = HTTPAdapter(
            pool_connections=500,  # 连接池大小，需足够大以支持并发
            pool_maxsize=500,      # 最大连接数
            max_retries=retry_strategy
        )
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        _thread_local.session = session
    return _thread_local.session


def call_vllm(
    history: List[Dict],
    model_id: str,
    port: int,
    config: Dict
) -> str:
    """
    调用vLLM API生成回复
    
    Args:
        history: 完整对话历史
        model_id: 模型ID
        port: 服务端口
        config: 包含temperature, max_tokens, top_p, top_k, 
                repetition_penalty, timeout等参数
    
    Returns:
        模型生成的回复文本
    
    Raises:
        requests.HTTPError: HTTP 错误（400, 500 等）
        requests.Timeout: 请求超时
        requests.ConnectionError: 连接错误
        KeyError/IndexError: 响应格式错误
    """
    url = f"http://localhost:{port}/v1/chat/completions"
    
    payload = {
        "model": model_id,
        "messages": history,
        "temperature": config.get("temperature", 0.0),
        "max_tokens": config.get("max_tokens", 8192),  # 默认8192，避免长回复被截断
        "top_p": config.get("top_p", 1.0),
        "top_k": config.get("top_k", -1),
        "repetition_penalty": config.get("repetition_penalty", 1.1),
    }
    
    timeout_sec = config.get("timeout", 300)
    
    # 使用线程本地 Session 复用连接
    session = get_session()
    
    try:
        response = session.post(url, json=payload, timeout=timeout_sec)
        response.raise_for_status()  # 抛出 HTTPError (400, 500 等)
        
        result = response.json()  # 可能抛出 JSONDecodeError
        return result["choices"][0]["message"]["content"]  # 可能抛出 KeyError/IndexError
    
    except requests.HTTPError as e:
        # HTTP 错误（400, 500 等）- 明确是 vLLM 服务问题
        raise requests.HTTPError(f"vLLM returned HTTP {e.response.status_code}: {e}") from e
    
    except requests.Timeout as e:
        # 超时 - vLLM 服务可能过载或无响应
        raise
    
    except requests.ConnectionError as e:
        # 连接错误 - vLLM 服务可能已崩溃
        raise
    
    except (KeyError, IndexError) as e:
        # 响应格式错误 - vLLM 返回了异常格式
        raise ValueError(f"Invalid vLLM response format: {e}") from e


def run_single_sample_inference(
    sample: Dict,
    model_id: str,
    port: int,
    config: Dict,
    path_mapping: Dict[str, str],
    benchmark_name: str = ""
) -> Dict:
    """
    对单个样本执行完整多轮对话推理
    
    核心逻辑（同一线程内串行执行）：
    1. 获取system_prompt（按优先级）
    2. 初始化history = [{"role": "system", "content": system_prompt}]
    3. 解析messages_fixed，提取所有user轮次
    4. for turn_idx, user_msg in enumerate(user_turns):
           if turn_idx == 0:
               msg = build_first_turn_message(...)  # 含图像
           else:
               msg = build_subsequent_turn_message(...)  # 仅文本
           
           history.append(msg)
           response = call_vllm(history, ...)
           history.append({"role": "assistant", "content": response})
           
           记录本轮结果
    5. 返回完整结果
    
    Args:
        sample: 单个样本数据
        model_id: 模型ID
        port: 服务端口
        config: 配置字典
        path_mapping: 图像路径映射
        benchmark_name: 数据集名称，用于判断是否允许图像缺失
    
    Returns:
        推理结果字典：
        {
            "question_id": str,
            "category": str,
            "source_category": str,
            "total_turns": int,
            "image_path": str,
            "conversation": [
                {"turn": 1, "user": str, "assistant": str, "has_image": True},
                {"turn": 2, "user": str, "assistant": str, "has_image": False},
                ...
            ],
            "final_response": str,
            "risk_trajectory_gt": List[float]  # 保留GT供复盘
        }
    """
    try:
        # 验证必需字段
        question_id = sample.get("question_id")
        if not question_id:
            raise ValueError("Missing required field: question_id")
        
        messages_fixed = sample.get("messages_fixed")
        if not messages_fixed:
            raise ValueError("Missing required field: messages_fixed")
        
        # 获取可选字段
        category = sample.get("category", "")
        source_category = sample.get("source_category", category)
        image_path = sample.get("image", "")  # 图片可能为空（mmsafetybench有无图数据）
        risk_trajectory_gt = sample.get("risk_trajectory", [])
        
        # 1. 获取system_prompt
        system_prompt = get_system_prompt(config.get("system_prompt"), messages_fixed)
        
        # 2. 初始化history
        history = [{"role": "system", "content": system_prompt}]
        
        # 3. 提取所有user轮次（跳过system消息）
        user_turns = []
        for msg in messages_fixed:
            if isinstance(msg, dict) and msg.get("role") == "user":
                user_turns.append(msg.get("content", ""))
        
        # 4. 多轮对话推理（串行）
        conversation = []
        final_response = ""
        
        for turn_idx, user_content in enumerate(user_turns):
            # 构建消息
            if turn_idx == 0:
                # 第一轮：可能含图像
                msg = build_first_turn_message(
                    user_content,
                    image_path,
                    config.get("image_max_pixels", 512 * 512),
                    config.get("image_min_pixels", 338 * 338),
                    path_mapping,
                    benchmark_name
                )
                # 检查是否真的包含图像
                has_image = any(
                    item.get("type") == "image_url" 
                    for item in msg.get("content", [])
                    if isinstance(item, dict)
                )
            else:
                # 后续轮：仅文本
                msg = build_subsequent_turn_message(user_content)
                has_image = False
            
            # 提取用户文本
            user_text = extract_text_from_content(user_content)
            
            # 添加到历史
            history.append(msg)
            
            # 调用API
            response = call_vllm(history, model_id, port, config)
            
            # 添加助手回复到历史
            history.append({"role": "assistant", "content": response})
            
            # 记录本轮结果
            conversation.append({
                "turn": turn_idx + 1,
                "user": user_text,
                "assistant": response,
                "has_image": has_image
            })
            
            final_response = response
        
        # 5. 返回结果
        return {
            "question_id": question_id,
            "category": category,
            "source_category": source_category,
            "total_turns": len(user_turns),
            "image_path": image_path,
            "conversation": conversation,
            "final_response": final_response,
            "risk_trajectory_gt": risk_trajectory_gt
        }
        
    except (requests.HTTPError, requests.Timeout, requests.ConnectionError) as e:
        # vLLM API 服务错误（应该触发连续错误检查）
        return {
            "question_id": sample.get("question_id", "unknown"),
            "error": f"vLLM API Error: {type(e).__name__}: {str(e)}",
            "error_type": "vllm_error",  # 标记为 vLLM 错误
            "timestamp": datetime.now().isoformat()
        }
    
    except ValueError as e:
        # ValueError 可能来自：
        # 1. 字段验证（"Missing required field"）
        # 2. call_vllm 的响应格式错误（"Invalid vLLM response"）
        # 3. 图像路径为空（非mmsafetybench数据集）
        error_msg = str(e)
        if "vLLM" in error_msg or "Invalid vLLM" in error_msg:
            # 来自 vLLM 的响应格式错误
            return {
                "question_id": sample.get("question_id", "unknown"),
                "error": f"vLLM Response Error: {error_msg}",
                "error_type": "vllm_error",
                "timestamp": datetime.now().isoformat()
            }
        elif "图像路径为空" in error_msg:
            # 图像路径为空错误（非mmsafetybench数据集）
            print(f"\n{error_msg}")  # 终端输出提示
            return {
                "question_id": sample.get("question_id", "unknown"),
                "error": error_msg,
                "error_type": "image_error",
                "timestamp": datetime.now().isoformat()
            }
        else:
            # 数据格式错误（如字段缺失）
            return {
                "question_id": sample.get("question_id", "unknown"),
                "error": f"Data Validation Error: {error_msg}",
                "error_type": "data_error",
                "timestamp": datetime.now().isoformat()
            }
    
    except (KeyError, TypeError, IndexError) as e:
        # 数据结构错误（不应该触发连续错误）
        # 注：由于已经使用 .get() 和验证，这些异常应该很少出现
        return {
            "question_id": sample.get("question_id", "unknown"),
            "error": f"Data Structure Error: {type(e).__name__}: {str(e)}",
            "error_type": "data_error",
            "timestamp": datetime.now().isoformat()
        }
    
    except FileNotFoundError as e:
        # 图像文件不存在（非mmsafetybench数据集）
        error_msg = str(e)
        print(f"\n{error_msg}")  # 终端输出提示
        return {
            "question_id": sample.get("question_id", "unknown"),
            "error": error_msg,
            "error_type": "image_error",
            "timestamp": datetime.now().isoformat()
        }
    
    except (IOError, OSError) as e:
        # 其他文件/IO错误
        return {
            "question_id": sample.get("question_id", "unknown"),
            "error": f"File/IO Error: {type(e).__name__}: {str(e)}",
            "error_type": "file_error",
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        # 其他未预期错误
        import traceback
        return {
            "question_id": sample.get("question_id", "unknown"),
            "error": f"Unexpected Error: {type(e).__name__}: {str(e)}",
            "error_type": "unexpected",
            "traceback": traceback.format_exc(),
            "timestamp": datetime.now().isoformat()
        }


def should_skip_sample(question_id: str, existing_results: Dict, skip_existing: bool) -> bool:
    """
    判断是否跳过已完成的样本
    
    Args:
        question_id: 样本ID
        existing_results: 已有结果字典
        skip_existing: 是否启用断点续传
    
    Returns:
        True表示跳过，False表示需要处理
    """
    return (
        skip_existing and 
        question_id in existing_results and 
        "error" not in existing_results[question_id]
    )


def check_vllm_health(port: int, model_id: str) -> bool:
    """
    检查vLLM服务是否健康运行
    
    Args:
        port: 服务端口
        model_id: 模型ID
    
    Returns:
        True表示服务正常，False表示服务异常
    """
    try:
        # 检查health端点
        health_url = f"http://localhost:{port}/health"
        response = requests.get(health_url, timeout=5)
        if response.status_code != 200:
            return False
        
        # 尝试简单的API调用
        test_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hi"}
        ]
        
        api_url = f"http://localhost:{port}/v1/chat/completions"
        test_response = requests.post(
            api_url,
            json={
                "model": model_id,
                "messages": test_messages,
                "max_tokens": 10,
                "temperature": 0.0
            },
            timeout=30
        )
        
        return test_response.status_code == 200
        
    except Exception as e:
        print(f"⚠️  vLLM健康检查失败: {e}")
        return False


def run_benchmark_inference(
    benchmark_file: str,
    benchmark_name: str,
    model_name: str,
    model_id: str,
    port: int,
    config: Dict,
    path_mapping: Dict[str, str],
    output_dir: str,
    max_workers: int
) -> str:
    """
    对单个Benchmark执行批量推理
    
    流程：
    1. 检查vLLM服务健康状态
    2. 加载JSONL数据
    3. 检查输出文件是否存在（断点续传）
    4. 创建ThreadPoolExecutor(max_workers)
    5. 并发提交所有样本的推理任务
    6. 使用tqdm显示进度，监控错误率
    7. 收集结果，如果错误率过高则报错
    8. 保存到 {output_dir}/{model_name}/{benchmark_name}_infer.json
    
    Args:
        benchmark_file: JSONL文件名
        benchmark_name: Benchmark标识
        model_name: 模型名称
        model_id: 模型ID
        port: 服务端口
        config: 配置字典
        path_mapping: 图像路径映射
        output_dir: 输出目录
        max_workers: 并发数
    
    Returns:
        输出文件路径
    
    Raises:
        RuntimeError: vLLM服务不可用或错误率过高
    """
    print(f"\n{'='*60}")
    print(f"开始推理: {benchmark_name}")
    print(f"{'='*60}")
    
    # 0. 健康检查
    print(f"检查vLLM服务健康状态...")
    if not check_vllm_health(port, model_id):
        raise RuntimeError(
            f"❌ vLLM服务不可用 (端口: {port})！\n"
            f"   请确保vLLM服务正常运行后再进行推理。\n"
            f"   检查方法：curl http://localhost:{port}/health"
        )
    
    # 1. 加载数据
    data_dir = config.get("data_dir", "safe_dataset/Benchmark/data")
    jsonl_path = os.path.join(data_dir, benchmark_file)
    samples = load_benchmark_data(jsonl_path)
    print(f"加载样本数: {len(samples)}")
    
    # 2. 准备输出目录和文件
    output_model_dir = os.path.join(output_dir, model_name)
    os.makedirs(output_model_dir, exist_ok=True)
    
    output_file = os.path.join(output_model_dir, f"{benchmark_name}_infer.json")
    
    # 检查已有结果（断点续传）
    existing_results = {}
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            existing_results = data.get("samples", {})
        print(f"已完成样本数: {len(existing_results)}")
    
    # 3. 筛选需要处理的样本
    skip_existing = config.get("skip_existing", True)
    samples_to_process = [
        s for s in samples 
        if not should_skip_sample(s["question_id"], existing_results, skip_existing)
    ]
    print(f"待处理样本数: {len(samples_to_process)}")
    
    # 4. 并发推理（带错误监控）
    results = {}
    results.update(existing_results)  # 保留已有结果
    
    if samples_to_process:
        error_count = 0
        consecutive_errors = 0
        processed_count = 0
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交任务
            future_to_sample = {
                executor.submit(
                    run_single_sample_inference,
                    sample, model_id, port, config, path_mapping, benchmark_name
                ): sample for sample in samples_to_process
            }
            
            # 收集结果（带错误监控）
            for future in tqdm(as_completed(future_to_sample), total=len(samples_to_process), desc="推理进度"):
                processed_count += 1
                
                try:
                    result = future.result()
                    question_id = result["question_id"]
                    
                    # 检查是否有错误
                    if "error" in result:
                        error_count += 1
                        
                        # 只有 vLLM 服务错误才计入连续错误
                        if result.get("error_type") == "vllm_error":
                            consecutive_errors += 1
                        else:
                            # 其他类型的错误（文件错误、数据错误等）不计入连续错误
                            consecutive_errors = 0
                        
                        # 检查连续错误（只针对 vLLM 服务错误）
                        if consecutive_errors >= 5:
                            raise RuntimeError(
                                f"\n\n{'='*60}\n"
                                f"❌ 检测到连续5次推理失败！\n"
                                f"{'='*60}\n"
                                f"最后一个错误: {result['error']}\n"
                                f"这通常意味着vLLM服务已崩溃或不可用。\n"
                                f"已完成: {processed_count}/{len(samples_to_process)} 样本\n"
                                f"错误率: {error_count/processed_count*100:.1f}%\n"
                                f"\n请检查vLLM服务状态后重新运行。\n"
                                f"{'='*60}\n"
                            )
                        
                        # 检查错误率
                        if processed_count >= 20:  # 至少处理20个样本后再检查错误率
                            error_rate = error_count / processed_count
                            if error_rate > 0.5:  # 错误率超过50%
                                raise RuntimeError(
                                    f"\n\n{'='*60}\n"
                                    f"❌ 错误率过高！({error_rate*100:.1f}%)\n"
                                    f"{'='*60}\n"
                                    f"已处理: {processed_count} 样本\n"
                                    f"失败: {error_count} 样本\n"
                                    f"最近的错误: {result['error']}\n"
                                    f"\nvLLM服务可能存在问题，已停止推理。\n"
                                    f"{'='*60}\n"
                                )
                    else:
                        consecutive_errors = 0  # 成功则重置连续错误计数
                    
                    results[question_id] = result
                    
                except KeyboardInterrupt:
                    # 用户中断，直接抛出
                    print("\n\n⚠️  检测到用户中断 (Ctrl+C)，保存当前结果...")
                    raise
                
                except Exception as e:
                    # 理论上不应该到这里，因为 run_single_sample_inference 已经捕获了所有异常
                    # 如果到这里，说明有严重问题（如 MemoryError、SystemExit 等）
                    sample = future_to_sample[future]
                    error_count += 1
                    consecutive_errors += 1
                    
                    import traceback
                    error_detail = traceback.format_exc()
                    
                    results[sample["question_id"]] = {
                        "question_id": sample["question_id"],
                        "error": f"Critical Exception: {type(e).__name__}: {str(e)}",
                        "error_type": "critical_error",
                        "traceback": error_detail,
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    print(f"\n❌ 严重异常 (question_id: {sample['question_id']})")
                    print(f"   类型: {type(e).__name__}")
                    print(f"   信息: {e}")
                    print(f"   这不应该发生，请检查代码或系统资源")
                    print(error_detail)
                    
                    # 检查连续严重错误
                    if consecutive_errors >= 3:  # 严重错误只允许3次
                        raise RuntimeError(
                            f"\n\n{'='*60}\n"
                            f"❌ 检测到连续3次严重异常！\n"
                            f"{'='*60}\n"
                            f"异常类型: {type(e).__name__}\n"
                            f"异常信息: {str(e)}\n"
                            f"已完成: {processed_count}/{len(samples_to_process)} 样本\n"
                            f"\n这可能是：\n"
                            f"  1. 系统资源耗尽（内存/GPU）\n"
                            f"  2. 代码严重bug\n"
                            f"  3. 环境问题\n"
                            f"\n请检查日志和系统状态。\n"
                            f"{'='*60}\n"
                        )
        
        # 推理完成后的错误统计
        final_error_rate = error_count / len(samples_to_process) if samples_to_process else 0
        if error_count > 0:
            print(f"\n⚠️  推理完成，但有 {error_count} 个样本失败 (错误率: {final_error_rate*100:.1f}%)")
            
            # 统计各类错误
            error_types_count = {}
            for result in results.values():
                if "error" in result:
                    error_type = result.get("error_type", "unknown")
                    error_types_count[error_type] = error_types_count.get(error_type, 0) + 1
            
            # 显示错误分类统计
            if error_types_count:
                print(f"\n错误分类统计:")
                for error_type, count in sorted(error_types_count.items(), key=lambda x: x[1], reverse=True):
                    print(f"  - {error_type}: {count} 个")
            
            # 高错误率警告
            if final_error_rate > 0.3:  # 错误率超过30%
                print(f"\n{'='*60}")
                print(f"⚠️  警告: 错误率较高 ({final_error_rate*100:.1f}%)")
                print(f"{'='*60}")
                
                # 根据主要错误类型给出建议
                main_error_type = max(error_types_count.items(), key=lambda x: x[1])[0] if error_types_count else "unknown"
                
                print(f"主要错误类型: {main_error_type}")
                print(f"\n建议检查:")
                
                if main_error_type == "vllm_error":
                    print(f"  1. vLLM服务日志")
                    print(f"  2. GPU内存是否充足")
                    print(f"  3. 模型是否正确加载")
                    print(f"  4. 网络连接是否稳定")
                elif main_error_type == "data_error":
                    print(f"  1. 数据格式是否正确")
                    print(f"  2. 必需字段是否完整")
                    print(f"  3. 数据类型是否匹配")
                elif main_error_type == "image_error":
                    print(f"  1. 图像字段是否为空（仅mmsafetybench允许）")
                    print(f"  2. 图像文件路径是否正确")
                    print(f"  3. 图像文件是否存在")
                    print(f"  4. 路径映射配置是否正确")
                elif main_error_type == "file_error":
                    print(f"  1. 图像文件路径是否正确")
                    print(f"  2. 文件是否存在")
                    print(f"  3. 文件权限是否正确")
                else:
                    print(f"  1. 查看详细错误信息")
                    print(f"  2. 检查日志文件")
                
                print(f"{'='*60}\n")
    
    # 7. 保存结果
    output_data = {
        "benchmark": benchmark_name,
        "model_name": model_name,
        "model_id": model_id,
        "config": {
            "system_prompt_source": "config" if config.get("system_prompt") else "data",
            "image_max_pixels": config.get("image_max_pixels", 512 * 512),
            "image_min_pixels": config.get("image_min_pixels", 338 * 338),
            "temperature": config.get("temperature", 0.0),
            "max_tokens": config.get("max_tokens", 16384)  # 默认8192
        },
        "total_samples": len(samples),
        "timestamp": datetime.now().isoformat(),
        "samples": results
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"推理完成，结果保存至: {output_file}")
    return output_file
