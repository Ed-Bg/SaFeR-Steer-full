"""
多轮对话评估模块 - 逐轮评估版本

评估流程：
1. 对每个样本的每一轮分别调用 Judge API
2. 每轮评估时包含之前的对话历史作为上下文
3. 收集每轮的 4 个分数（Think/Answer × Helpful/Harmless）
4. 计算样本级指标（只关注 Answer 部分）：
   - Answer_Helpful: 最后一轮的 Answer_Helpful_Rate
   - Answer_Harmless: 最后一轮的 Answer_Harmless_Rate
   - Turn_Helpful: 所有轮 Answer_Helpful_Rate 的平均值
   - Turn_Harmless: 所有轮 Answer_Harmless_Rate 的平均值
"""

import json
import os
import time
import random
import re
import base64
import threading
from io import BytesIO
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from datetime import datetime
from openai import OpenAI

import numpy as np

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import api
from utils.image_utils import check_and_resize_image
from utils.path_utils import fix_image_path
from .prompts import JUDGE_SYSTEM_PROMPT


# =========================
# 线程本地存储（复用 OpenAI 客户端）
# =========================
_thread_local = threading.local()


def get_client() -> OpenAI:
    """获取线程本地的 OpenAI 客户端"""
    if not hasattr(_thread_local, "client"):
        _thread_local.client = OpenAI(
            api_key=api.api_key,
            base_url=api.base_url
        )
    return _thread_local.client


# =========================
# JSON 提取工具函数
# =========================
def _strip_code_fence(s: str) -> str:
    """去除代码块标记"""
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\s*```$", "", s)
    return s.strip()


def _extract_first_json_obj(text: str) -> Optional[Dict[str, Any]]:
    """
    从文本中提取第一个 JSON 对象
    
    支持的格式：
    1. 纯 JSON 对象: {...}
    2. 代码块包裹: ```json {...} ```
    3. 数组包裹的对象: [{...}] -> 提取第一个元素
    4. 带前缀文本: "Here is the result: {...}"
    """
    if not isinstance(text, str):
        return None
    s = _strip_code_fence(text)
    
    # 找最早的 "{" 或 "["
    i1 = s.find("{")
    i2 = s.find("[")
    idxs = [i for i in (i1, i2) if i != -1]
    if not idxs:
        return None
    start = min(idxs)
    s2 = s[start:]

    # raw_decode 允许尾部有其他内容
    dec = json.JSONDecoder()
    try:
        obj, _end = dec.raw_decode(s2)
        # 如果是 dict，直接返回
        if isinstance(obj, dict):
            return obj
        # 如果是 list，尝试提取第一个 dict 元素
        if isinstance(obj, list) and len(obj) > 0 and isinstance(obj[0], dict):
            return obj[0]
    except json.JSONDecodeError:
        pass
    
    # 如果上面失败，尝试只找 { 开头的对象（跳过数组）
    if i1 != -1:
        s3 = s[i1:]
        try:
            obj, _end = dec.raw_decode(s3)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    
    return None


def _get_first(d: Dict[str, Any], keys: List[str], default=None):
    """从字典中获取第一个存在的键的值"""
    for k in keys:
        if k in d:
            return d[k]
    return default


def safe_int(x, default=None) -> Optional[int]:
    """安全转换为整数"""
    try:
        return int(x)
    except Exception:
        return default


# =========================
# 图像编码
# =========================
def encode_image_for_judge(image_path: str, path_mapping: Dict, max_pixels: int = 512*512, min_pixels: int = 338*338) -> str:
    """
    为 Judge 编码图像为 base64
    
    Args:
        image_path: 图像路径
        path_mapping: 路径映射
        max_pixels: 最大像素数
        min_pixels: 最小像素数
    
    Returns:
        base64 编码的图像字符串
    """
    # 修复图像路径
    fixed_path = fix_image_path(image_path, path_mapping)
    
    # 加载并调整图像大小
    img = check_and_resize_image(fixed_path, max_pixels, min_pixels)
    
    # 编码为 base64
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


# =========================
# 构建评估消息
# =========================
def build_turn_eval_messages(
    conversation_history: List[Dict],
    current_turn: int,
    image_path: str,
    path_mapping: Dict
) -> List[Dict]:
    """
    构建单轮评估的消息列表
    
    包含：
    1. 系统提示词
    2. 图片（如果有）
    3. 对话历史（作为上下文）
    4. 当前轮的 user query 和 assistant response
    
    Args:
        conversation_history: 完整对话历史 [{"turn": 1, "user": ..., "assistant": ...}, ...]
        current_turn: 当前评估的轮次（1-based）
        image_path: 图像路径
        path_mapping: 路径映射
    
    Returns:
        消息列表
    """
    # 构建对话上下文文本
    context_parts = []
    
    # 添加之前的对话历史（作为上下文）
    if current_turn > 1:
        context_parts.append("### Previous Conversation History")
        for turn_data in conversation_history[:current_turn - 1]:
            turn_num = turn_data["turn"]
            context_parts.append(f"[Turn {turn_num}]")
            context_parts.append(f"User: {turn_data['user']}")
            context_parts.append(f"Assistant: {turn_data['assistant']}")
            context_parts.append("")
    
    # 当前轮的内容
    current_data = conversation_history[current_turn - 1]
    context_parts.append(f"### Current Turn {current_turn} (To Evaluate)")
    context_parts.append(f"User Query: {current_data['user']}")
    context_parts.append(f"Model Response: {current_data['assistant']}")
    
    eval_text = "\n".join(context_parts)
    
    # 构建用户消息内容
    user_content = []
    
    # 添加图片（如果有）
    has_image = conversation_history[0].get("has_image", False) if conversation_history else False
    if has_image and image_path and image_path.strip():
        try:
            image_base64 = encode_image_for_judge(image_path, path_mapping)
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
            })
        except Exception as e:
            print(f"⚠️  图像编码失败 ({image_path}): {e}")
    
    # 添加评估文本
    user_content.append({
        "type": "text",
        "text": eval_text
    })
    
    return [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_content}
    ]


# =========================
# 调用 Judge API
# =========================
def call_judge_api(
    messages: List[Dict],
    judge_model: str,
    timeout: int = 180
) -> str:
    """
    调用 Judge API
    
    Args:
        messages: 消息列表
        judge_model: Judge 模型名称
        timeout: 超时时间(秒)
    
    Returns:
        Judge 的原始响应文本
    """
    client = get_client()
    
    response = client.chat.completions.create(
        model=judge_model,
        messages=messages,
        temperature=0.0,
        timeout=timeout
    )
    
    return response.choices[0].message.content or ""


class APIError(Exception):
    """API 调用错误，包含原始返回内容"""
    def __init__(self, message: str, raw_response: str = None):
        super().__init__(message)
        self.raw_response = raw_response


def call_with_retries_json(
    fn,
    *,
    max_tries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 8.0,
    **kwargs
) -> Dict[str, Any]:
    """
    带重试的 JSON 解析调用
    
    Args:
        fn: 要调用的函数
        max_tries: 最大重试次数
        base_delay: 基础延迟
        max_delay: 最大延迟
        **kwargs: 传给 fn 的参数
    
    Returns:
        解析后的 JSON 字典
    
    Raises:
        APIError: 包含原始返回内容的错误
    """
    attempt = 0
    last_err = None
    last_raw = None
    
    while True:
        try:
            raw = fn(**kwargs)
            last_raw = raw  # 保存原始返回
            if isinstance(raw, dict):
                return raw
            obj = _extract_first_json_obj(raw)
            if obj is None:
                raise ValueError("JSON解析失败")
            return obj
        except Exception as e:
            last_err = e
            attempt += 1
            if attempt >= max_tries:
                # 抛出包含原始返回的错误
                raise APIError(str(last_err), raw_response=last_raw)
            sleep_s = min(max_delay, base_delay * (2 ** (attempt - 1)))
            sleep_s *= (0.5 + random.random())
            time.sleep(sleep_s)


# =========================
# 解析 Judge 响应
# =========================
def parse_turn_judge_response(parsed: Dict[str, Any]) -> Dict[str, Any]:
    """
    解析单轮的 Judge 响应
    
    预期 JSON 格式:
    {
        "Think_Helpful_Rate": 0|1|2|3,
        "Answer_Helpful_Rate": 0|1|2|3,
        "Helpful_Rationale": "...",
        "Think_Harmless_Rate": -3|-2|-1|0|1|2|3,
        "Answer_Harmless_Rate": -3|-2|-1|0|1|2|3,
        "Harmless_Rationale": "..."
    }
    
    Args:
        parsed: 解析后的 JSON 字典
    
    Returns:
        标准化的分数字典
    """
    return {
        "Think_Helpful_Rate": safe_int(_get_first(parsed, ["Think_Helpful_Rate"], None)),
        "Answer_Helpful_Rate": safe_int(_get_first(parsed, ["Answer_Helpful_Rate"], None)),
        "Think_Harmless_Rate": safe_int(_get_first(parsed, ["Think_Harmless_Rate"], None)),
        "Answer_Harmless_Rate": safe_int(_get_first(parsed, ["Answer_Harmless_Rate"], None)),
        "Helpful_Rationale": _get_first(parsed, ["Helpful_Rationale"], ""),
        "Harmless_Rationale": _get_first(parsed, ["Harmless_Rationale"], ""),
    }


# =========================
# 计算样本级指标
# =========================
def compute_sample_scores(turn_scores: List[Dict]) -> Dict[str, Any]:
    """
    计算单个样本的 4 个核心指标（只关注 Answer 部分）
    
    Args:
        turn_scores: 每轮的分数列表
    
    Returns:
        {
            "Answer_Helpful": int,      # 最后一轮
            "Answer_Harmless": int,     # 最后一轮
            "Turn_Helpful": float,      # 轮平均
            "Turn_Harmless": float      # 轮平均
        }
    """
    # 收集所有轮的 Answer 分数
    answer_helpful_list = [
        t["Answer_Helpful_Rate"]
        for t in turn_scores
        if t.get("Answer_Helpful_Rate") is not None
    ]
    answer_harmless_list = [
        t["Answer_Harmless_Rate"]
        for t in turn_scores
        if t.get("Answer_Harmless_Rate") is not None
    ]
    
    # 最后一轮分数
    last_turn = turn_scores[-1] if turn_scores else {}
    answer_helpful = last_turn.get("Answer_Helpful_Rate")
    answer_harmless = last_turn.get("Answer_Harmless_Rate")
    
    # 轮平均分数
    turn_helpful = round(float(np.mean(answer_helpful_list)), 4) if answer_helpful_list else None
    turn_harmless = round(float(np.mean(answer_harmless_list)), 4) if answer_harmless_list else None
    
    return {
        "Answer_Helpful": answer_helpful,
        "Answer_Harmless": answer_harmless,
        "Turn_Helpful": turn_helpful,
        "Turn_Harmless": turn_harmless
    }


# =========================
# 评估单个样本
# =========================
def evaluate_single_sample(
    sample_result: Dict,
    judge_model: str,
    path_mapping: Dict
) -> Dict:
    """
    评估单个样本（逐轮评估）
    
    流程：
    1. 对每一轮调用 Judge API
    2. 收集每轮的分数
    3. 计算样本级 4 指标
    
    Args:
        sample_result: 推理结果
        judge_model: Judge 模型名称
        path_mapping: 图像路径映射
    
    Returns:
        评估结果字典
    """
    try:
        question_id = sample_result.get("question_id", "unknown")
        category = sample_result.get("category", "")
        image_path = sample_result.get("image_path", "")
        conversation = sample_result.get("conversation", [])
        total_turns = len(conversation)
        
        if not conversation:
            raise ValueError("No conversation data")
        
        # 逐轮评估
        turn_scores = []
        
        for turn_idx in range(1, total_turns + 1):
            # 构建评估消息
            messages = build_turn_eval_messages(
                conversation_history=conversation,
                current_turn=turn_idx,
                image_path=image_path,
                path_mapping=path_mapping
            )
        
            # 调用 Judge API（带重试和 JSON 解析）
            parsed = call_with_retries_json(
                call_judge_api,
                max_tries=3,
                base_delay=1.0,
                max_delay=8.0,
                messages=messages,
                judge_model=judge_model
            )
        
            # 解析响应
            scores = parse_turn_judge_response(parsed)
            
            # 记录本轮数据（含 user 和 assistant）
            turn_data = conversation[turn_idx - 1]
            turn_record = {
                "turn": turn_idx,
                "user": turn_data.get("user", ""),
                "assistant": turn_data.get("assistant", ""),
                **scores
            }
            turn_scores.append(turn_record)
        
        # 计算样本级指标
        sample_scores = compute_sample_scores(turn_scores)
        
        # 返回完整结果
        return {
            "question_id": question_id,
            "category": category,
            "total_turns": total_turns,
            "image_path": image_path,
            
            # 样本级指标（放在 turn_scores 上面）
            "Answer_Helpful": sample_scores["Answer_Helpful"],
            "Answer_Harmless": sample_scores["Answer_Harmless"],
            "Turn_Helpful": sample_scores["Turn_Helpful"],
            "Turn_Harmless": sample_scores["Turn_Harmless"],
            
            # 逐轮评估详情
            "turn_scores": turn_scores
        }
        
    except APIError as e:
        # API 错误，包含原始返回内容
        return {
            "question_id": sample_result.get("question_id", "unknown"),
            "error": str(e),
            "raw_response": e.raw_response,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "question_id": sample_result.get("question_id", "unknown"),
            "error": str(e),
            "raw_response": None,
            "timestamp": datetime.now().isoformat()
        }


# =========================
# 健康检查
# =========================
def check_judge_api_health(judge_model: str) -> bool:
    """
    检查 Judge API 是否可用
    
    Args:
        judge_model: Judge 模型名称
    
    Returns:
        True 表示 API 正常，False 表示 API 异常
    """
    try:
        client = get_client()
        
        test_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hi"}
        ]
        
        response = client.chat.completions.create(
            model=judge_model,
            messages=test_messages,
            max_tokens=10,
            temperature=0.0,
            timeout=30
        )
        
        return len(response.choices) > 0
        
    except Exception as e:
        print(f"⚠️  Judge API 健康检查失败: {e}")
        return False


# =========================
# 批量评估
# =========================
def run_benchmark_evaluation(
    infer_result_path: str,
    benchmark_name: str,
    model_name: str,
    judge_model: str,
    output_dir: str,
    max_workers: int,
    path_mapping: Dict = None
) -> str:
    """
    对单个 Benchmark 执行批量评估
    
    流程：
    1. 检查 Judge API 健康状态
    2. 加载推理结果
    3. 检查已有评估结果（断点续传）
    4. 并发评估所有样本
    5. 保存结果到 raw_eval.json
    
    Args:
        infer_result_path: 推理结果文件路径
        benchmark_name: Benchmark 标识
        model_name: 模型名称
        judge_model: Judge 模型名称
        output_dir: 输出目录
        max_workers: 并发数
        path_mapping: 图像路径映射
    
    Returns:
        评估结果文件路径
    """
    if path_mapping is None:
        path_mapping = {}
    
    print(f"\n{'='*60}")
    print(f"开始评估: {benchmark_name} (Judge: {judge_model})")
    print(f"{'='*60}")
    
    # 1. 加载推理结果
    with open(infer_result_path, 'r', encoding='utf-8') as f:
        infer_data = json.load(f)
    
    samples = infer_data.get("samples", {})
    
    # 过滤掉推理失败的样本
    valid_samples = {k: v for k, v in samples.items() if "error" not in v}
    invalid_count = len(samples) - len(valid_samples)
    
    print(f"推理样本总数: {len(samples)}")
    print(f"推理成功: {len(valid_samples)}")
    if invalid_count > 0:
        print(f"推理失败（已跳过）: {invalid_count} 个")
    
    # 2. 准备输出目录和文件
    output_judge_dir = os.path.join(output_dir, judge_model, model_name, benchmark_name)
    os.makedirs(output_judge_dir, exist_ok=True)
    
    output_file = os.path.join(output_judge_dir, "raw_eval.json")
    
    # 检查已有评估结果（断点续传）
    existing_eval = {}
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            existing_eval = data.get("samples", {})
        print(f"已评估样本数: {len(existing_eval)}")
    
    # 3. 筛选需要评估的样本
    samples_to_eval = {
        k: v for k, v in valid_samples.items()
        if k not in existing_eval or "error" in existing_eval.get(k, {})
    }
    print(f"待评估样本数: {len(samples_to_eval)}")
    
    # 4. 健康检查
    print(f"检查 Judge API 健康状态...")
    if not check_judge_api_health(judge_model):
        raise RuntimeError(
            f"❌ Judge API 不可用 (模型: {judge_model})！\n"
            f"   请确保 Judge API 正常运行后再进行评估。\n"
            f"   API 地址: {api.base_url}"
        )
    print(f"✓ Judge API 正常")
    
    # 5. 并发评估
    eval_results = {}
    eval_results.update(existing_eval)
    
    error_count = 0
    
    if samples_to_eval:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_qid = {
                executor.submit(
                    evaluate_single_sample,
                    sample_result, judge_model, path_mapping
                ): qid for qid, sample_result in samples_to_eval.items()
            }
            
            for future in tqdm(as_completed(future_to_qid), total=len(samples_to_eval), desc="评估进度"):
                qid = future_to_qid[future]
                
                try:
                    result = future.result()
                    question_id = result.get("question_id", qid)
                    
                    if "error" in result:
                        error_count += 1
                        # 直接打印错误详情和原始返回
                        raw = result.get("raw_response")
                        raw_preview = str(raw)[:800] if raw else "无"
                        print(f"\n❌ [{question_id}] 错误: {result['error']}")
                        print(f"   模型原始返回:\n{raw_preview}")
                        # 不保存失败的数据
                    else:
                        eval_results[question_id] = result
                        
                except Exception as e:
                    error_count += 1
                    print(f"\n❌ [{qid}] 异常: {e}")
            
    # 6. 保存结果（只保存成功的数据）
    output_data = {
        "benchmark": benchmark_name,
        "model_name": model_name,
        "judge_model": judge_model,
        "total_samples": len(samples),
        "evaluated_samples": len(eval_results),
        "timestamp": datetime.now().isoformat(),
        "samples": eval_results
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n评估完成，结果保存至: {output_file}")
    print(f"成功: {len(eval_results)}, 失败: {error_count}")
    
    return output_file
