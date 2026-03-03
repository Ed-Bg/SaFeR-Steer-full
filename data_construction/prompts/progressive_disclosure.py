#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通用多轮对话分片数据生成器
支持多种数据集格式的适配
"""

import os
import json
import time
import argparse
import concurrent.futures
import base64
import hashlib
import threading
import re
import math
import pandas as pd
from threading import Semaphore
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

from datasets import load_dataset, load_from_disk, Dataset
from PIL import Image
import requests
from tqdm import tqdm

# 全局API并发限制器
_api_semaphore: Semaphore = None


# ============================================================
# 图片处理模块
# ============================================================

def check_and_resize_image(
    image_input: Any, 
    max_pixels: int = 512 * 512, 
    min_pixels: int = 338 * 338
) -> Image.Image:
    """
    检查并调整图片大小。
    
    功能：
        - 图片过大时压缩
        - 图片过小时放大
        - 统一转换为RGB格式
    
    支持的输入类型：
        - 文件路径(str)
        - PIL Image对象
        - bytes/bytearray
        - dict({'bytes': b'...'})
    
    参数：
        image_input: 图片输入
        max_pixels: 最大像素数，超过则压缩
        min_pixels: 最小像素数，小于则放大
    
    返回：
        处理后的PIL Image对象（RGB格式）
    """
    if isinstance(image_input, Image.Image):
        image = image_input.convert("RGB")
    elif isinstance(image_input, (bytes, bytearray)):
        image = Image.open(BytesIO(image_input)).convert("RGB")
    elif isinstance(image_input, dict):
        if 'bytes' in image_input and image_input['bytes'] is not None:
             image = Image.open(BytesIO(image_input['bytes'])).convert("RGB")
        elif 'path' in image_input and image_input['path'] and os.path.exists(image_input['path']):
             image = Image.open(image_input['path']).convert("RGB")
        else:
             # Try other keys or fail gracefully
             raise ValueError(f"无法从字典解析图片: {list(image_input.keys())}")
    elif isinstance(image_input, str):
        if os.path.isfile(image_input):
            image = Image.open(image_input).convert("RGB")
        else:
            try:
                img_bytes = base64.b64decode(image_input)
                image = Image.open(BytesIO(img_bytes)).convert("RGB")
            except Exception:
                raise ValueError(f"无法解析图片输入: {image_input[:50]}...")
    else:
        raise TypeError(f"不支持的图片类型: {type(image_input)}")
    
    image.load()
    pixels = image.width * image.height

    # 压缩大图
    if pixels > max_pixels:
        f = math.sqrt(max_pixels / pixels)
        image = image.resize((max(1, int(image.width * f)), max(1, int(image.height * f))))

    # 放大小图
    pixels = image.width * image.height
    if pixels < min_pixels:
        f = math.sqrt(min_pixels / pixels)
        image = image.resize((max(1, int(image.width * f)), max(1, int(image.height * f))))

    return image


def encode_image_b64jpeg(
    image_input: Any,
    max_pixels: int = 512 * 512,
    min_pixels: int = 338 * 338,
    quality: int = 85
) -> str:
    """
    将图片转为压缩后的base64 JPEG字符串。
    
    参数：
        image_input: 图片输入
        max_pixels: 最大像素数
        min_pixels: 最小像素数
        quality: JPEG压缩质量(1-100)
    
    返回：
        base64编码的JPEG字符串
    """
    img = check_and_resize_image(image_input, max_pixels, min_pixels)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ============================================================
# API调用模块
# ============================================================

def call_chat_completions(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: List[Dict[str, Any]],
    temperature: float,
    max_tokens: int,
    timeout: int = 500,
    max_retries: int = 8,
) -> str:
    """
    调用OpenAI兼容的Chat Completions API。
    
    参数：
        base_url: API基础URL
        api_key: API密钥
        model: 模型名称
        messages: 消息列表
        temperature: 温度参数
        max_tokens: 最大token数
        timeout: 超时时间(秒)
        max_retries: 最大重试次数
    
    返回：
        助手回复内容，失败返回空字符串
    """
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    for attempt in range(1, max_retries + 1):
        try:
            if _api_semaphore:
                _api_semaphore.acquire()
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
            finally:
                if _api_semaphore:
                    _api_semaphore.release()
            
            # 400错误：内容被过滤，不重试
            if resp.status_code == 400:
                print(f"⚠️  400 内容被拒绝: {resp.text[:200]}")
                return ""
            
            # 429限流：等待后重试
            if resp.status_code == 429:
                backoff = int(resp.headers.get("Retry-After", 2 ** attempt))
                print(f"⚠️  429 限流，等待 {backoff}s...")
                time.sleep(backoff)
                continue

            # 其他非200状态码，打印body以便调试
            if resp.status_code != 200:
                print(f"⚠️  API Error {resp.status_code}: {resp.text[:500]}")
            
            resp.raise_for_status()
            data = resp.json()
            
            # 安全获取content
            choices = data.get("choices", [])
            if not choices:
                print(f"⚠️  API返回空choices: {json.dumps(data, ensure_ascii=False)[:300]}")
                return ""
            
            content = choices[0].get("message", {}).get("content")
            if not content:
                # 检查是否有 finish_reason 提示问题
                finish_reason = choices[0].get("finish_reason", "")
                if finish_reason == "content_filter":
                    print(f"⚠️  内容被过滤 (content_filter)")
                elif finish_reason == "length":
                    print(f"⚠️  响应被截断 (length)")
                else:
                    print(f"⚠️  API返回空content, finish_reason={finish_reason}")
            return content if content else ""
            
        except requests.exceptions.RequestException as e:
            if attempt == max_retries:
                print(f"⚠️  API调用失败: {e}")
                return ""
            backoff = min(2 ** attempt, 60)
            print(f"⚠️  API重试 {attempt}/{max_retries} (原因: {e})，等待 {backoff}s...")
            time.sleep(backoff)
    
    return ""


# ============================================================
# 分片模块
# ============================================================

def extract_json_array(text: str) -> List[str]:
    """
    从可能包含markdown或额外内容的文本中提取JSON数组。
    
    参数：
        text: 待解析的文本
    
    返回：
        字符串列表，解析失败返回空列表
    """
    if not text:
        return []

    # 检测拒绝响应（不在这里处理重试，由调用方处理）
    refusal_keywords = ["sorry", "can't assist", "cannot help", "unable to", "I apologize"]
    if any(kw in text.lower() for kw in refusal_keywords):
        print(f"⚠️  检测到拒绝响应: {text[:80]}...")
        return []
    
    # 尝试直接解析
    try:
        arr = json.loads(text)
        if isinstance(arr, list) and all(isinstance(s, str) for s in arr):
            return arr
    except json.JSONDecodeError:
        pass

    # 正则回退：查找[...]块
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            arr = json.loads(match.group())
            if isinstance(arr, list) and all(isinstance(s, str) for s in arr):
                return arr
        except json.JSONDecodeError:
            pass

    print(f"⚠️  JSON解析失败，原始输出: {text[:100]}...")
    return []


def validate_shard_output(shards: List[str], min_shards: int, max_shards: int) -> Dict[str, Any]:
    """
    验证分片输出是否符合要求。
    
    参数：
        shards: 分片列表
        min_shards: 最小分片数
        max_shards: 最大分片数
    
    返回：
        {"valid": bool, "errors": List[str], "warnings": List[str]}
    """
    errors = []
    warnings = []
    
    if not shards:
        errors.append("分片列表为空")
        return {"valid": False, "errors": errors, "warnings": warnings}
    
    if len(shards) < min_shards:
        errors.append(f"分片数量 {len(shards)} 小于最小要求 {min_shards}")
    
    if len(shards) > max_shards:
        warnings.append(f"分片数量 {len(shards)} 超过最大要求 {max_shards}")
    
    for i, shard in enumerate(shards):
        if not shard or not shard.strip():
            errors.append(f"分片 {i+1} 为空")
        elif len(shard) > 500:
            warnings.append(f"分片 {i+1} 过长 ({len(shard)} 字符)")
    
    return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}


def shard_instruction(
    *,
    base_url: str,
    api_key: str,
    sharder_model: str,
    question: str,
    image_b64: str = "",
    min_shards: int,
    max_shards: int,
    max_retries: int = 3,
) -> Tuple[List[str], bool]:
    """
    将单个完整指令分解为多轮分片。
    
    参数：
        base_url: API基础URL
        api_key: API密钥
        sharder_model: 分片模型名称
        question: 原始问题
        image_b64: 图片base64编码（可选）
        min_shards: 最小分片数
        max_shards: 最大分片数
        max_retries: 遇到拒绝响应时的最大重试次数
    
    返回：
        (分片列表, 是否为降级方案)
    """
    system = f"""You are simulating multi-turn user queries for alignment research .

TASK:
Rewrite the given single-turn problem into {min_shards}-{max_shards} sequential user queries (a coherent multi-turn conversation).
This is for preserving general capabilities (math reasoning, logic, coding, reading comprehension), NOT for adversarial attacks.

INPUT:
You will receive either:
(A) Text-only: a question/instruction (and optional reference answer),
or
(B) Image + question: use both if an image exists.

CORE GOAL:
Keep the original task objective unchanged (solve the same problem), but split it into a natural multi-turn user conversation.

MULTI-TURN DESIGN RULES:
1) Output ONLY user messages (first-person perspective).
2) Each turn must be ONE short, natural question or statement.
3) Every turn must be answerable and helpful in context.
4) The conversation must flow coherently: later turns refer to earlier turns.
5) The final turn must ask for the final solution / final answer.

OUTPUT FORMAT (STRICT):
1) Think step-by-step in <think>...</think>.
2) Return ONLY a JSON array of strings in <answer>...</answer>.
3) Do NOT put any comments outside the JSON list.

EXAMPLE:
<think>...</think>
<answer>["I’m working on a math problem.", "Can you help me interpret what it’s asking?", "What’s a good first step?", "Can you verify my intermediate result?", "What is the final answer?"]</answer>
"""
    # 构建消息，支持图文结合
    if image_b64:
        user_content = [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
            {"type": "text", "text": question},
        ]
    else:
        user_content = question

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]
    
    # 尝试调用API，遇到拒绝时重试
    shards = []
    for attempt in range(1, max_retries + 1):
        # 对于 thinking 模型（如 gemini-3-pro-preview），需要更多 token 来容纳推理过程
        # 推理过程可能消耗数百到数千 token，实际输出才能正常生成
        out = call_chat_completions(
            base_url=base_url,
            api_key=api_key,
            model=sharder_model,
            messages=messages,
            temperature=1.0,
            max_tokens=8192,
        )
        
        shards = extract_json_array(out)
        
        # 如果成功获取分片，直接返回
        if shards:
            return shards, False
        
        # 检测是否为拒绝响应
        refusal_keywords = ["sorry", "can't assist", "cannot help", "unable to", "I apologize"]
        is_refusal = any(kw in out.lower() for kw in refusal_keywords) if out else False
        
        if is_refusal and attempt < max_retries:
            print(f"⚠️  检测到拒绝响应，第 {attempt}/{max_retries} 次重试...")
            time.sleep(1)  # 短暂延迟后重试
            continue
        elif not out:
            # API调用完全失败（返回空字符串）
            print(f"⚠️  API调用失败，第 {attempt}/{max_retries} 次")
            if attempt < max_retries:
                time.sleep(2)
                continue
        
        # 其他情况（解析失败但非拒绝）直接退出重试
        break
    
    # 所有尝试都失败：降级为原始问题单轮对话
    if not shards:
        print(f"⚠️  API多次失败或拒绝，降级为原始问题（单轮对话）")
        return [question], True
    
    return shards, False


# def rule_based_shard(question: str, min_shards: int, max_shards: int) -> List[str]:
#     """
#     规则分片：当API被拒时的降级方案。
    
#     参数：
#         question: 原始问题
#         min_shards: 最小分片数
#         max_shards: 最大分片数
    
#     返回：
#         分片列表
#     """
#     # 按句子分割
#     sentences = re.split(r'[.!?。！？]', question)
#     sentences = [s.strip() for s in sentences if s.strip()]
    
#     if len(sentences) >= min_shards:
#         # 合并到目标分片数
#         target = min(len(sentences), max_shards)
#         chunk_size = len(sentences) // target
#         shards = []
#         for i in range(target):
#             start = i * chunk_size
#             end = start + chunk_size if i < target - 1 else len(sentences)
#             shards.append(". ".join(sentences[start:end]))
#         return shards
    
#     # 句子太少，按词分割
#     words = question.split()
#     if len(words) >= min_shards * 3:
#         chunk_size = len(words) // min_shards
#         shards = []
#         for i in range(min_shards):
#             start = i * chunk_size
#             end = start + chunk_size if i < min_shards - 1 else len(words)
#             shards.append(" ".join(words[start:end]))
#         return shards
    
#     # 最后降级：原始问题作为单分片
#     return [question]


# ============================================================
# 对话构建模块
# ============================================================

def build_fixed_messages(
    *,
    image_rel: str,
    original_question: str,
    shards: List[str],
    include_original_as_turn1: bool,
    image_guidance_prefix: str,
) -> List[Dict[str, Any]]:
    """
    构建固定格式的多轮对话消息。
    
    对话结构：
      - system: 系统提示
      - user turn 1: 图片 + 文本（第一轮问题）
      - user turn 2..N: 纯文本（分片引导）
    
    参数：
        image_rel: 图片的base64字符串
        original_question: 原始问题
        shards: 分片列表
        include_original_as_turn1: 第一轮是否使用原始问题
        image_guidance_prefix: 最后一轮的图片引导前缀
    
    返回：
        消息列表
    """
    # 系统消息
    # system_msg = {
    #     "role": "system",
    #     "content": (
    #         "You are a helpful AI assistant. "
    #         "Follow the user instructions. "
    #         "If the request is unsafe, refuse and provide safe alternatives."
    #     ),
    # }

    messages: List[Dict[str, Any]] = []

    # 第一轮：图片 + 文本
    if include_original_as_turn1:
        first_text = original_question
    else:
        first_text = shards[0] if shards else original_question

    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_rel},
                {"type": "text", "text": first_text},
            ],
        }
    )

    # 后续轮次：纯文本分片引导
    for i in range(1, len(shards)):
        shard_text = shards[i]
        
        # 最后一轮添加图片引导前缀
        if i == len(shards) - 1 and image_guidance_prefix:
            shard_text = image_guidance_prefix + shard_text
        
        messages.append(
            {
                "role": "user",
                "content": [{"type": "text", "text": shard_text}],
            }
        )

    return messages


# ============================================================
# IO模块
# ============================================================

def load_data(
    dataset_type: str,
    hf_path: str,
    hf_name: str,
    hf_split: str,
    csv_path: str = "",
    image_dir: str = ""
) -> List[Dict[str, Any]]:
    """
    加载数据集（支持多种格式）。
    
    参数：
        dataset_type: 数据集类型 (huggingface/csv)
        hf_path: HuggingFace数据集路径
        hf_name: 数据集子集名称
        hf_split: 数据集split名称
        csv_path: CSV文件路径（仅用于CSV类型）
        image_dir: 图片目录（仅用于CSV类型）
    
    返回：
        数据列表
    """
    if dataset_type == "huggingface":
        try:
            if hf_name:
                ds = load_dataset(hf_path, hf_name, split=hf_split)
            else:
                ds = load_dataset(hf_path, split=hf_split)
            return list(ds)
        except Exception as e:
            try:
                ds = load_from_disk(hf_path)
                return list(ds)
            except Exception:
                raise RuntimeError(f"加载HuggingFace数据集失败: {e}")
    
    elif dataset_type == "csv":
        try:
            df = pd.read_csv(csv_path)
            data = []
            for _, row in df.iterrows():
                item = row.to_dict()
                # 加载图片
                if 'image_path' in item and image_dir:
                    img_path = os.path.join(image_dir, item['image_path'])
                    if os.path.exists(img_path):
                        item['image'] = img_path
                data.append(item)
            return data
        except Exception as e:
            raise RuntimeError(f"加载CSV数据集失败: {e}")

    elif dataset_type == "json" or dataset_type == "jsonl":
        # Using csv_path as a generic file path argument if hf_path is not suitable, 
        # but sticking to existing pattern, maybe reuse csv_path or hf_path?
        # Let's use hf_path if provided, or csv_path. 
        # Ideally parsing arguments should be cleaner, but for now:
        target_path = hf_path if hf_path and os.path.exists(hf_path) else csv_path
        if not target_path:
             raise ValueError("JSON/JSONL dataset requires a valid path in --hf_path or --csv_path")
             
        try:
            return read_jsonl(target_path)
        except Exception as e:
            raise RuntimeError(f"加载JSONL数据集失败: {e}")
    
    else:
        raise ValueError(f"不支持的数据集类型: {dataset_type}")


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    """读取JSONL文件。"""
    rows: List[Dict[str, Any]] = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def append_jsonl(path: str, row: Dict[str, Any]) -> None:
    """追加单条记录到JSONL文件（线程安全）。"""
    dir_path = os.path.dirname(path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ============================================================
# 字段映射模块
# ============================================================

def map_fields(data_item: Dict[str, Any], field_mapping: Dict[str, str]) -> Dict[str, Any]:
    """
    根据字段映射配置提取数据。
    
    参数：
        data_item: 原始数据项
        field_mapping: 字段映射配置，格式：{"target_field": "source_field"}
    
    返回：
        映射后的数据字典
    """
    result = {}
    for target_field, source_field in field_mapping.items():
        if source_field in data_item:
            result[target_field] = data_item[source_field]
        else:
            result[target_field] = ""  # 默认空值
    return result


# ============================================================
# 主函数
# ============================================================

def main():
    """主函数：解析参数并执行数据处理流程。"""
    parser = argparse.ArgumentParser(
        description="通用多轮对话分片数据生成器",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # 数据集配置
    dataset_group = parser.add_argument_group("数据集配置")
    dataset_group.add_argument("--dataset_type", type=str, required=True, choices=["huggingface", "csv", "json", "jsonl"])
    dataset_group.add_argument("--hf_path", type=str, default="")
    dataset_group.add_argument("--hf_name", type=str, default="")
    dataset_group.add_argument("--hf_split", type=str, default="train")
    dataset_group.add_argument("--csv_path", type=str, default="/home/maloqaily/data/datasets/JailBreakV-28k/JailBreakV_28K/JailBreakV_28K.csv")
    dataset_group.add_argument("--image_dir", type=str, default="")
    dataset_group.add_argument("--out_dir", type=str, required=True)
    dataset_group.add_argument("--output_suffix", type=str, default="")
    
    # 字段映射（JSON格式）
    dataset_group.add_argument(
        "--field_mapping",
        type=str,
        required=True,
        help='JSON格式的字段映射，例如：\'{"question":"question","image":"image","response":"chosen","category":"category"}\''
    )

    # 分片配置
    shard_group = parser.add_argument_group("分片配置")
    shard_group.add_argument("--min_shards", type=int, default=2)
    shard_group.add_argument("--max_shards", type=int, default=10)
    shard_group.add_argument("--sharder_model", type=str, default="gpt-4.1")

    # API配置
    api_group = parser.add_argument_group("API配置")
    api_group.add_argument("--base_url", type=str, default="https://xiaoai.plus/v1")
    api_group.add_argument("--api_key", type=str, default="")
    api_group.add_argument("--api_key_env", type=str, default="XIAOAI_API_KEY")

    # 图片处理配置
    image_group = parser.add_argument_group("图片处理配置")
    image_group.add_argument("--max_pixels", type=int, default=512*512)
    image_group.add_argument("--min_pixels", type=int, default=338*338)
    image_group.add_argument("--image_quality", type=int, default=100)

    # 对话构建配置
    dialogue_group = parser.add_argument_group("对话构建配置")
    dialogue_group.add_argument("--include_original_as_turn1", action="store_true")
    dialogue_group.add_argument(
        "--image_guidance_prefix",
        type=str,
        default="Based on the clues provided in the image, "
    )

    # 并发控制
    parallel_group = parser.add_argument_group("并发控制")
    parallel_group.add_argument("--max_workers", type=int, default=512)
    parallel_group.add_argument("--api_concurrent", type=int, default=100)

    # 其他配置
    other_group = parser.add_argument_group("其他配置")
    other_group.add_argument("--resume_jsonl", type=str, default="")

    args = parser.parse_args()

    # 解析API密钥
    api_key = args.api_key.strip()
    if not api_key:
        api_key = os.environ.get(args.api_key_env, "").strip()
    if not api_key:
        raise RuntimeError(f"缺少API密钥。请使用 --api_key 或设置环境变量 {args.api_key_env}")

    # 初始化API并发限制
    global _api_semaphore
    _api_semaphore = Semaphore(args.api_concurrent)
    print(f"🔧 API并发限制: {args.api_concurrent}")

    # 解析字段映射
    try:
        field_mapping = json.loads(args.field_mapping)
    except Exception as e:
        raise ValueError(f"字段映射JSON解析失败: {e}")

    # 加载数据
    print(f"📖 加载数据集...")
    data = load_data(
        dataset_type=args.dataset_type,
        hf_path=args.hf_path,
        hf_name=args.hf_name,
        hf_split=args.hf_split,
        csv_path=args.csv_path,
        image_dir=args.image_dir
    )
    print(f"✅ 加载完成，共 {len(data)} 条数据")

    # 输出路径配置
    os.makedirs(args.out_dir, exist_ok=True)
    
    # 构建输出文件名
    name_parts = ["multiturn"]
    if args.include_original_as_turn1:
        name_parts.append("originalTurn1")
    name_parts.append(f"{args.min_shards}-{args.max_shards}")
    name_parts.append(args.sharder_model.replace("/", "_"))
    if args.output_suffix:
        name_parts.append(args.output_suffix)
    
    out_name = "_".join(name_parts) + ".jsonl"
    out_path = os.path.join(args.out_dir, out_name)
    err_path = out_path + ".errors"

    # 断点续传
    done_ids = set()
    resume_path = args.resume_jsonl if args.resume_jsonl else out_path
    if os.path.exists(resume_path):
        print(f"📖 检测到已有输出: {resume_path}")
        try:
            existing_rows = read_jsonl(resume_path)
            for r in existing_rows:
                qid = r.get("question_id", None)
                if qid is not None:
                    done_ids.add(str(qid))
            print(f"   跳过 {len(done_ids)} 条已处理数据")
        except Exception as e:
            print(f"⚠️  警告：读取断点文件失败: {e}")
    else:
        print(f"🆕 开始全新处理: {out_path}")

    # 辅助函数
    def make_question_id(question: str, img_bytes: bytes, idx: int) -> str:
        """生成唯一question_id"""
        q_norm = (question or "").strip().encode("utf-8")
        # 加上idx确保顺序索引也能区分
        combined = q_norm + (img_bytes if img_bytes else b"") + str(idx).encode("utf-8")
        return hashlib.sha1(combined).hexdigest()

    def pil_to_bytes(img: Image.Image, format: str = "PNG") -> bytes:
        """将PIL Image转为bytes"""
        buffer = BytesIO()
        img.save(buffer, format=format)
        return buffer.getvalue()

    def to_base64_str(img_val: Any) -> Tuple[str, bytes]:
        """将图片转为base64字符串并返回原始字节"""
        try:
            if isinstance(img_val, Image.Image):
                img_bytes = pil_to_bytes(img_val)
                b64_str = base64.b64encode(img_bytes).decode("utf-8")
                return b64_str, img_bytes

            if isinstance(img_val, (bytes, bytearray)):
                img_bytes = bytes(img_val)
                b64_str = base64.b64encode(img_bytes).decode("utf-8")
                return b64_str, img_bytes

            if isinstance(img_val, dict):
                if 'bytes' in img_val and img_val['bytes'] is not None:
                     img_bytes = bytes(img_val['bytes'])
                     b64_str = base64.b64encode(img_bytes).decode("utf-8")
                     return b64_str, img_bytes
                
                if 'path' in img_val and img_val['path']:
                     img_path = img_val['path']
                     if os.path.exists(img_path):
                        with open(img_path, "rb") as f:
                            img_bytes = f.read()
                        b64_str = base64.b64encode(img_bytes).decode("utf-8")
                        return b64_str, img_bytes

            if isinstance(img_val, str):
                if os.path.isfile(img_val):
                    with open(img_val, "rb") as f:
                        img_bytes = f.read()
                    b64_str = base64.b64encode(img_bytes).decode("utf-8")
                    return b64_str, img_bytes
                try:
                    decoded = base64.b64decode(img_val, validate=True)
                    return img_val, decoded
                except Exception:
                    pass
                img_bytes = img_val.encode("utf-8")
                b64_str = base64.b64encode(img_bytes).decode("utf-8")
                return b64_str, img_bytes

        except Exception as e:
            print(f"⚠️  图片转换失败: {e}")
        return "", b""

    # 写入锁
    write_lock = threading.Lock()

    def process_one(idx: int, d: Dict[str, Any]) -> Dict[str, Any]:
        """处理单条数据"""
        # 字段映射
        mapped = map_fields(d, field_mapping)
        
        original_question = mapped.get("question", "")
        category = mapped.get("category", "")
        original_response = mapped.get("response", "")
        image_val = mapped.get("image", None)
        
        # 图片处理,可能会出错
        image_b64, image_bytes = to_base64_str(image_val)
        
        # 为分片API生成压缩图片
        image_b64_compressed = ""
        if image_val is not None:
            try:
                image_b64_compressed = encode_image_b64jpeg(
                    image_val,
                    max_pixels=args.max_pixels,
                    min_pixels=args.min_pixels,
                    quality=args.image_quality
                )
            except Exception as e:
                print(f"⚠️  [{idx}] 图片压缩失败: {e}")
        
        # 生成question_id
        qid = make_question_id(original_question, image_bytes, idx)
        
        # 检查是否已处理
        if str(qid) in done_ids:
            return {"_skipped": True, "question_id": qid}

        # 分片
        shards, is_fallback = shard_instruction(
            base_url=args.base_url,
            api_key=api_key,
            sharder_model=args.sharder_model,
            question=original_question,
            image_b64=image_b64_compressed,
            min_shards=args.min_shards,
            max_shards=args.max_shards,
        )
        
        # 验证（非降级时）
        if not is_fallback:
            validation = validate_shard_output(shards, args.min_shards, args.max_shards)
            if not validation["valid"]:
                print(f"⚠️  [{idx}] 分片验证失败: {validation['errors']}")

        # 构建消息
        messages_fixed = build_fixed_messages(
            image_rel=image_b64,
            original_question=original_question,
            shards=shards,
            include_original_as_turn1=bool(args.include_original_as_turn1),
            image_guidance_prefix=args.image_guidance_prefix,
        )

        result = {
            "question_id": qid,
            "image": image_b64,
            "original_question": original_question,
            "shards": shards,
            "shards_num": len(shards),
            "is_fallback": is_fallback,
            "original_response": original_response,
            "category": category,
            "min_shards": int(args.min_shards),
            "max_shards": int(args.max_shards),
            "sharder_model": args.sharder_model,
            "messages_fixed": messages_fixed,
        }

        # 边运行边保存
        with write_lock:
            append_jsonl(out_path, result)
            done_ids.add(str(qid))

        return result

    # 统计
    success_count = 0
    error_count = 0
    skip_count = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futs: List[Tuple[int, concurrent.futures.Future]] = []
        for i, d in enumerate(data):
            futs.append((i, ex.submit(process_one, i, d)))

        for i, fut in tqdm(futs, total=len(futs), desc="处理中"):
            try:
                result = fut.result()
                if result.get("_skipped"):
                    skip_count += 1
                else:
                    success_count += 1
            except Exception as e:
                error_count += 1
                d = data[i]
                error_row = {
                    "question_id": None,
                    "original_question": d.get("question", ""),
                    "error": str(e),
                }
                with write_lock:
                    append_jsonl(err_path, error_row)
                print(f"❌ 处理失败 [{i}]: {e}")

    # 打印统计
    print(f"\n{'='*60}")
    print(f"✅ 处理完成！")
    print(f"{'='*60}")
    print(f"  ✅ 成功:  {success_count}")
    print(f"  ⏭️  跳过:  {skip_count}")
    print(f"  ❌ 错误:  {error_count}")
    print(f"  📁 输出:  {out_path}")
    if error_count > 0:
        print(f"  ⚠️  错误日志: {err_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
