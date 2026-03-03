#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通用数据集多轮SFT对话生成器

从通用评估数据集（MMStar, ScienceQA等）中采样数据，
将单轮QA转换为多轮对话格式，用于多轮对话安全SFT训练中的通用能力保持。

特点：
- 生成完整的多轮对话（用户问题 + 助手回答）
- 保证最终答案的正确性
- 中间轮次包含推理过程
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
import random
import pandas as pd
from threading import Semaphore
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image
import requests
from tqdm import tqdm

# 全局API并发限制器
_api_semaphore: Semaphore = None


# ============================================================
# 数据集配置
# ============================================================

DATASET_CONFIGS = {
    "geo3k": {
        "path": "data/",
        "format": "parquet",
        "field_mapping": {
            "extra_info": "extra_info",
            "images": "images",
        },
    },
    "Align-Anything": {
        "path": "data/",
        "format": "parquet",
        "field_mapping": {
            "question": "question",
            "image": "image",
            "response_1": "response_1",  # 这是索引，需要结合choices使用
            "response_2": "response_2",  # 这是索引，需要结合choices使用
            "overall_response": "overall_response",
        },
    },
    "ScienceQA": {
        "path": "eval/general_dataset/ScienceQA/data/train-00000-of-00001-1028f23e353fbe3e.parquet",
        "format": "parquet",
        "field_mapping": {
            "question": "question",
            "image": "image",
            "answer": "answer",  # 这是索引，需要结合choices使用
            "choices": "choices",
            "category": "category",
        },
    },
}


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
    """
    img = check_and_resize_image(image_input, max_pixels, min_pixels)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def to_base64_str(img_val: Any) -> Tuple[str, bytes]:
    """将图片转为base64字符串并返回原始字节"""
    try:
        if img_val is None:
            return "", b""
            
        if isinstance(img_val, Image.Image):
            buffer = BytesIO()
            img_val.save(buffer, format="PNG")
            img_bytes = buffer.getvalue()
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

    except Exception as e:
        print(f"⚠️  图片转换失败: {e}")
    return "", b""


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
            
            if resp.status_code == 400:
                print(f"⚠️  400 内容被拒绝: {resp.text[:200]}")
                return ""
            
            if resp.status_code == 429:
                backoff = int(resp.headers.get("Retry-After", 2 ** attempt))
                print(f"⚠️  429 限流，等待 {backoff}s...{resp.text[:500]}")
                time.sleep(backoff)
                continue

            if resp.status_code != 200:
                print(f"⚠️  API Error {resp.status_code}: {resp.text[:500]}")
            
            resp.raise_for_status()
            data = resp.json()
            
            choices = data.get("choices", [])
            if not choices:
                print(f"⚠️  API返回空choices: {json.dumps(data, ensure_ascii=False)[:300]}")
                return ""
            
            content = choices[0].get("message", {}).get("content")
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
# 多轮对话生成模块
# ============================================================

def extract_json_from_response(text: str) -> List[Dict[str, str]]:
    """
    从响应中提取JSON数组。
    """
    if not text:
        return []

    # 检测拒绝响应
    # refusal_keywords = ["sorry", "can't assist", "cannot help", "unable to", "I apologize"]
    # if any(kw in text.lower() for kw in refusal_keywords):
    #     print(f"⚠️  检测到拒绝响应: {text[:80]}...")
    #     return []
    
    # 尝试直接解析
    try:
        arr = json.loads(text)
        if isinstance(arr, list) and all(isinstance(item, dict) for item in arr):
            return arr
    except json.JSONDecodeError:
        pass

    # 正则回退：查找[...]块
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            arr = json.loads(match.group())
            if isinstance(arr, list) and all(isinstance(item, dict) for item in arr):
                return arr
        except json.JSONDecodeError:
            pass

    print(f"⚠️  JSON解析失败，原始输出: {text[:200]}...")
    return []


def generate_multiturn_dialogue(
    *,
    base_url: str,
    api_key: str,
    model: str,
    question: str,
    answer: str,
    image_b64: str = "",
    min_turns: int = 2,
    max_turns: int = 6,
    max_retries: int = 3,
) -> Tuple[List[Dict[str, str]], bool]:
    """
    将单轮QA转换为多轮对话，包含用户问题和助手回答。
    
    参数：
        question: 原始问题
        answer: 正确答案
        image_b64: 图片base64编码（可选）
        min_turns: 最小轮数
        max_turns: 最大轮数
    
    返回：
        (对话列表, 是否为降级方案)
        对话列表格式: [{"user": "...", "assistant": "..."}, ...]
    """
    system_prompt = f"""You are a dataset engineer specialized in creating high-quality multi-turn dialogue data for SFT (Supervised Fine-Tuning).
Your job is to convert a SINGLE-TURN QA sample into a NATURAL multi-turn tutoring conversation.

GOAL
Turn the original question into a {min_turns}-{max_turns} turn conversation (each turn includes one user message and one assistant message),
ending with a final assistant answer that matches the reference answer exactly.

INPUT
- Original Question: the problem to solve
- Reference Answer: the target final answer (ground truth)
- Image: optional visual context (may be empty)

HARD CONSTRAINTS (MUST FOLLOW)
1. Output exactly {min_turns}-{max_turns} turns.
2. Each turn must be an object with BOTH fields: "user" and "assistant".
3. The conversation must be coherent, progressive, and educational.
4. The assistant must explain reasoning step-by-step (tutoring style), but do not add irrelevant fluff.
5. The FINAL assistant response must contain the final answer consistent with the Reference Answer (except A/B/C/D).
6. The final answer must be equivalent to the Reference Answer, and should be written in the same form whenever possible
7. Do NOT change the math/science problem meaning. Do NOT introduce extra assumptions.
8. The assistant response should contains reasoning steps and the final answer
9. The reasoning steps should be wrapped in `<thinking></thinking>` tag, and wrap the final answer in `<answer></answer>` tag.
10. The reasoning steps should follow the structures below.

THINK STEPS (REQUIRED)
Step 1: Summarize the request and key image/context signals (brief).
Step 2: Analyse the question.
Step 3: Draft the response.

ABSOLUTE RULE: NO MULTIPLE-CHOICE LETTER OUTPUT
- NEVER output choice letters such as A, B, C, D as the final answer.
- NEVER write phrases like "The answer is A/B/C/D", "Option B", or "Answer: B", even if the original dataset is multiple-choice.
- ALWAYS output the actual choice label (e.g., a number, unit, expression, or short phrase), not a choice label.

DIALOGUE DESIGN (HOW TO EXPAND ONE QA INTO MULTI-TURN)
- Make the user behave like a student: asking clarification, requesting steps, checking intermediate results.
- Make the assistant guide the solution progressively:
  - interpret the question and identify givens/unknowns
  - choose formulas / principles
  - compute intermediate steps (show calculations)
  - verify units / reasoning
  - conclude with the final answer
- Avoid repetition: each turn must contribute NEW progress.

MATH & SCIQA RULES
- Keep symbols and variable definitions consistent across turns.
- If the reference answer includes units, the final answer must include the same units.
- If the reference answer is a short phrase, keep the final answer concise and aligned.
- If an image is provided, only use information that can be inferred from it, and do not hallucinate details.

OUTPUT FORMAT (STRICT JSON)
Return ONLY a JSON array (no markdown, no extra commentary):
[
  {{"user": "...", "assistant": "<thinking>...</thinking><answer>...</answer>"}},
  {{"user": "...", "assistant": "<thinking>...</thinking><answer>...</answer>"}},
  ...
  {{"user": "...", "assistant": "<thinking>...</thinking><answer>...(final answer)...</answer>"}}
]

FINAL ANSWER REQUIREMENT (VERY IMPORTANT)
- The final assistant message must explicitly present the final answer.
- The final answer must match the Reference Answer exactly or equivalently.
"""

    user_content_parts = []
    
    # 添加图片（如果有）
    if image_b64:
        user_content_parts.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}
        })
    
    # 添加问题和答案
    user_text = f"""Original Question:
{question}

Reference Answer (MUST be preserved in final response):
{answer}

Please generate a {min_turns}-{max_turns} turn dialogue that naturally leads to this answer with proper reasoning."""
    
    user_content_parts.append({"type": "text", "text": user_text})

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content_parts if image_b64 else user_text},
    ]
    
    # 尝试调用API
    for attempt in range(1, max_retries + 1):
        out = call_chat_completions(
            base_url=base_url,
            api_key=api_key,
            model=model,
            messages=messages,
            temperature=0.7,
            max_tokens=4096,
        )
        
        dialogue = extract_json_from_response(out)
        
        # 验证对话格式
        if dialogue and len(dialogue) >= min_turns:
            # 检查每轮是否都有user和assistant
            valid = all(
                isinstance(turn, dict) and "user" in turn and "assistant" in turn
                for turn in dialogue
            )
            if valid:
                return dialogue, False
        
        if attempt < max_retries:
            print(f"⚠️  对话生成失败，第 {attempt}/{max_retries} 次重试...")
            time.sleep(1)
    
    # 降级：返回原始单轮对话
    print(f"⚠️  多轮生成失败，降级为原始单轮对话")
    return [{"user": question, "assistant": answer}], True


def build_conversation_messages(
    *,
    image_b64: str,
    dialogue: List[Dict[str, str]],
) -> List[Dict[str, Any]]:
    """
    构建最终的对话消息格式。
    
    返回格式适配SFT训练：
    [
        {"role": "user", "content": [{"type": "image", "image": "..."}, {"type": "text", "text": "..."}]},
        {"role": "assistant", "content": "..."},
        {"role": "user", "content": [{"type": "text", "text": "..."}]},
        {"role": "assistant", "content": "..."},
        ...
    ]
    """
    messages: List[Dict[str, Any]] = []
    
    for i, turn in enumerate(dialogue):
        user_text = turn.get("user", "")
        assistant_text = turn.get("assistant", "")
        
        # 第一轮包含图片
        if i == 0 and image_b64:
            messages.append({
                "role": "user",
                "content": [
                    {"type": "image", "image": image_b64},
                    {"type": "text", "text": user_text},
                ],
            })
        else:
            messages.append({
                "role": "user",
                "content": [{"type": "text", "text": user_text}],
            })
        
        messages.append({
            "role": "assistant",
            "content": assistant_text,
        })
    
    return messages


# ============================================================
# 数据加载模块
# ============================================================

def load_dataset(config: Dict[str, Any], base_dir: str) -> List[Dict[str, Any]]:
    """
    根据配置加载数据集。
    """
    path = os.path.join(base_dir, config["path"])
    format_type = config["format"]
    field_mapping = config["field_mapping"]
    
    if format_type == "parquet":
        df = pd.read_parquet(path)
        data = []
        for _, row in df.iterrows():
            item = {}
            for target_field, source_field in field_mapping.items():
                if source_field in row:
                    item[target_field] = row[source_field]
                else:
                    item[target_field] = None
            data.append(item)
        return data
    else:
        raise ValueError(f"不支持的格式: {format_type}")


def process_answer(item: Dict[str, Any], source: str) -> str:
    """
    处理不同数据集的答案格式。
    """
    if source == "ScienceQA":
        # ScienceQA的answer是索引，需要结合choices
        answer_idx = item.get("answer")
        choices = item.get("choices")
        if choices is not None and answer_idx is not None:
            try:
                if hasattr(choices, 'tolist'):
                    choices = choices.tolist()
                return str(choices[int(answer_idx)])
            except (IndexError, TypeError, ValueError):
                return str(answer_idx)
        return str(answer_idx) if answer_idx is not None else ""
    elif source == "Align-Anything":
        a = item.get(f"response_{item.get('overall_response')}")
        return a
    elif source == "geo3k":
        a = item.get("extra_info").get("answer")
        return a
    else:
        # MMStar等直接返回answer
        answer = item.get("answer", "")
        return str(answer) if answer is not None else ""

def process_question(item, source):
    if source == "geo3k":
        q = item.get("extra_info").get("question").replace("<image>", "")
        return q
    else:
        return item.get("question", "")

def process_image(item, source):
    if source == "geo3k":
        image = item.get("images")[0]
        return image
    else:
        return item.get("image", None)


# ============================================================
# IO模块
# ============================================================

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
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="通用数据集多轮SFT对话生成器",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # 采样配置
    parser.add_argument("--align_anything_samples", type=int, default=500, help="从align_anything采样的数量")
    parser.add_argument("--scienceqa_samples", type=int, default=250, help="从ScienceQA采样的数量")
    parser.add_argument("--geo3k_samples", type=int, default=250, help="从geo3k采样的数量")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    
    # 输出配置
    parser.add_argument("--base_dir", type=str, default="data/", help="项目根目录")
    parser.add_argument("--out_dir", type=str, default="question_shard_kit/outputs/general_multiturn", help="输出目录")
    
    # 分片配置
    parser.add_argument("--min_turns", type=int, default=2, help="最小对话轮数")
    parser.add_argument("--max_turns", type=int, default=6, help="最大对话轮数")
    parser.add_argument("--sharder_model", type=str, default="gpt-4.1", help="分片模型")

    # API配置（本地Qwen3-VL-32B）
    parser.add_argument("--base_url", type=str, default="http://149.88.89.156:3002/v1", help="API基础URL")
    parser.add_argument("--api_key", type=str, default="sk-Ws8IkLMiHz6a8acXbcAslbn8TCJkogtO2Ra7T8Y3gsGH4NxP", help="API密钥")

    # 图片处理配置
    parser.add_argument("--max_pixels", type=int, default=512*512)
    parser.add_argument("--min_pixels", type=int, default=338*338)
    parser.add_argument("--image_quality", type=int, default=85)

    # 并发控制
    parser.add_argument("--max_workers", type=int, default=64, help="最大并发工作线程")
    parser.add_argument("--api_concurrent", type=int, default=64, help="API并发限制")

    args = parser.parse_args()

    # 设置随机种子
    random.seed(args.seed)

    # 初始化API并发限制
    global _api_semaphore
    _api_semaphore = Semaphore(args.api_concurrent)
    print(f"🔧 API并发限制: {args.api_concurrent}")
    print(f"🔧 本地模型端点: {args.base_url}")

    # 输出路径
    out_dir = os.path.join(args.base_dir, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    
    out_name = f"general_multiturn_{args.min_turns}-{args.max_turns}turns_{args.align_anything_samples}align_anything_{args.geo3k_samples}geo3k_{args.scienceqa_samples}scienceqa.jsonl"
    out_path = os.path.join(out_dir, out_name)
    err_path = out_path + ".errors"

    # 加载数据集
    print(f"\n📖 加载数据集...")
    
    align_anything_data = load_dataset(DATASET_CONFIGS["Align-Anything"], args.base_dir)
    print(f"  ✅ Align-Anything: {len(align_anything_data)} 条")

    geo3k_data = load_dataset(DATASET_CONFIGS["geo3k"], args.base_dir)
    print(f"  ✅ geo3k: {len(geo3k_data)} 条")
    
    scienceqa_data = load_dataset(DATASET_CONFIGS["ScienceQA"], args.base_dir)
    # 过滤掉没有图片的ScienceQA数据
    scienceqa_data = [d for d in scienceqa_data if d.get("image") is not None]
    print(f"  ✅ ScienceQA (有图片): {len(scienceqa_data)} 条")

    # 采样
    print(f"\n🎲 采样数据...")
    
    align_anything_samples = min(args.align_anything_samples, len(align_anything_data))
    geo3k_samples = min(args.geo3k_samples, len(geo3k_data))
    scienceqa_samples = min(args.scienceqa_samples, len(scienceqa_data))
    
    sampled_align_anything = random.sample(align_anything_data, align_anything_samples)
    sampled_geo3k = random.sample(geo3k_data, geo3k_samples)
    sampled_scienceqa = random.sample(scienceqa_data, scienceqa_samples)
    
    # 添加来源标记
    for item in sampled_align_anything:
        item["_source"] = "Align-Anything"
    for item in sampled_geo3k:
        item["_source"] = "geo3k"
    for item in sampled_scienceqa:
        item["_source"] = "ScienceQA"
    
    all_samples = sampled_align_anything + sampled_geo3k + sampled_scienceqa
    random.shuffle(all_samples)
    
    print(f"  ✅ 总计: {len(all_samples)} 条")

    # 断点续传
    done_ids = set()
    if os.path.exists(out_path):
        print(f"\n📖 检测到已有输出: {out_path}")
        try:
            existing_rows = read_jsonl(out_path)
            for r in existing_rows:
                qid = r.get("question_id", None)
                if qid is not None:
                    done_ids.add(str(qid))
            print(f"   跳过 {len(done_ids)} 条已处理数据")
        except Exception as e:
            print(f"⚠️  警告：读取断点文件失败: {e}")
    else:
        print(f"\n🆕 开始全新处理: {out_path}")

    # 辅助函数
    def make_question_id(question: str, img_bytes: bytes, idx: int) -> str:
        q_norm = (question or "").strip().encode("utf-8")
        combined = q_norm + (img_bytes if img_bytes else b"") + str(idx).encode("utf-8")
        return hashlib.sha1(combined).hexdigest()

    # 写入锁
    write_lock = threading.Lock()

    def process_one(idx: int, item: Dict[str, Any]) -> Dict[str, Any]:
        """处理单条数据"""
        source = item.get("_source", "unknown")
        question = process_question(item, source)
        answer = process_answer(item, source)
        category = item.get("category", "")
        image_val = process_image(item, source)
        
        # 图片处理
        image_b64, image_bytes = to_base64_str(image_val)
        
        # 为API生成压缩图片
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
        qid = make_question_id(question, image_bytes, idx)
        
        # 检查是否已处理
        if str(qid) in done_ids:
            return {"_skipped": True, "question_id": qid}

        # 生成多轮对话
        dialogue, is_fallback = generate_multiturn_dialogue(
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.sharder_model,
            question=question,
            answer=answer,
            image_b64=image_b64_compressed,
            min_turns=args.min_turns,
            max_turns=args.max_turns,
        )

        # 构建最终消息格式
        conversations = build_conversation_messages(
            image_b64=image_b64,
            dialogue=dialogue,
        )

        result = {
            "question_id": qid,
            "image": image_b64,
            "original_question": question,
            "original_answer": answer,
            "source": source,
            "category": category,
            "dialogue": dialogue,
            "turns": len(dialogue),
            "is_fallback": is_fallback,
            "conversations": conversations,
            "min_turns": args.min_turns,
            "max_turns": args.max_turns,
            "sharder_model": args.sharder_model,
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
    fallback_count = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futs: List[Tuple[int, concurrent.futures.Future]] = []
        for i, item in enumerate(all_samples):
            futs.append((i, ex.submit(process_one, i, item)))

        for i, fut in tqdm(futs, total=len(futs), desc="生成多轮对话"):
            try:
                result = fut.result()
                if result.get("_skipped"):
                    skip_count += 1
                else:
                    success_count += 1
                    if result.get("is_fallback"):
                        fallback_count += 1
            except Exception as e:
                error_count += 1
                item = all_samples[i]
                error_row = {
                    "question_id": None,
                    "original_question": item.get("question", ""),
                    "error": str(e),
                }
                with write_lock:
                    append_jsonl(err_path, error_row)
                print(f"❌ 处理失败 [{i}]: {e}")

    # 打印统计
    print(f"\n{'='*60}")
    print(f"✅ 处理完成！")
    print(f"{'='*60}")
    print(f"  ✅ 成功:      {success_count}")
    print(f"  ⏭️  跳过:      {skip_count}")
    print(f"  ⚠️  降级:      {fallback_count}")
    print(f"  ❌ 错误:      {error_count}")
    print(f"  📁 输出:      {out_path}")
    if error_count > 0:
        print(f"  ⚠️  错误日志:  {err_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
