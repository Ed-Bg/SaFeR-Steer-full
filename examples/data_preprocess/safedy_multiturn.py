# Copyright 2023-2025 SGLang Team
# Copyright Amazon.com, Inc. or its affiliates.
# Copyright 2025 Reallm Labs Ltd. or its affiliates
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Preprocess the SafeDy DyS2k1 dataset to multimodal multiturn parquet format.
"""

import argparse
import os
import threading

from PIL import Image, ImageOps

try:
    import multiprocess.resource_tracker as _resource_tracker
except Exception:
    _resource_tracker = None

import datasets

from verl.utils.hdfs_io import copy, makedirs

if _resource_tracker is not None and not hasattr(threading.RLock(), "_recursion_count"):
    # Work around multiprocess on Python 3.12 where RLock lacks _recursion_count.
    _orig_stop_locked = _resource_tracker.ResourceTracker._stop_locked
    _orig_ensure_running = _resource_tracker.ResourceTracker.ensure_running

    def _stop_locked(self, *args, **kwargs):
        try:
            return _orig_stop_locked(self, *args, **kwargs)
        except AttributeError as exc:
            if "_recursion_count" in str(exc):
                return
            raise

    def _ensure_running(self, *args, **kwargs):
        try:
            return _orig_ensure_running(self, *args, **kwargs)
        except AttributeError as exc:
            if "_recursion_count" in str(exc):
                return
            raise

    _resource_tracker.ResourceTracker._stop_locked = _stop_locked
    _resource_tracker.ResourceTracker.ensure_running = _ensure_running

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default=None, help="The save directory for the preprocessed dataset.")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument(
        "--local_dataset_path",
        default="datasets/MT_RL/data/DyS2k1.jsonl",
        help="The local path to the raw DyS2k1 jsonl dataset.",
    )
    parser.add_argument(
        "--local_save_dir",
        default="data/safedy_multiturn",
        help="The save directory for the preprocessed dataset.",
    )
    parser.add_argument("--data_source", default="safedy/DyS2k1", help="Data source name to store in the dataset.")
    parser.add_argument("--test_size", type=float, default=0.1, help="Test split size ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train/test split.")

    args = parser.parse_args()
    local_dataset_path = os.path.expanduser(args.local_dataset_path) if args.local_dataset_path else None
    data_source = args.data_source
    if local_dataset_path is None:
        raise ValueError("--local_dataset_path is required for DyS2k1 jsonl preprocessing.")

    dataset = datasets.load_dataset("json", data_files=local_dataset_path, split="train")

    if args.test_size and args.test_size > 0:
        split_dataset = dataset.train_test_split(test_size=args.test_size, seed=args.seed, shuffle=True)
        train_dataset = split_dataset["train"]
        test_dataset = split_dataset["test"]
    else:
        train_dataset = dataset
        test_dataset = dataset.select([])

    def _resize_image_to_bounds(image, min_size=386, max_size=512):
        width, height = image.size
        max_side = max(width, height)
        if max_side > max_size:
            scale = max_size / max_side
            new_width = max(1, int(round(width * scale)))
            new_height = max(1, int(round(height * scale)))
            image = image.resize((new_width, new_height), resample=Image.BICUBIC)
            width, height = image.size

        min_side = min(width, height)
        if min_side < min_size:
            pad_w = max(0, min_size - width)
            pad_h = max(0, min_size - height)
            left = pad_w // 2
            right = pad_w - left
            top = pad_h // 2
            bottom = pad_h - top
            image = ImageOps.expand(image, border=(left, top, right, bottom), fill=0)
        return image

    def _load_images(image_paths, min_size=386, max_size=512):
        images = []
        for path in image_paths or []:
            img = Image.open(os.path.expanduser(path)).convert("RGB")
            images.append(_resize_image_to_bounds(img, min_size=min_size, max_size=max_size))
        return images

    system_prompt = """
  You are a helpful AI assistant.
  Please read the user's question and follow the steps below to answer:
  1. Think step-by-step about the reasoning process. Enclose your thoughts between <thinking> and </thinking>.
  2. Then, provide the final answer clearly. Enclose it between <answer> and </answer>.
  
  Format:
  <thinking>
  Your detailed reasoning steps here.
  </thinking>
  <answer>
  Final answer here.
  </answer>
"""

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            def _strip_messages(messages):
                cleaned = []
                for msg in messages or []:
                    if not isinstance(msg, dict):
                        continue
                    role = msg.get("role")
                    content = msg.get("content")
                    if role is None or content is None:
                        continue
                    cleaned.append({"role": role, "content": content})
                return cleaned

            question_msgs = _strip_messages(example.get("question", []))
            # import pdb;pdb.set_trace()
            metadata = example.get("metadata") or {}
            rest_msgs = _strip_messages(metadata.get("rest_messages", []))
            image_paths = example.get("images") or []
            if not image_paths and metadata.get("image"):
                image_paths = [metadata["image"]]
            images = _load_images(image_paths)
            label = example.get("label")
            data = {
                "data_source": data_source,
                "prompt": [{"role": "system", "content": system_prompt}] + question_msgs,
                "images": images,
                "ability": "safety",
                "reward_model": {"style": "rule", "ground_truth": label},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "label": label,
                    "question_id": metadata.get("question_id"),
                    "interaction_kwargs": {
                        "name": "saferdy",
                        "rest_messages": rest_msgs,
                        "feedback_prefix": "Judge feedback:\n",
                    },
                },
                "metadata": metadata,
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True, num_proc=8)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True, num_proc=8)

    hdfs_dir = args.hdfs_dir
    local_save_dir = args.local_dir
    if local_save_dir is not None:
        print("Warning: Argument 'local_dir' is deprecated. Please use 'local_save_dir' instead.")
    else:
        local_save_dir = args.local_save_dir

    local_save_dir = os.path.expanduser(local_save_dir)
    os.makedirs(local_save_dir, exist_ok=True)
    train_dataset.to_parquet(os.path.join(local_save_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_save_dir, "test.parquet"))
    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_save_dir, dst=hdfs_dir)
