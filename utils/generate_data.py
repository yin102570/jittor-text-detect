import os
import json
import random
import datasets
import numpy as np
import math
from tqdm import tqdm
from multiprocessing.pool import ThreadPool
import jittor as jt

# 兼容原代码的导入（确保路径正确）
from . import custom_datasets
from .load_models_tokenizers import load_base_model, load_mask_filling_model

# 辅助函数：与原PyTorch代码完全一致
def drop_last_word(text):
    return ' '.join(text.split()[:-1])

def truncate_to_substring(text, substring, occurrence=1):
    parts = text.split(substring)
    if len(parts) <= occurrence + 1:
        return text
    return substring.join(parts[:occurrence + 1])

def trim_to_shorter_length(text1, text2):
    min_len = min(len(text1.split()), len(text2.split()))
    text1_trimmed = ' '.join(text1.split()[:min_len])
    text2_trimmed = ' '.join(text2.split()[:min_len])
    return text1_trimmed, text2_trimmed

def sample_from_model(args, config, batch_data, min_words):
    """
    从Jittor模型采样文本（与原PyTorch逻辑一致）
    """
    base_model = config["base_model"]
    base_tokenizer = config["base_tokenizer"]
    sampled_texts = []
    for data in batch_data:
        # 分词
        inputs = base_tokenizer(
            data,
            return_tensors="jt",
            truncation=True,
            max_length=256
        )
        # 模型生成
        outputs = base_model.generate(
            **inputs,
            max_length=512,
            min_length=min_words,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=True,
            pad_token_id=base_tokenizer.eos_token_id
        )
        # 解码
        sampled_text = base_tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )
        sampled_texts.append(sampled_text)
    return sampled_texts

def generate_samples(args, config, raw_data, batch_size):
    """
    生成机器文本和扰动文本（Jittor版本，替换随机种子）
    """
    # Jittor随机种子（替换torch.manual_seed）
    jt.set_seed(42)
    np.random.seed(42)
    random.seed(42)

    data = {
        "original": [],
        "samples": [],
    }

    for batch_idx in range(0, len(raw_data), batch_size):
        batch_data = raw_data[batch_idx: batch_idx + batch_size]
        print(f'生成第 {batch_idx // batch_size + 1} 批样本（{len(batch_data)}条）')

        sampled_texts = sample_from_model(args, config, batch_data, min_words=30 if args.dataset in ['pubmed'] else 55)

        for o, s in zip(batch_data, sampled_texts):
            if args.dataset == 'pubmed':
                s = truncate_to_substring(s, 'Question:', 2)
                o = o.replace(custom_datasets.SEPARATOR, ' ')
            o, s = trim_to_shorter_length(o, s)
            if o.strip() and s.strip():
                data["original"].append(o)
                data["samples"].append(s)
            else:
                print("警告：过滤掉空文本")
    return data

def generate_data(args, config):
    """
    核心数据生成函数（与原PyTorch逻辑完全一致）
    """
    # 加载数据集
    if args.dataset in custom_datasets.__dict__:
        dataset = custom_datasets.__dict__[args.dataset](args)
    else:
        dataset = datasets.load_dataset(args.dataset, split="train")

    # 提取原始数据
    raw_data = []
    for item in tqdm(dataset, desc="加载原始数据集"):
        if args.dataset_key in item:
            text = item[args.dataset_key].strip()
            if text and len(text.split()) >= 10:  # 过滤短文本
                raw_data.append(text)
        # 限制数据量，避免内存溢出
        if len(raw_data) >= args.max_raw_data:
            break

    # 生成样本
    data = generate_samples(args, config, raw_data, args.batch_size)
    print(f"✅ 数据生成完成：{len(data['original'])} 条原始文本，{len(data['samples'])} 条生成文本")
    return data

