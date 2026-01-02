# rank.py
# 完全移除 PyTorch 依赖，适配 Jittor 环境
import jittor as jt
import numpy as np


def get_rank(args, config, text, log=False):
    """计算文本中每个token在模型似然排序中的平均排名（Jittor 版本）"""
    if not hasattr(args, 'openai_model'):
        args.openai_model = None
    if args.openai_model is not None:
        raise NotImplementedError("get_rank暂不支持OpenAI模型")
    
    base_model = config.get("base_model")
    base_tokenizer = config.get("base_tokenizer")

    if not base_model or not base_tokenizer:
        raise ValueError("基础模型或分词器未正确加载")

    # 检查文本有效性
    if not text or not text.strip():
        print("⚠️ 检测到空文本，返回默认排名")
        return 0.0

    try:
        with jt.no_grad():  # Jittor 无梯度上下文
            # 文本编码（返回 Jittor 张量）
            tokenized = base_tokenizer(
                text,
                return_tensors="jt",
                padding=False,
                truncation=True,
                max_length=512
            )

            # 模型前向传播获取logits
            logits = base_model(**tokenized).logits[:, :-1]  # 移除最后一个位置
            labels = tokenized.input_ids[:, 1:]  # 移除第一个token作为标签

            if logits.shape[1] != labels.shape[1]:
                raise ValueError(f"logits与labels长度不匹配: {logits.shape[1]} vs {labels.shape[1]}")

            # 计算每个标签token的排名（Jittor 张量操作）
            sorted_logits = logits.argsort(dim=-1, descending=True)  # 按似然降序排序
            # Jittor 张量匹配操作
            labels_expanded = labels.unsqueeze(-1)
            matches = (sorted_logits == labels_expanded).nonzero()

            # 验证匹配结果格式
            if matches.ndim != 2 or matches.shape[1] != 3:
                raise ValueError(f"匹配结果格式错误，预期形状为 (N, 3)，实际为 {matches.shape}")

            ranks, timesteps = matches[:, -1], matches[:, -2]

            # 验证每个时间步都有唯一匹配
            expected_timesteps = jt.arange(len(timesteps))
            if not jt.all(timesteps == expected_timesteps):
                raise ValueError("每个时间步应恰好有一个匹配")

            # 转换为1-indexed排名
            ranks = ranks.float() + 1.0

            # 可选：取对数
            if log:
                ranks = jt.log(ranks)

            return ranks.mean().item()

    except Exception as e:
        print(f"❌ 计算排名时出错: {str(e)}")
        return 0.0

