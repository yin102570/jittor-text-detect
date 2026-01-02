# likelihood.py（纯本地Jittor模型版本，无OpenAI依赖）
import numpy as np
import jittor as jt

def get_ll(args, config, text):
    """计算单个文本的对数似然值（仅本地Jittor模型，无OpenAI依赖）"""
    DEVICE = args.DEVICE
    base_model = config.get("base_model")
    base_tokenizer = config.get("base_tokenizer")

    # 检查文本有效性
    if not text or not text.strip():
        print("⚠️ 检测到空文本，返回默认值")
        return 0.0

    # 仅保留本地Jittor模型逻辑，删除OpenAI分支
    if not base_model or not base_tokenizer:
        print("❌ 未加载基础模型或分词器")
        return 0.0

    try:
        with jt.no_grad():  # Jittor 无梯度上下文
            tokenized = base_tokenizer(
                text,
                return_tensors="jt",
                padding=False,
                truncation=True,
                max_length=512
            )
            labels = tokenized.input_ids
            loss = base_model(**tokenized, labels=labels).loss
            return -loss.item()  # 返回负损失作为似然值
    except Exception as e:
        print(f"❌ 模型计算对数似然失败: {str(e)}")
        return 0.0


def get_lls(args, config, texts):
    """批量计算文本的对数似然值（仅本地Jittor模型，无OpenAI依赖）"""
    GPT2_TOKENIZER = config.get("GPT2_TOKENIZER")

    # 过滤空文本
    valid_texts = [text for text in texts if text and text.strip()]
    invalid_count = len(texts) - len(valid_texts)

    if invalid_count > 0:
        print(f"⚠️ 过滤掉 {invalid_count} 个空文本")

    if not valid_texts:
        print("⚠️ 没有有效文本进行处理")
        return []

    # 仅本地模型处理，删除OpenAI相关逻辑
    return [get_ll(args, config, text) for text in valid_texts]


