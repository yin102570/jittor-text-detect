# ==================== 评分判别（AUC）专项验证脚本 ====================
# 核心：只验证评分和标签的对应关系、AUC计算逻辑，排除其他干扰
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# -------------------------- 配置项（替换为你的项目参数） --------------------------
MODEL_NAME = "gpt2-t5-small"  # 你的模型名
DATA_PATH = "./data/your_dataset.jsonl"  # 你的数据集路径
LABEL_COL = "label"  # 标签列名（0=人类文本，1=AI文本）
TEXT_COL = "text"  # 文本列名
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------------------------------------------------------------

def load_data_and_scores():
    """加载数据集+计算真实评分（似然度），模拟项目中的评分逻辑"""
    print("===== 步骤1：加载数据并计算真实评分 =====")
    # 1. 加载数据集（适配jsonl/csv）
    try:
        if DATA_PATH.endswith(".jsonl"):
            df = pd.read_json(DATA_PATH, lines=True)
        elif DATA_PATH.endswith(".csv"):
            df = pd.read_csv(DATA_PATH)
        else:
            raise ValueError(f"不支持的格式：{DATA_PATH}")
    except Exception as e:
        print(f"❌ 加载数据失败：{e}")
        return None, None

    # 过滤空值、短文本
    df = df.dropna(subset=[TEXT_COL, LABEL_COL])
    df = df[df[TEXT_COL].apply(lambda x: len(str(x).strip()) >= 10)]
    if len(df) < 20:
        print(f"❌ 有效样本过少（{len(df)}条），至少需要20条")
        return None, None

    # 2. 加载模型计算评分（模拟项目中的似然度评分）
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
        model.eval()
    except Exception as e:
        print(f"❌ 加载模型失败：{e}")

        return None, None

    # 3. 计算每条文本的评分（似然度=-loss）
    scores = []
    for text in df[TEXT_COL].tolist():
        try:
            inputs = tokenizer(
                text, return_tensors="pt", truncation=True, max_length=512
            ).to(DEVICE)
            with torch.no_grad():
                loss = model(**inputs, labels=inputs["input_ids"]).loss.item()
                scores.append(-loss)  # 核心评分：似然度（越大越可能是人类文本）
        except:
            scores.append(np.nan)

    # 过滤NaN评分
    df["score"] = scores
    df = df.dropna(subset=["score"])
    if len(df) < 10:
        print(f"❌ 有效评分样本过少（{len(df)}条）")
        return None, None

    # 输出基础信息
    print(f"✅ 有效样本数：{len(df)}")
    print(f"✅ 标签分布：\n{df[LABEL_COL].value_counts()}")
    print(f"✅ 评分统计：")
    print(f"   人类文本评分均值：{df[df[LABEL_COL] == 0]['score'].mean():.4f}")
    print(f"   AI文本评分均值：{df[df[LABEL_COL] == 1]['score'].mean():.4f}")

    return df["score"].values, df[LABEL_COL].values


def validate_auc_calculation(scores, labels):
    """核心验证：评分判别（AUC计算）逻辑"""
    print("\n===== 步骤2：验证评分判别（AUC）逻辑 =====")
    # 1. 基础校验
    if len(np.unique(labels)) < 2:
        print(f"❌ 标签只有1类，无法计算AUC")
        return
    if len(np.unique(scores)) < 2:
        print(f"❌ 评分无差异（所有值相同），无法区分文本")
        return

    # 2. 验证原始AUC（项目中可能的错误版本）
    try:
        original_auc = roc_auc_score(labels, scores)
        print(f"✅ 原始AUC（项目当前逻辑）：{original_auc:.4f}")
    except Exception as e:
        print(f"❌ 原始AUC计算失败：{e}")
        return

    # 3. 验证评分方向是否搞反（最常见错误！）
    reversed_auc = roc_auc_score(labels, -scores)
    print(f"✅ 反向评分AUC（评分取反）：{reversed_auc:.4f}")

    # 4. 验证PR AUC（辅助确认）
    precision, recall, _ = precision_recall_curve(labels, scores)
    pr_auc = auc(recall, precision)
    print(f"✅ PR AUC（原始评分）：{pr_auc:.4f}")

    # 5. 定位核心问题
    print("\n===== 问题定位 =====")
    if original_auc < 0.6 and reversed_auc > 0.7:
        print(f"🔴 关键错误：评分方向搞反！AI文本评分应该更低，你搞反了标签/评分的对应关系")
        print(f"   解决方案：计算AUC时用 -scores 替代 scores，或交换标签（0=AI，1=人类）")
    elif original_auc < 0.6 and reversed_auc < 0.6:
        print(f"🔴 关键错误：评分无区分度！人类/AI文本的评分均值几乎无差异（看步骤1的均值）")
        print(f"   解决方案：换更大模型（如gpt2-medium）、调整扰动参数、优化评分计算逻辑")
    elif original_auc > 0.7:
        print(f"🟢 评分判别逻辑正常！AUC低是其他环节（数据/扰动）问题")
    else:
        print(f"🟡 评分有微弱区分度，建议调整评分计算逻辑（如归一化、加扰动特征）")


# 执行专项验证
if __name__ == "__main__":
    scores, labels = load_data_and_scores()
    if scores is not None and labels is not None:
        validate_auc_calculation(scores, labels)

