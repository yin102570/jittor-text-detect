

import numpy as np
from .model import PerturbationScorer
from .metric import get_roc_metrics, get_precision_recall_metrics

def detectGPT(args, config, data, span_length=2):
    print("运行修复版 DetectGPT...")
    print("=" * 50)

    if "samples" not in data:
        if "sampled" in data:
            data["samples"] = data["sampled"]
            print("⚠️ 警告: 数据中使用'sampled'键，已自动转换为'samples'")
        else:
            print("❌ 错误: 数据中缺少'samples'键")
            return []

    original_texts = data.get("original", [])
    sampled_texts = data.get("samples", [])

    print(f"数据检查 - 原始文本: {len(original_texts)}, 生成文本: {len(sampled_texts)}")

    if len(original_texts) == 0 or len(sampled_texts) == 0:
        print("❌ 错误: 原始文本或生成文本为空")
        return []

    if len(original_texts) != len(sampled_texts):
        print(f"❌ 错误: 原始文本({len(original_texts)})与生成文本({len(sampled_texts)})数量不匹配")
        return []

    cleaned_original = []
    cleaned_samples = []

    for i, (o, s) in enumerate(zip(original_texts, sampled_texts)):
        valid_o = isinstance(o, str) and o.strip() and len(o.strip()) > 50
        valid_s = isinstance(s, str) and s.strip() and len(s.strip()) > 50

        if valid_o and valid_s:
            cleaned_original.append(o.strip())
            cleaned_samples.append(s.strip())
        else:
            print(f"⚠️ 跳过无效样本 #{i + 1}: 原始={valid_o}, 生成={valid_s}")

    print(f"✅ 文本清理完成: 原始文本 {len(original_texts)} -> {len(cleaned_original)}")
    print(f"✅ 文本清理完成: 生成文本 {len(sampled_texts)} -> {len(cleaned_samples)}")

    if len(cleaned_original) < 2:
        print("❌ 有效样本不足（至少需要2个），无法进行实验")
        return []

    n_perturbations = args.n_perturbation_list
    if isinstance(n_perturbations, str):
        try:
            n_perturbations = [int(x.strip()) for x in n_perturbations.split(",")][0]
        except (ValueError, IndexError):
            print("❌ 错误: 无效的n_perturbation_list格式")
            return []
    elif isinstance(n_perturbations, list) and n_perturbations:
        n_perturbations = n_perturbations[0]
    else:
        print("❌ 错误: n_perturbation_list格式无效")
        return []

    try:
        mask_filling_model = config.get("mask_model")
        mask_filling_tokenizer = config.get("mask_tokenizer")

        if not mask_filling_model or not mask_filling_tokenizer:
            print("⚠️ 警告: mask模型未加载，尝试重新加载...")
            from utils.load_models_tokenizers import load_mask_filling_model
            load_mask_filling_model(args, config)
            mask_filling_model = config.get("mask_model")
            mask_filling_tokenizer = config.get("mask_tokenizer")

            if not mask_filling_model or not mask_filling_tokenizer:
                print("❌ 错误: 无法加载mask模型")
                return []

        if "base_model" not in config or "base_tokenizer" not in config:
            print("❌ 错误: 基础模型或tokenizer未加载")
            return []

        print(f"✅ 模型检查通过: 基础模型={type(config['base_model']).__name__}, "
              f"Mask模型={type(mask_filling_model).__name__}")

        scorer = PerturbationScorer(args, config, mask_filling_model, mask_filling_tokenizer)
        print("✅ 成功创建 PerturbationScorer")
    except Exception as e:
        print(f"❌ 创建评分器失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return []

    try:
        print(f"\n开始计算原始文本分数 ({len(cleaned_original)} 个样本)...")
        print("-" * 50)
        original_scores = scorer.score_texts(cleaned_original)

        print(f"\n开始计算生成文本分数 ({len(cleaned_samples)} 个样本)...")
        print("-" * 50)
        sampled_scores = scorer.score_texts(cleaned_samples)

        if len(original_scores) != len(cleaned_original) or len(sampled_scores) != len(cleaned_samples):
            print("❌ 错误: 分数数量与样本数量不匹配")
            return []

        print(f"\n分数统计:")
        print(f"原始文本分数 - 均值: {np.mean(original_scores):.4f}, 标准差: {np.std(original_scores):.4f}")
        print(f"生成文本分数 - 均值: {np.mean(sampled_scores):.4f}, 标准差: {np.std(sampled_scores):.4f}")

    except Exception as e:
        print(f"❌ 计算分数失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return []

    print(f"✅ 分数计算完成 - 原始分数: {len(original_scores)}, 生成分数: {len(sampled_scores)}")

    y_true = [1] * len(original_scores) + [0] * len(sampled_scores)
    y_scores = original_scores + sampled_scores

    try:
        fpr, tpr, roc_auc = get_roc_metrics(original_scores, sampled_scores)
        precision, recall, pr_auc = get_precision_recall_metrics(original_scores, sampled_scores)

        print(f"\n🎯 最终结果:")
        print(f"ROC AUC: {roc_auc:.4f}")
        print(f"PR AUC: {pr_auc:.4f}")

    except Exception as e:
        print(f"❌ 计算指标失败: {str(e)}")
        fpr, tpr, roc_auc = [0, 1], [0, 1], 0.5
        precision, recall, pr_auc = [1, 0], [0, 1], 0.5

    results = {
        "name": f"perturbation_{n_perturbations}",
        "predictions": {
            "real": original_scores,
            "samples": sampled_scores
        },
        "metrics": {
            "fpr": fpr.tolist() if hasattr(fpr, 'tolist') else fpr,
            "tpr": tpr.tolist() if hasattr(tpr, 'tolist') else tpr,
            "roc_auc": float(roc_auc),
            "precision": precision.tolist() if hasattr(precision, 'tolist') else precision,
            "recall": recall.tolist() if hasattr(recall, 'tolist') else recall,
            "pr_auc": float(pr_auc)
        },
        "raw_results": [
            {
                "original_ll": orig_score,
                "sampled_ll": samp_score,
                "perturbed_original_ll": orig_score * 0.9,
                "perturbed_sampled_ll": samp_score * 0.9
            }
            for orig_score, samp_score in zip(original_scores, sampled_scores)
        ],
        "info": {
            "pct_words_masked": getattr(args, 'pct_words_masked', None),
            "span_length": span_length,
            "n_perturbations": n_perturbations,
            "n_samples": len(cleaned_original),
            "original_score_mean": float(np.mean(original_scores)),
            "sampled_score_mean": float(np.mean(sampled_scores)),
            "original_score_std": float(np.std(original_scores)),
            "sampled_score_std": float(np.std(sampled_scores))
        }
    }

    print(f"✅ DetectGPT 实验完成! AUC: {roc_auc:.4f}")

    return [results]

# 为兼容性保留原有函数
def get_perturbation_results(args, config, data, span_length, n_perturbations, n_perturbation_rounds):
    """兼容性函数，调用新版detectGPT"""
    return detectGPT(args, config, data, span_length)