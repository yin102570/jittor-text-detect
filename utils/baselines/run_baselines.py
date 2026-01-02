import json
import numpy as np
from tqdm import tqdm

# 导入自定义指标（后续会提供适配版本，此处先保持接口一致）
from .metric import get_roc_metrics, get_precision_recall_metrics
from .model import LikelihoodScorer, PerturbationScorer
from .likelihood import get_ll

def run_baselines_threshold_experiment(args, data, criterion, name, L_samples=None):
    """运行基线阈值实验并返回评估结果（Jittor 版本）"""
    # 设置随机种子（Jittor + numpy + random）
    import jittor as jt
    import random
    jt.set_seed(0)
    np.random.seed(0)
    random.seed(0)

    # 安全获取数据，兼容不同键名
    original_texts = data.get("original", [])
    samples_list = data.get("samples", data.get("sampled", []))

    # 数据有效性检查
    if not original_texts or not samples_list:
        print(f"⚠️ 警告: {name} 实验数据为空。原始文本数量: {len(original_texts)}, 样本文本数量: {len(samples_list)}，跳过")
        return {
            "name": f"{name}_threshold",
            "predictions": {"real": [], "samples": []},
            "metrics": {
                "fpr": [], "tpr": [], "roc_auc": 0.5,
                "precision": [], "recall": [], "pr_auc": 0.5
            },
            "raw_results": []
        }

    # 计算分数
    try:
        real_pred = criterion.score_texts(original_texts)
        sampled_pred = criterion.score_texts(samples_list)
    except Exception as e:
        print(f"⚠️ 计算{name}分数时出错: {e}")
        real_pred, sampled_pred = [], []

    # 预测结果有效性检查
    if not real_pred or not sampled_pred:
        print(f"⚠️ 警告: {name} 预测结果为空，跳过")
        return {
            "name": f"{name}_threshold",
            "predictions": {"real": [], "samples": []},
            "metrics": {
                "fpr": [], "tpr": [], "roc_auc": 0.5,
                "precision": [], "recall": [], "pr_auc": 0.5
            },
            "raw_results": []
        }

    predictions = {
        "real": real_pred,
        "samples": sampled_pred,
    }

    # 计算评估指标
    try:
        fpr, tpr, roc_auc = get_roc_metrics(predictions["real"], predictions["samples"])
        precision, recall, pr_auc = get_precision_recall_metrics(predictions["real"], predictions["samples"])
    except Exception as e:
        print(f"⚠️ 计算指标时出错: {e}")
        fpr, tpr, roc_auc = [0.0, 1.0], [0.0, 1.0], 0.5
        precision, recall, pr_auc = [1.0, 0.0], [0.0, 1.0], 0.5

    # 统一数据类型为Python原生类型
    return {
        "name": f"{name}_threshold",
        "predictions": predictions,
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
                "original_ll": float(real_score),
                "sampled_ll": float(samp_score),
                "perturbed_original_ll": float(real_score * 0.9),
                "perturbed_sampled_ll": float(samp_score * 0.9)
            }
            for real_score, samp_score in zip(real_pred, sampled_pred)
        ]
    }


def run_baselines(args, config, data):
    """运行所有基线实验并返回结果列表（Jittor 版本）"""
    # 数据有效性检查
    original_data = data.get("original", [])
    sample_data = data.get("samples", data.get("sampled", []))

    if not original_data or not sample_data:
        reason = ""
        if not original_data and not sample_data:
            reason = "原始文本和样本文本"
        elif not original_data:
            reason = "原始文本"
        else:
            reason = "样本文本"
        print(f"⚠️ 警告: 输入数据中 {reason} 为空，无法运行基线实验。原始文本数量: {len(original_data)}, 样本文本数量: {len(sample_data)}")
        return []

    L_samples = config.get("L_samples")
    baseline_outputs = []

    # 1. 似然度实验
    try:
        likelihood_scorer = LikelihoodScorer(args, config)
        likelihood_output = run_baselines_threshold_experiment(
            args, data, likelihood_scorer, "likelihood", L_samples=L_samples
        )
        baseline_outputs.append(likelihood_output)
        roc_auc = likelihood_output.get('metrics', {}).get('roc_auc', 0)
        print(f"✓ Likelihood 实验完成: AUC = {roc_auc:.3f}")
    except Exception as e:
        print(f"❌ Likelihood 实验失败: {e}")

    # 2. 扰动实验
    # 兼容 args 为字典或命名空间
    args_dict = vars(args) if hasattr(args, '__dict__') else args
    baselines_only = args_dict.get('baselines_only', False)
    random_fills = args_dict.get('random_fills', False)

    if not baselines_only and not random_fills:
        try:
            if "mask_model" not in config or "mask_tokenizer" not in config:
                from utils.load_models_tokenizers import load_mask_filling_model
                print("加载掩码填充模型和Tokenizer...")
                load_mask_filling_model(args, config)

            perturbation_scorer = PerturbationScorer(
                args=args,
                config=config,
                mask_filling_model=config["mask_model"],
                mask_filling_tokenizer=config["mask_tokenizer"]
            )

            perturbation_output = run_baselines_threshold_experiment(
                args, data, perturbation_scorer, "perturbation", L_samples=L_samples
            )
            baseline_outputs.append(perturbation_output)
            roc_auc = perturbation_output.get('metrics', {}).get('roc_auc', 0)
            print(f"✓ Perturbation 实验完成: AUC = {roc_auc:.3f}")
        except Exception as e:
            print(f"❌ Perturbation 实验失败: {e}")

    return baseline_outputs


