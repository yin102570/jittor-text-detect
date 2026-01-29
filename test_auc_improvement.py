#!/usr/bin/env python3
"""
AUC 极致优化测试脚本
测试多种优化策略对 AUC 的影响
"""

import sys
import os
import numpy as np

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import jittor as jt
from utils.setting import set_experiment_config, initial_setup
from utils.load_models_tokenizers import load_base_model_and_tokenizer, load_mask_filling_model
from utils.baselines.detectGPT import detectGPT
from utils.baselines.metric import get_roc_metrics, get_precision_recall_metrics

def test_auc_improvement():
    """测试 AUC 改进效果"""

    print("=" * 60)
    print("🚀 AUC 极致优化测试")
    print("=" * 60)

    # 1. 配置优化后的参数
    class Args:
        dataset = 'builtin'
        max_raw_data = 100
        batch_size = 8
        n_perturbation_list = '15'
        base_model_name = 'gpt2'
        mask_filling_model_name = 't5-small'
        cache_dir = './cache'
        pct_words_masked = 0.25  # 提升到 0.25
        span_length = 1  # 降低到 1
        n_perturbation_rounds = 15  # 提升到 15
        DEVICE = 'auto'
        min_samples = 10

    args = Args()

    print(f"\n📊 优化参数配置:")
    print(f"  - 掩码比例: {args.pct_words_masked}")
    print(f"  - 掩码跨度: {args.span_length}")
    print(f"  - 扰动轮数: {args.n_perturbation_rounds}")
    print(f"  - 样本数量: {args.max_raw_data}")

    # 2. 加载数据
    from run import load_builtin_data_with_labels
    data = load_builtin_data_with_labels(args)

    print(f"\n✅ 数据加载完成:")
    print(f"  - 原始文本: {len(data.get('original', []))}")
    print(f"  - 生成文本: {len(data.get('samples', []))}")

    # 3. 加载模型
    print(f"\n🔧 加载模型...")
    config = {}

    try:
        print(f"  - 加载基础模型: {args.base_model_name}")
        config['base_model'], config['base_tokenizer'] = load_base_model_and_tokenizer(args)

        print(f"  - 加载掩码模型: {args.mask_filling_model_name}")
        config['mask_model'], config['mask_tokenizer'] = load_mask_filling_model(args)

        print(f"✅ 模型加载完成")
    except Exception as e:
        print(f"❌ 模型加载失败: {str(e)}")
        return

    # 4. 运行优化后的 DetectGPT
    print(f"\n🎯 运行极致优化版 DetectGPT...")
    print("-" * 60)

    results = detectGPT(args, config, data, span_length=args.span_length)

    if not results:
        print("\n❌ DetectGPT 返回空结果")
        return

    result = results[0]

    print("\n" + "=" * 60)
    print("📊 最终结果")
    print("=" * 60)

    metrics = result.get('metrics', {})
    roc_auc = metrics.get('roc_auc', 0)
    pr_auc = metrics.get('pr_auc', 0)

    print(f"\n🎯 AUC 指标:")
    print(f"  - ROC AUC:  {roc_auc:.4f}")
    print(f"  - PR  AUC:  {pr_auc:.4f}")

    # 计算第三个 AUC (F1 AUC)
    predictions = result.get('predictions', {})
    real_scores = predictions.get('real', [])
    sample_scores = predictions.get('samples', [])

    if real_scores and sample_scores:
        # 计算 F1 曲线
        from sklearn.metrics import f1_score

        y_true = [1] * len(real_scores) + [0] * len(sample_scores)
        y_scores = real_scores + sample_scores

        # 计算不同阈值下的 F1 分数
        thresholds = np.linspace(min(y_scores), max(y_scores), 100)
        f1_scores = []

        for threshold in thresholds:
            y_pred = [1 if score >= threshold else 0 for score in y_scores]
            try:
                f1 = f1_score(y_true, y_pred, zero_division=0)
                f1_scores.append(f1)
            except:
                f1_scores.append(0)

        # F1 AUC (使用阈值作为横坐标)
        from sklearn.metrics import auc as sk_auc
        thresholds_norm = (thresholds - thresholds.min()) / (thresholds.max() - thresholds.min() + 1e-8)
        f1_auc_value = sk_auc(thresholds_norm, f1_scores)

        print(f"  - F1  AUC:  {f1_auc_value:.4f}")

        # 总体评价
        avg_auc = (roc_auc + pr_auc + f1_auc_value) / 3

        print(f"\n📈 平均 AUC: {avg_auc:.4f}")

        # 评级
        if avg_auc >= 0.95:
            rating = "⭐⭐⭐⭐⭐ 完美"
        elif avg_auc >= 0.90:
            rating = "⭐⭐⭐⭐ 优秀"
        elif avg_auc >= 0.85:
            rating = "⭐⭐⭐ 良好"
        elif avg_auc >= 0.80:
            rating = "⭐⭐ 及格"
        else:
            rating = "⭐ 需改进"

        print(f"🏆 综合评级: {rating}")

        # 改进建议
        if avg_auc < 0.85:
            print(f"\n💡 进一步优化建议:")
            print(f"  1. 增加扰动轮数到 20+")
            print(f"  2. 调整掩码比例到 0.30")
            print(f"  3. 使用更大的样本集 (200+)")
            print(f"  4. 启用集成分类器 (--ultimate)")

    print("\n" + "=" * 60)
    print("✅ 测试完成")
    print("=" * 60)

if __name__ == "__main__":
    test_auc_improvement()
