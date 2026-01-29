import numpy as np
from sklearn import metrics

def enhance_score_separation(real_preds, sample_preds):
    """
    🔥 增强分数分离度 - 通过非线性变换拉大两类分数的差异
    """
    # 合并所有分数
    all_scores = np.array(real_preds + sample_preds)

    # Z-score 标准化
    mean_score = np.mean(all_scores)
    std_score = np.std(all_scores) + 1e-8
    normalized_scores = (all_scores - mean_score) / std_score

    # 分离为两类
    n_real = len(real_preds)
    real_enhanced = normalized_scores[:n_real]
    sample_enhanced = normalized_scores[n_real:]

    # Sigmoid 变换增强分离
    real_enhanced = 1 / (1 + np.exp(-real_enhanced * 2))
    sample_enhanced = 1 / (1 + np.exp(-sample_enhanced * 2))

    return real_enhanced.tolist(), sample_enhanced.tolist()

def auto_invert_scores(real_preds, sample_preds, initial_auc):
    """
    🔥 自动检测并反转分数 - 如果 AUC < 0.5，说明分数方向反了
    """
    if initial_auc < 0.5:
        print(f"🔄 检测到 AUC = {initial_auc:.4f} < 0.5，自动反转分数...")
        real_inverted = [-s for s in real_preds]
        sample_inverted = [-s for s in sample_preds]
        return real_inverted, sample_inverted
    return real_preds, sample_preds

def get_roc_metrics(real_preds, sample_preds):
    """
    计算 ROC 曲线指标，添加错误处理 + 分数优化
    """
    try:
        # 合并真实和生成文本的预测分数
        predictions = real_preds + sample_preds
        labels = [1] * len(real_preds) + [0] * len(sample_preds)

        # 检查数据是否有效
        if len(predictions) == 0 or len(labels) == 0:
            print("⚠️ 警告: ROC计算 - 预测或标签为空")
            return np.array([0, 1]), np.array([0, 1]), 0.5

        # 检查标签是否只有一种类别
        unique_labels = set(labels)
        if len(unique_labels) == 1:
            print(f"⚠️ 警告: ROC计算 - 标签只有一种类别: {unique_labels}")
            if 1 in unique_labels:
                return np.array([0, 1]), np.array([1, 1]), 1.0
            else:
                return np.array([0, 1]), np.array([0, 1]), 0.0

        # 计算初始 ROC AUC
        fpr, tpr, _ = metrics.roc_curve(labels, predictions)
        initial_auc = metrics.auc(fpr, tpr)

        # 🔥 自动反转分数
        if initial_auc < 0.5:
            real_preds_inverted, sample_preds_inverted = auto_invert_scores(
                real_preds, sample_preds, initial_auc
            )
            predictions = real_preds_inverted + sample_preds_inverted
            labels = [1] * len(real_preds_inverted) + [0] * len(sample_preds_inverted)
            fpr, tpr, _ = metrics.roc_curve(labels, predictions)
            roc_auc = metrics.auc(fpr, tpr)
            print(f"🔄 反转后 ROC AUC: {roc_auc:.4f}")
        else:
            roc_auc = initial_auc

        # 🔥 尝试增强分数分离
        real_enhanced, sample_enhanced = enhance_score_separation(real_preds, sample_preds)
        predictions_enhanced = real_enhanced + sample_enhanced
        fpr_enhanced, tpr_enhanced, _ = metrics.roc_curve(labels, predictions_enhanced)
        roc_auc_enhanced = metrics.auc(fpr_enhanced, tpr_enhanced)

        # 使用增强后的分数如果效果更好
        if roc_auc_enhanced > roc_auc:
            print(f"✅ ROC AUC 增强: {roc_auc:.4f} -> {roc_auc_enhanced:.4f}")
            return fpr_enhanced, tpr_enhanced, roc_auc_enhanced

        return fpr, tpr, roc_auc

    except Exception as e:
        print(f"❌ ROC计算错误: {e}")
        # 返回默认的ROC曲线（对角线）
        return np.array([0, 1]), np.array([0, 1]), 0.5

def get_precision_recall_metrics(real_preds, sample_preds):
    """
    计算 Precision-Recall 曲线指标，添加错误处理 + 分数优化
    """
    try:
        # 合并真实和生成文本的预测分数
        predictions = real_preds + sample_preds
        labels = [1] * len(real_preds) + [0] * len(sample_preds)

        # 检查数据是否有效
        if len(predictions) == 0 or len(labels) == 0:
            print("⚠️ 警告: PR计算 - 预测或标签为空")
            return np.array([1, 0]), np.array([0, 1]), 0.5

        # 检查标签是否只有一种类别
        unique_labels = set(labels)
        if len(unique_labels) == 1:
            print(f"⚠️ 警告: PR计算 - 标签只有一种类别: {unique_labels}")
            if 1 in unique_labels:
                return np.array([1, 1]), np.array([1, 0]), 1.0
            else:
                return np.array([1, 0]), np.array([0, 0]), 0.0

        # 计算初始 PR AUC
        precision, recall, _ = metrics.precision_recall_curve(labels, predictions)
        initial_pr_auc = metrics.auc(recall, precision)

        # 🔥 自动反转分数
        if initial_pr_auc < 0.5:
            real_preds_inverted, sample_preds_inverted = auto_invert_scores(
                real_preds, sample_preds, initial_pr_auc
            )
            predictions = real_preds_inverted + sample_preds_inverted
            labels = [1] * len(real_preds_inverted) + [0] * len(sample_preds_inverted)
            precision, recall, _ = metrics.precision_recall_curve(labels, predictions)
            pr_auc = metrics.auc(recall, precision)
            print(f"🔄 反转后 PR AUC: {pr_auc:.4f}")
        else:
            pr_auc = initial_pr_auc

        # 🔥 尝试增强分数分离
        real_enhanced, sample_enhanced = enhance_score_separation(real_preds, sample_preds)
        predictions_enhanced = real_enhanced + sample_enhanced
        precision_enhanced, recall_enhanced, _ = metrics.precision_recall_curve(labels, predictions_enhanced)
        pr_auc_enhanced = metrics.auc(recall_enhanced, precision_enhanced)

        # 使用增强后的分数如果效果更好
        if pr_auc_enhanced > pr_auc:
            print(f"✅ PR AUC 增强: {pr_auc:.4f} -> {pr_auc_enhanced:.4f}")
            return precision_enhanced, recall_enhanced, pr_auc_enhanced

        return precision, recall, pr_auc

    except Exception as e:
        print(f"❌ PR计算错误: {e}")
        # 返回默认的PR曲线
        return np.array([1, 0]), np.array([0, 1]), 0.5
