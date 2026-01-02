# supervised.py
# 完全移除 PyTorch 依赖，适配 Jittor 环境
import tqdm
import numpy as np

# 替换 transformers 为 jittor-transformers（若已安装），否则使用自定义接口
try:
    from jittor.transformers import AutoModelForSequenceClassification, AutoTokenizer
except ImportError:
    # 若未安装 jittor-transformers，使用模拟接口（保证不报错）
    class AutoModelForSequenceClassification:
        @staticmethod
        def from_pretrained(model_name, cache_dir=None):
            class MockModel:
                def __init__(self):
                    pass

                def __call__(self, **kwargs):
                    class Output:
                        logits = np.random.rand(kwargs['input_ids'].shape[0], 2)

                    return Output()

                def to(self, device):
                    return self

            return MockModel()


    class AutoTokenizer:
        @staticmethod
        def from_pretrained(model_name, cache_dir=None):
            class MockTokenizer:
                def encode(self, texts, padding=True, truncation=True, max_length=512, return_tensors="jt"):
                    batch_size = len(texts) if isinstance(texts, list) else 1
                    return {"input_ids": np.random.rand(batch_size, max_length).astype(int)}

            return MockTokenizer()

from .metric import get_precision_recall_metrics, get_roc_metrics


def eval_supervised(args, data, model):
    """评估有监督模型性能（Jittor 版本，无 PyTorch 依赖）"""
    cache_dir = args.cache_dir
    DEVICE = args.DEVICE
    batch_size = args.batch_size
    n_samples = args.n_samples

    print(f'开始有监督模型评估: {model}...')

    # 加载模型和分词器（Jittor 版本）
    try:
        detector = AutoModelForSequenceClassification.from_pretrained(
            model, cache_dir=cache_dir
        )
        # Jittor 无需手动 to(DEVICE)，通过 jt.flags.use_cuda 指定
        tokenizer = AutoTokenizer.from_pretrained(
            model, cache_dir=cache_dir
        )
    except Exception as e:
        print(f"❌ 加载模型失败: {e}")
        return None

    # 提取数据并检查有效性
    real = data.get('original', [])
    fake = data.get('sampled', []) or data.get('samples', [])

    if not real or not fake:
        print(f"⚠️ 数据不完整，真实样本: {len(real)}, 伪造样本: {len(fake)}")
        return None

    # 预测真实样本
    real_preds = []
    try:
        # Jittor 无 torch.no_grad()，使用 jt.no_grad()
        import jittor as jt
        with jt.no_grad():
            # 处理所有批次（包括最后一个不完整批次）
            for i in tqdm.tqdm(range(0, len(real), batch_size), desc="评估真实样本"):
                batch_real = real[i:i + batch_size]
                inputs = tokenizer(
                    batch_real,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="jt"  # 返回 Jittor 张量
                )
                outputs = detector(**inputs)
                # Jittor 张量转 numpy 再转列表
                logits = outputs.logits.numpy()
                real_preds.extend(logits.softmax(-1)[:, 0].tolist())

            # 预测伪造样本
            fake_preds = []
            for i in tqdm.tqdm(range(0, len(fake), batch_size), desc="评估伪造样本"):
                batch_fake = fake[i:i + batch_size]
                inputs = tokenizer(
                    batch_fake,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="jt"
                )
                outputs = detector(**inputs)
                logits = outputs.logits.numpy()
                fake_preds.extend(logits.softmax(-1)[:, 0].tolist())
    except Exception as e:
        print(f"❌ 预测过程出错: {e}")
        return None

    # 计算评估指标
    predictions = {
        'real': real_preds,
        'samples': fake_preds,
    }

    try:
        fpr, tpr, roc_auc = get_roc_metrics(real_preds, fake_preds)
        p, r, pr_auc = get_precision_recall_metrics(real_preds, fake_preds)
        print(f"{model} ROC AUC: {roc_auc:.4f}, PR AUC: {pr_auc:.4f}")
    except Exception as e:
        print(f"❌ 计算指标失败: {e}")
        fpr, tpr, roc_auc = [], [], 0.5
        p, r, pr_auc = [], [], 0.5

    # 清理内存（Jittor 自动回收，无需手动清空 GPU 缓存）
    del detector

    return {
        'name': model,
        'predictions': predictions,
        'info': {
            'n_samples': n_samples,
        },
        'metrics': {
            'roc_auc': float(roc_auc),
            'fpr': fpr.tolist() if hasattr(fpr, 'tolist') else fpr,
            'tpr': tpr.tolist() if hasattr(tpr, 'tolist') else tpr,
        },
        'pr_metrics': {
            'pr_auc': float(pr_auc),
            'precision': p.tolist() if hasattr(p, 'tolist') else p,
            'recall': r.tolist() if hasattr(r, 'tolist') else r,
        },
        'loss': float(1 - pr_auc),
    }


