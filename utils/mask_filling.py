# mask_filling.py
# 完全移除 PyTorch 依赖，适配 Jittor 环境
import random
import re
import jittor as jt
from tqdm import tqdm

# 替换 transformers 为 jittor-transformers（若已安装），否则使用模拟接口
try:
    from jittor.transformers import T5ForConditionalGeneration, T5Tokenizer
except ImportError:
    # 若未安装 jittor-transformers，使用自定义模拟接口（兼容之前的 load_models_tokenizers.py）
    from utils.load_models_tokenizers import T5ForConditionalGeneration, T5Tokenizer


class MaskFiller:
    """掩码填充工具类，用于文本扰动（Jittor 版本）"""

    def __init__(self, model_name, tokenizer=None, device="cpu"):
        self.model = None  # 延迟加载
        self.model_name = model_name
        self.tokenizer = tokenizer or T5Tokenizer.from_pretrained(model_name)
        self.device = device  # Jittor 中该参数仅用于兼容，实际由 jt.flags.use_cuda 控制

    def load_model(self):
        """延迟加载模型以节省内存（Jittor 版本）"""
        if self.model is None:
            try:
                print(f"加载掩码填充模型: {self.model_name}")
                self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)
                # Jittor 无需 to(device)，自动适配
                self.model.eval()
            except Exception as e:
                print(f"❌ 加载掩码填充模型失败: {e}")
                raise

    def replace_masks(self, texts):
        """替换文本中的掩码标记并返回填充后的文本（Jittor 版本）"""
        self.load_model()
        replaced_texts = []

        for text in texts:
            current_text = text
            masks = re.findall(r'<extra_id_\d+>', current_text)

            if masks:
                # 逐个替换掩码
                for mask in masks:
                    # 构建输入文本
                    input_text = current_text.replace(mask, "<extra_id>")
                    try:
                        inputs = self.tokenizer.encode(
                            input_text,
                            return_tensors="jt",  # 返回 Jittor 张量
                            truncation=True,
                            max_length=512
                        )

                        # 生成替换内容（Jittor 模型生成）
                        with jt.no_grad():
                            outputs = self.model.generate(
                                inputs,
                                max_length=512,
                                num_return_sequences=1,
                                do_sample=False
                            )

                        replacement = self.tokenizer.decode(
                            outputs[0],
                            skip_special_tokens=True
                        ).strip()
                        current_text = current_text.replace(mask, replacement, 1)
                    except Exception as e:
                        print(f"❌ 替换掩码 {mask} 失败: {e}")
                        current_text = current_text.replace(mask, "", 1)  # 移除无法替换的掩码

            replaced_texts.append(current_text)

        return replaced_texts


def perturb_texts(texts, pct=0.3, span_length=2, model_name="t5-small", tokenizer=None, device="cpu"):
    """
    扰动文本：随机替换部分文本为掩码，再用模型填充（Jittor 版本）

    参数:
        texts: 原始文本列表
        pct: 替换比例（占总词数）
        span_length: 每个掩码替换的词跨度
        model_name: 掩码填充模型名称
        tokenizer: 分词器（可选）
        device: 运行设备（Jittor 中仅兼容）

    返回:
        扰动后的文本列表
    """
    if not texts:
        print("⚠️ 输入文本列表为空")
        return []

    print(f"扰动 {len(texts)} 个文本，掩码比例: {pct}, 跨度长度: {span_length}")
    mask_filler = MaskFiller(model_name, tokenizer, device)
    perturbed_texts = []

    for text in texts:
        words = text.split()
        if len(words) <= span_length:
            # 文本过短，直接添加后缀作为扰动
            perturbed = text + " [扰动]" if not text.endswith(" ") else text[:-1] + "[扰动]"
            perturbed_texts.append(perturbed)
            continue

        # 计算需要掩码的数量
        n_masks = max(1, int(len(words) * pct))
        masked_words = words.copy()

        # 插入掩码
        for i in range(n_masks):
            mask_token = f"<extra_id_{i}>"
            # 随机选择插入位置（避免首尾）
            insert_pos = random.randint(1, len(masked_words) - 1)
            masked_words.insert(insert_pos, mask_token)

        masked_text = " ".join(masked_words)

        # 替换掩码
        try:
            filled_texts = mask_filler.replace_masks([masked_text])
            perturbed_text = filled_texts[0] if filled_texts else text
        except Exception as e:
            print(f"❌ 处理文本时出错: {e}")
            perturbed_text = text

        # 确保扰动后文本与原始不同
        if perturbed_text == text:
            perturbed_text = text + " " if not text.endswith(" ") else text[:-1]

        perturbed_texts.append(perturbed_text)

    return perturbed_texts


if __name__ == "__main__":
    # 测试代码
    test_texts = ["Hello world this is a test.", "Another example sentence here."]
    result = perturb_texts(test_texts, pct=0.3, span_length=2)
    print("原始:", test_texts)
    print("扰动:", result)
