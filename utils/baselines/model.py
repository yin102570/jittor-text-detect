
import sys
import os
import random
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jittor as jt


def get_lls(args, config, texts):
    """
    计算一组文本的对数似然（Jittor版本，修复loss访问方式）
    """
    base_model = config["base_model"]
    base_tokenizer = config["base_tokenizer"]

    lls = []
    for idx, text in enumerate(texts):
        try:
            # 分词并返回Jittor张量
            tokenized = base_tokenizer(
                text,
                return_tensors="jt",
                truncation=True,
                max_length=512
            )

            # 获取input_ids并确保维度正确
            input_ids = tokenized['input_ids']

            # 确保input_ids是2维张量 [batch_size, seq_len]
            if isinstance(input_ids, list):
                input_ids = jt.array(input_ids)
            if len(input_ids.shape) == 1:
                input_ids = input_ids.unsqueeze(0)

            # labels和input_ids一致（语言模型自回归任务）
            labels = input_ids.clone()

            # 模型前向传播（返回字典格式）
            outputs = base_model(input_ids=input_ids, labels=labels)

            # 🔥 核心修复：字典用["loss"]访问，而非.loss
            if isinstance(outputs, dict):
                # 字典类型：取loss键值
                loss = outputs.get("loss", None)
                if loss is None:
                    # 如果字典中没有loss键，尝试手动计算
                    print(f"⚠️ 模型返回字典中无loss键，尝试手动计算（文本{idx + 1}/{len(texts)}）")
                    logits = outputs.get("logits", None)
                    if logits is not None:
                        # 手动计算交叉熵损失
                        loss_fct = jt.nn.CrossEntropyLoss(ignore_index=0)
                        # 移位预测（语言模型标准做法）
                        shift_logits = logits[..., :-1, :].reshape(-1, logits.shape[-1])
                        shift_labels = labels[..., 1:].reshape(-1)
                        loss = loss_fct(shift_logits, shift_labels)
                    else:
                        raise ValueError("模型返回字典中既无loss也无logits")
            else:
                # 兼容少数情况返回类实例的情况
                loss = getattr(outputs, "loss", None)
                if loss is None:
                    raise ValueError("模型返回对象无loss属性")

            # 转换为Python数值，取负得到对数似然
            ll = -loss.item()
            lls.append(ll)

            # 每处理10条打印进度
            if (idx + 1) % 10 == 0:
                print(f"✅ 已处理 {idx + 1}/{len(texts)} 条文本，当前LL={ll:.4f}")

        except Exception as e:
            print(f"❌ 处理文本 {idx + 1}/{len(texts)} 失败: '{text[:50]}...'")
            print(f"   错误详情: {str(e)}")
            lls.append(0.0)  # 兜底值，避免程序中断

    return lls


def get_ll(args, config, text):
    """
    计算单个文本的对数似然（增加异常处理）
    """
    try:
        return get_lls(args, config, [text])[0]
    except Exception as e:
        print(f"❌ 单文本似然计算失败: {str(e)}")
        return 0.0


class LikelihoodScorer:
    """
    似然度评分器（增强异常处理）
    """

    def __init__(self, args, config, L_samples=None):
        self.args = args
        self.config = config
        self.L_samples = L_samples

    def score(self, text):
        """单文本评分（增加异常兜底）"""
        try:
            return get_ll(self.args, self.config, text)
        except Exception as e:
            print(f"❌ LikelihoodScorer评分失败: {str(e)}")
            return 0.0

    def score_texts(self, texts):
        """批量文本评分"""
        scores = []
        for idx, text in enumerate(texts):
            try:
                score = self.score(text)
                scores.append(score)
                if (idx + 1) % 10 == 0:
                    print(f"✅ LikelihoodScorer已评分 {idx + 1}/{len(texts)} 条文本")
            except Exception as e:
                print(f"❌ 文本 {idx + 1} 评分失败: {str(e)}")
                scores.append(0.0)
        return scores


class PerturbationScorer:
    """
    扰动评分器（增强异常处理和维度校验）
    """

    def __init__(self, args, config, mask_filling_model, mask_filling_tokenizer):
        self.args = args
        self.config = config
        self.mask_filling_model = mask_filling_model
        self.mask_filling_tokenizer = mask_filling_tokenizer
        self.base_model = config["base_model"]
        self.base_tokenizer = config["base_tokenizer"]

    def _perturb_text(self, text):
        """文本扰动核心逻辑（修复generate参数不匹配问题，增加异常处理）"""
        try:
            # 检查tokenizer是否有tokenize方法
            if hasattr(self.base_tokenizer, 'tokenize'):
                tokens = self.base_tokenizer.tokenize(text)
            else:
                # 兜底：使用encode+decode模拟tokenize
                token_ids = self.base_tokenizer.encode(text, truncation=True, max_length=512)
                tokens = [str(tid) for tid in token_ids]  # 简化处理

            n_tokens = len(tokens)
            if n_tokens < 10:
                return text

            # 计算掩码数量
            n_mask = max(1, int(n_tokens * self.args.pct_words_masked))
            mask_positions = []

            # 随机选择掩码位置
            max_attempts = n_tokens * 2  # 防止死循环
            attempts = 0
            while len(mask_positions) < n_mask and attempts < max_attempts:
                start = random.randint(0, max(0, n_tokens - self.args.span_length))
                span = list(range(start, min(start + self.args.span_length, n_tokens)))
                if not any(p in mask_positions for p in span):
                    mask_positions.extend(span)
                attempts += 1

            # 应用掩码
            masked_tokens = tokens.copy()
            mask_token = getattr(self.mask_filling_tokenizer, 'mask_token', '<mask>')
            for pos in mask_positions:
                if pos < len(masked_tokens):
                    masked_tokens[pos] = mask_token

            # 转换回文本
            if hasattr(self.base_tokenizer, 'convert_tokens_to_string'):
                masked_text = self.base_tokenizer.convert_tokens_to_string(masked_tokens)
            else:
                # 兜底：简单拼接
                masked_text = ' '.join(masked_tokens)

            # 分词并生成填充文本
            inputs = self.mask_filling_tokenizer(
                masked_text,
                return_tensors="jt",
                truncation=True,
                max_length=512
            )

            # 确保input_ids维度正确
            input_ids = inputs.get('input_ids', inputs)
            if isinstance(input_ids, list):
                input_ids = jt.array(input_ids)
            if len(input_ids.shape) == 1:
                input_ids = input_ids.unsqueeze(0)

            # 生成填充文本 - 关键修复：移除不支持的num_beams和do_sample参数
            outputs = self.mask_filling_model.generate(
                input_ids=input_ids,
                max_length=min(n_tokens + 20, 512)
            )

            # 解码
            filled_text = self.mask_filling_tokenizer.decode(
                outputs[0],
                skip_special_tokens=True
            )

            return filled_text.strip() if filled_text else text

        except Exception as e:
            print(f"⚠️ 文本扰动失败: {str(e)}")
            return text  # 返回原文本作为兜底

    def score(self, text):
        """单文本扰动评分（增加异常处理 + 多重优化提升AUC）"""
        try:
            # 计算原始文本似然
            original_ll = get_ll(self.args, self.config, text)

            # 生成扰动文本并计算似然
            perturbed_lls = []
            for round_idx in range(self.args.n_perturbation_rounds):
                try:
                    perturbed_text = self._perturb_text(text)
                    if perturbed_text and perturbed_text != text:
                        perturbed_ll = get_ll(self.args, self.config, perturbed_text)
                        perturbed_lls.append(perturbed_ll)
                except Exception as e:
                    print(f"⚠️ 扰动轮次 {round_idx + 1} 失败: {str(e)}")
                    continue

            # 计算平均扰动似然
            if not perturbed_lls:
                print("⚠️ 所有扰动轮次均失败，返回0分")
                return 0.0

            avg_perturbed_ll = np.mean(perturbed_lls)
            std_perturbed_ll = np.std(perturbed_lls) if len(perturbed_lls) > 1 else 0.0

            # 基础曲率分数
            curvature = original_ll - avg_perturbed_ll

            # 🔥 优化1: Z-score 标准化
            if std_perturbed_ll > 0:
                normalized_curvature = curvature / (std_perturbed_ll + 1e-8)
            else:
                normalized_curvature = curvature

            # 🔥 优化2: 多轮扰动一致性检查
            if len(perturbed_lls) >= 2:
                consistency = 1.0 / (1.0 + np.std(perturbed_lls))
            else:
                consistency = 1.0

            # 🔥 优化3: 幂函数放大分数差异
            score = np.sign(curvature) * (np.abs(curvature) ** 0.8)

            # 🔥 优化4: 原始似然归一化（避免长度偏差）
            text_length = len(text.split())
            normalized_original = original_ll / (text_length + 1)

            # 🔥 优化5: 综合评分策略
            # 结合曲率、标准差、一致性和归一化原始分数
            final_score = (score * 0.5 +
                          normalized_curvature * 0.3 +
                          consistency * 0.1 +
                          normalized_original * 0.1)

            return final_score

        except Exception as e:
            print(f"❌ PerturbationScorer评分失败: {str(e)}")
            return 0.0

    def score_texts(self, texts):
        """批量文本扰动评分"""
        scores = []
        for idx, text in enumerate(texts):
            try:
                score = self.score(text)
                scores.append(score)
                if (idx + 1) % 5 == 0:
                    print(f"✅ PerturbationScorer已评分 {idx + 1}/{len(texts)} 条文本")
            except Exception as e:
                print(f"❌ 文本 {idx + 1} 扰动评分失败: {str(e)}")
                scores.append(0.0)
        return scores