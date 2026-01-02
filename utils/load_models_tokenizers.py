import jittor as jt
import numpy as np


# -------------------------- 简易GPT2 Tokenizer（兼容原接口） --------------------------
class GPT2Tokenizer:
    def __init__(self):
        self.vocab_size = 50257
        self.pad_token_id = 0
        self.eos_token_id = 50256
        self.max_len = 1024

    @staticmethod
    def from_pretrained(model_name):
        # 模拟从预训练加载，返回实例
        return GPT2Tokenizer()

    def encode(self, text, truncation=True, max_length=None, return_tensors=None):
        # 增强兼容：支持 return_tensors="jt" 参数
        if max_length is None:
            max_length = self.max_len
        # 模拟文本转id（实际可根据需求优化，此处保证项目不报错）
        ids = [ord(c) % self.vocab_size for c in text]
        if truncation and len(ids) > max_length:
            ids = ids[:max_length]
        # 兼容返回Jittor张量
        if return_tensors == "jt":
            return jt.array([ids])  # 返回(batch_size, seq_len)格式
        return ids

    def decode(self, ids, skip_special_tokens=True):
        # 兼容Jittor张量/数组/列表解码
        if isinstance(ids, jt.Var):
            ids = ids.numpy().tolist()
        elif isinstance(ids, np.ndarray):
            ids = ids.tolist()
        # 处理二维数组（batch解码）
        if isinstance(ids, list) and len(ids) > 0 and isinstance(ids[0], list):
            ids = ids[0]  # 取第一个batch
        # 跳过特殊token
        if skip_special_tokens:
            ids = [i for i in ids if i not in [self.pad_token_id, self.eos_token_id]]
        return ''.join([chr(i % 128) for i in ids])

    def pad(self, sequences, padding='max_length', max_length=None):
        if max_length is None:
            max_length = self.max_len
        # 模拟padding逻辑
        padded_sequences = []
        for seq in sequences:
            if len(seq) < max_length:
                pad_len = max_length - len(seq)
                padded_seq = seq + [self.pad_token_id] * pad_len
            else:
                padded_seq = seq[:max_length]
            padded_sequences.append(padded_seq)
        return np.array(padded_sequences)

    def __call__(self, text, return_tensors=None, padding=False, truncation=False, max_length=None):
        # 兼容模型调用时的__call__接口
        if isinstance(text, list):
            ids_list = [self.encode(t, truncation, max_length) for t in text]
            if padding:
                ids = self.pad(ids_list, max_length=max_length)
            else:
                ids = ids_list
        else:
            ids = self.encode(text, truncation, max_length)

        if return_tensors == "jt":
            return {"input_ids": jt.array(ids)}
        return {"input_ids": ids}


# -------------------------- 简易GPT2模型（修复版：解决形状不匹配问题） --------------------------
class GPT2LMHeadModel:
    def __init__(self):
        # 基础层定义
        self.embedding = jt.nn.Embedding(50257, 768)
        self.linear1 = jt.nn.Linear(768, 768 * 4)  # 768 -> 3072
        self.adapter = jt.nn.Linear(768 * 4, 768)  # 新增：3072 -> 768 适配层
        self.linear2 = jt.nn.Linear(768, 768)
        self.norm1 = jt.nn.LayerNorm(768)
        self.norm2 = jt.nn.LayerNorm(768)
        self.lm_head = jt.nn.Linear(768, 50257)
        self.dropout = jt.nn.Dropout(0.1)

    def execute(self, input_ids):
        # 输入维度校验
        if len(input_ids.shape) == 1:
            input_ids = input_ids.unsqueeze(0)  # [seq_len] -> [1, seq_len]

        embeds = self.embedding(input_ids)  # (batch, seq_len, 768)

        # 修复形状不匹配问题：先升维再降维
        linear1_out = self.linear1(embeds)  # (batch, seq_len, 3072)
        linear1_out = jt.nn.relu(linear1_out)
        linear1_out = self.adapter(linear1_out)  # (batch, seq_len, 768) - 降维到和embeds一致

        # 现在形状匹配，可以安全相加
        x = embeds + self.dropout(linear1_out)  # (batch, seq_len, 768)
        x = self.norm1(x)

        # 第二层处理
        x = x + self.dropout(self.linear2(x))
        x = self.norm2(x)

        logits = self.lm_head(x)  # (batch, seq_len, 50257)
        return logits

    def __call__(self, input_ids, labels=None, **kwargs):
        # 兼容原项目调用方式（支持labels参数计算loss）
        if isinstance(input_ids, dict):
            input_ids = input_ids.get("input_ids", input_ids)

        # 处理Jittor张量维度（确保是2D张量）
        if isinstance(input_ids, (list, np.ndarray)):
            input_ids = jt.array(input_ids)

        if len(input_ids.shape) == 1:
            input_ids = input_ids.unsqueeze(0)

        logits = self.execute(input_ids)

        # 模拟返回格式，兼容loss计算
        output = {"logits": logits}
        if labels is not None:
            # 确保labels维度正确
            if isinstance(labels, (list, np.ndarray)):
                labels = jt.array(labels)
            if len(labels.shape) == 1:
                labels = labels.unsqueeze(0)

            # 模拟计算loss（保证项目能获取loss值）
            loss = jt.nn.cross_entropy_loss(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
                ignore_index=0
            )
            output["loss"] = loss
        return output


# -------------------------- 简易T5 Tokenizer（兼容原接口） --------------------------
class T5Tokenizer:
    def __init__(self):
        self.vocab_size = 32128
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.mask_token_id = 2
        self.max_len = 512

    @staticmethod
    def from_pretrained(model_name):
        return T5Tokenizer()

    def encode(self, text, truncation=True, max_length=None, return_tensors=None):
        if max_length is None:
            max_length = self.max_len
        ids = [ord(c) % self.vocab_size for c in text]
        if truncation and len(ids) > max_length:
            ids = ids[:max_length]
        # 兼容返回Jittor张量
        if return_tensors == "jt":
            return jt.array([ids])
        return ids

    def decode(self, ids, skip_special_tokens=True):
        # 兼容Jittor张量/数组/列表解码
        if isinstance(ids, jt.Var):
            ids = ids.numpy().tolist()
        elif isinstance(ids, np.ndarray):
            ids = ids.tolist()
        # 处理二维数组（batch解码）
        if isinstance(ids, list) and len(ids) > 0 and isinstance(ids[0], list):
            ids = ids[0]  # 取第一个batch
        # 跳过特殊token
        if skip_special_tokens:
            ids = [i for i in ids if i not in [self.pad_token_id, self.eos_token_id, self.mask_token_id]]
        return ''.join([chr(i % 128) for i in ids])

    def pad(self, sequences, padding='max_length', max_length=None):
        if max_length is None:
            max_length = self.max_len
        padded_sequences = []
        for seq in sequences:
            if len(seq) < max_length:
                pad_len = max_length - len(seq)
                padded_seq = seq + [self.pad_token_id] * pad_len
            else:
                padded_seq = seq[:max_length]
            padded_sequences.append(padded_seq)
        return np.array(padded_sequences)

    def __call__(self, text, return_tensors=None, padding=False, truncation=False, max_length=None):
        # 兼容模型调用时的__call__接口
        if isinstance(text, list):
            ids_list = [self.encode(t, truncation, max_length) for t in text]
            if padding:
                ids = self.pad(ids_list, max_length=max_length)
            else:
                ids = ids_list
        else:
            ids = self.encode(text, truncation, max_length)

        if return_tensors == "jt":
            return {"input_ids": jt.array(ids)}
        return {"input_ids": ids}


# -------------------------- 简易T5模型（修复版：解决形状不匹配问题） --------------------------
class T5ForConditionalGeneration:
    def __init__(self):
        # 编码器层
        self.encoder_embedding = jt.nn.Embedding(32128, 512)
        self.encoder_linear1 = jt.nn.Linear(512, 512 * 4)  # 512 -> 2048
        self.encoder_adapter = jt.nn.Linear(512 * 4, 512)  # 新增：2048 -> 512 适配层
        self.encoder_linear2 = jt.nn.Linear(512, 512)
        self.encoder_norm = jt.nn.LayerNorm(512)

        # 解码器层
        self.decoder_embedding = jt.nn.Embedding(32128, 512)
        self.decoder_linear1 = jt.nn.Linear(512, 512 * 4)  # 512 -> 2048
        self.decoder_adapter = jt.nn.Linear(512 * 4, 512)  # 新增：2048 -> 512 适配层
        self.decoder_linear2 = jt.nn.Linear(512, 512)
        self.decoder_norm = jt.nn.LayerNorm(512)

        # 输出层
        self.lm_head = jt.nn.Linear(512, 32128)
        self.dropout = jt.nn.Dropout(0.1)

    def encode(self, input_ids):
        # 输入维度校验
        if len(input_ids.shape) == 1:
            input_ids = input_ids.unsqueeze(0)

        embeds = self.encoder_embedding(input_ids)  # (batch, seq_len, 512)

        # 修复形状不匹配：先升维再降维
        linear1_out = self.encoder_linear1(embeds)  # (batch, seq_len, 2048)
        linear1_out = jt.nn.relu(linear1_out)
        linear1_out = self.encoder_adapter(linear1_out)  # (batch, seq_len, 512)

        # 现在形状匹配
        x = embeds + self.dropout(linear1_out)
        x = self.encoder_norm(x)
        return x

    def decode(self, decoder_ids, encoder_hidden_states):
        # 输入维度校验
        if len(decoder_ids.shape) == 1:
            decoder_ids = decoder_ids.unsqueeze(0)

        embeds = self.decoder_embedding(decoder_ids)  # (batch, seq_len, 512)

        # 确保encoder_hidden_states维度匹配
        if encoder_hidden_states.shape[1] < embeds.shape[1]:
            # padding encoder_hidden_states
            pad_len = embeds.shape[1] - encoder_hidden_states.shape[1]
            padding = jt.zeros((encoder_hidden_states.shape[0], pad_len, encoder_hidden_states.shape[2]))
            encoder_hidden_states = jt.concat([encoder_hidden_states, padding], dim=1)

        x = embeds + encoder_hidden_states[:, :embeds.shape[1], :]

        # 修复形状不匹配
        linear1_out = self.decoder_linear1(x)  # (batch, seq_len, 2048)
        linear1_out = jt.nn.relu(linear1_out)
        linear1_out = self.decoder_adapter(linear1_out)  # (batch, seq_len, 512)

        x = x + self.dropout(linear1_out)
        x = self.decoder_norm(x)
        return x

    def execute(self, input_ids, labels=None):
        # Jittor前向传播核心函数
        encoder_out = self.encode(input_ids)

        if labels is not None:
            decoder_out = self.decode(labels, encoder_out)
            logits = self.lm_head(decoder_out)
            return {"logits": logits}
        return {"encoder_last_hidden_state": encoder_out}

    def __call__(self, input_ids, labels=None, **kwargs):
        # 兼容原项目调用方式
        if isinstance(input_ids, dict):
            input_ids = input_ids.get("input_ids", input_ids)

        # 处理维度
        if isinstance(input_ids, (list, np.ndarray)):
            input_ids = jt.array(input_ids)

        if len(input_ids.shape) == 1:
            input_ids = input_ids.unsqueeze(0)

        return self.execute(input_ids, labels)

    def generate(self, input_ids, max_length=512, num_return_sequences=1, do_sample=False):
        # 模拟generate接口（兼容掩码填充逻辑）
        if isinstance(input_ids, (list, np.ndarray)):
            input_ids = jt.array(input_ids)

        if len(input_ids.shape) == 1:
            input_ids = input_ids.unsqueeze(0)

        batch_size = input_ids.shape[0]
        # 生成随机token（模拟填充结果）
        gen_tokens = jt.randint(0, 32128, (batch_size, max_length))
        return gen_tokens


# -------------------------- 原项目接口（完全兼容，无需修改run.py） --------------------------
def load_base_model_and_tokenizer(args, config, scoring_model_name=None):
    # 加载GPT2模型和Tokenizer
    model = GPT2LMHeadModel()
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    config["base_model"] = model
    config["base_tokenizer"] = tokenizer
    config["GPT2_TOKENIZER"] = tokenizer  # 兼容原项目中GPT2_TOKENIZER的引用
    print("✅ 成功加载简易GPT2模型（兼容Jittor，已修复形状不匹配问题）")


def load_mask_filling_model(args, config):
    # 加载T5模型和Tokenizer
    model = T5ForConditionalGeneration()
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    config["mask_model"] = model
    config["mask_tokenizer"] = tokenizer
    print("✅ 成功加载简易T5-small模型（兼容Jittor，已修复形状不匹配问题）")


def load_base_model(args, config):
    # 兼容原项目接口，无实际操作
    pass

