# WSL2 运行指南

## 激活环境并运行

你提供的命令是正确的，以下是详细说明：

### 步骤 1: 激活 Conda 环境

```bash
conda activate jittor-cpu-wsl
```

### 步骤 2: 切换到项目目录

```bash
cd /mnt/d/HuaweiMoveData/Users/asdf1/Desktop/jittor-text-detect
```

### 步骤 3: 运行项目

#### 选项 A: 使用你提供的命令
```bash
python run.py --DEVICE cpu --batch_size 1 --max_raw_data 15 --base_model_name gpt2 --mask_filling_model_name t5-small
```

#### 选项 B: 使用快速测试脚本（推荐第一次使用）
```bash
bash run_wsl_quick.sh
```

#### 选项 C: 使用完整配置脚本
```bash
bash run_wsl.sh
```

## 参数说明

### 你命令中的参数解释

| 参数 | 值 | 说明 |
|------|-----|------|
| `--DEVICE` | cpu | 使用 CPU 运行（WSL2 中可能没有 GPU 支持） |
| `--batch_size` | 1 | 批次大小，1 表示逐个处理 |
| `--max_raw_data` | 15 | 加载的总样本数（人类+AI，各约 7-8 条） |
| `--base_model_name` | gpt2 | 基础语言模型，用于计算似然值 |
| `--mask_filling_model_name` | t5-small | 掩码填充模型，用于生成扰动文本 |

### 推荐的参数组合

#### 1. 快速测试（验证环境）- 5-10 分钟
```bash
python run.py \
    --DEVICE cpu \
    --batch_size 1 \
    --max_raw_data 20 \
    --min_samples 10 \
    --base_model_name gpt2 \
    --mask_filling_model_name t5-small \
    --n_perturbation_rounds 3 \
    --debug
```

**特点**：
- 20 个样本（10 人类 + 10 AI）
- 3 轮扰动（默认 5 轮）
- 启用调试模式
- 运行时间短，适合快速验证

#### 2. 中等规模测试 - 20-30 分钟
```bash
python run.py \
    --DEVICE cpu \
    --batch_size 1 \
    --max_raw_data 50 \
    --min_samples 10 \
    --base_model_name gpt2 \
    --mask_filling_model_name t5-small \
    --n_perturbation_rounds 5
```

**特点**：
- 50 个样本（25 人类 + 25 AI）
- 5 轮扰动（默认）
- 更可靠的统计结果

#### 3. 完整实验 - 1-2 小时
```bash
python run.py \
    --DEVICE cpu \
    --batch_size 4 \
    --max_raw_data 200 \
    --min_samples 10 \
    --base_model_name gpt2 \
    --mask_filling_model_name t5-small \
    --n_perturbation_rounds 10
```

**特点**：
- 200 个样本（100 人类 + 100 AI）
- 10 轮扰动
- 更大的批次大小（如果内存允许）

## 预期输出流程

### 1. 初始化阶段
```
============================================================
Jittor文本检测与生成（内置数据版）
============================================================
✅ Jittor自动适配设备: CPU
📁 保存结果到: ./tmp_results/...
Using cache dir ./cache
```

### 2. 数据加载阶段
```
📥 正在加载内置数据...
[OK] 加载带标签数据：人类文本 X 条，AI文本 X 条
🔍 开始数据有效性校验...
✅ 数据格式有效: 包含 X 条人类文本，X 条AI文本
```

### 3. 模型加载阶段
```
Loading model: gpt2
Loading tokenizer: gpt2
Loading mask model: t5-small
```

### 4. 计算阶段（最耗时）
```
🚀 开始运行基线模型...
✅ LikelihoodScorer进度: 5/10 (50.0%)
✅ LikelihoodScorer进度: 10/10 (100.0%)
```

### 5. 结果输出
```
🎯 最终结果:
ROC AUC: 0.XXXX
PR AUC: 0.XXXX
💾 正在保存结果...
✅ 所有结果已保存到: ...
```

## 常见问题

### Q1: 运行时间太长怎么办？
**A**:
- 减少样本数：`--max_raw_data 20`
- 减少扰动轮数：`--n_perturbation_rounds 3`
- 使用更小的模型：`--base_model_name gpt2-small`

### Q2: 内存不足错误？
**A**:
- 减小批次大小：`--batch_size 1`
- 减少样本数：`--max_raw_data 20`
- 使用更小的模型

### Q3: 模型下载失败？
**A**:
```bash
# 设置 HuggingFace 镜像源
export HF_ENDPOINT=https://hf-mirror.com

# 或手动下载模型后放到缓存目录
```

### Q4: 如何查看结果？
**A**:
```bash
# 查看结果目录
ls -lh tmp_results/

# 查看 JSON 结果文件
cat tmp_results/*/detectgpt_results.json | python -m json.tool

# 查看 AUC 指标
grep -i "auc" tmp_results/*/final_results.json
```

## 完整命令示例

### 你原始命令的增强版
```bash
conda activate jittor-cpu-wsl
cd /mnt/d/HuaweiMoveData/Users/asdf1/Desktop/jittor-text-detect

python run.py \
    --DEVICE cpu \
    --batch_size 1 \
    --max_raw_data 15 \
    --min_samples 10 \
    --dataset builtin \
    --base_model_name gpt2 \
    --mask_filling_model_name t5-small \
    --n_perturbation_rounds 5 \
    --pct_words_masked 0.15 \
    --span_length 3 \
    --output_dir ./tmp_results \
    --debug
```

### 使用配置脚本（推荐）
```bash
# 给脚本添加执行权限
chmod +x run_wsl_quick.sh
chmod +x run_wsl.sh

# 快速测试
bash run_wsl_quick.sh

# 完整运行
bash run_wsl.sh
```

## 性能优化建议

### 如果 WSL2 支持 GPU
```bash
# 检查是否有 GPU
nvidia-smi

# 如果有 GPU，修改参数
python run.py --DEVICE gpu --batch_size 8 --max_raw_data 200
```

### 如果只有 CPU
```bash
# 使用小批次、少样本、少扰动
python run.py \
    --DEVICE cpu \
    --batch_size 1 \
    --max_raw_data 20 \
    --n_perturbation_rounds 3
```

## 后续步骤

1. **第一次运行**：使用快速测试脚本验证环境
   ```bash
   bash run_wsl_quick.sh
   ```

2. **检查结果**：查看输出目录中的 JSON 文件
   ```bash
   ls tmp_quick_results/
   ```

3. **逐步扩大**：如果第一次运行成功，逐步增加样本数
   - 20 -> 50 -> 100 -> 200

4. **记录配置**：保存成功的配置参数以便复现

## 文件权限设置

在 WSL2 中使用脚本前，可能需要设置权限：
```bash
cd /mnt/d/HuaweiMoveData/Users/asdf1/Desktop/jittor-text-detect
chmod +x *.sh
```

## 总结

你的运行命令是正确的！只需要：
1. 激活 Conda 环境
2. 切换到项目目录
3. 运行命令

建议先用快速测试版本验证，然后再运行完整的实验。
