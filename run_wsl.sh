#!/bin/bash
# Jittor DetectGPT 运行脚本 (WSL2 环境)
# 使用方法: bash run_wsl.sh

# ========================================
# 配置参数
# ========================================

# 设备配置
DEVICE="cpu"  # 选项: cpu, gpu, auto

# 数据配置
MAX_RAW_DATA="15"        # 加载的总样本数（人类+AI）
MIN_SAMPLES="10"          # 最小样本要求
DATASET="builtin"          # 使用内置数据

# 模型配置
BASE_MODEL_NAME="gpt2"                    # 基础模型
MASK_FILLING_MODEL_NAME="t5-small"         # 掩码填充模型
SCORING_MODEL_NAME=""                      # 评分模型（空则使用基础模型）

# 批次配置
BATCH_SIZE="1"                             # 批次大小

# 扰动配置
N_PERTURBATION_ROUNDS="5"                  # 扰动轮数
PCT_WORDS_MASKED="0.15"                     # 掩码单词比例
SPAN_LENGTH="3"                             # 掩码跨度长度

# 输出配置
OUTPUT_DIR="./tmp_results"                   # 结果输出目录
DEBUG="false"                               # 是否启用调试模式

# ========================================
# 运行命令
# ========================================

echo "============================================================"
echo "Jittor DetectGPT 运行配置"
echo "============================================================"
echo "设备: $DEVICE"
echo "样本数: $MAX_RAW_DATA"
echo "基础模型: $BASE_MODEL_NAME"
echo "掩码模型: $MASK_FILLING_MODEL_NAME"
echo "扰动轮数: $N_PERTURBATION_ROUNDS"
echo "============================================================"

python run.py \
    --DEVICE $DEVICE \
    --batch_size $BATCH_SIZE \
    --max_raw_data $MAX_RAW_DATA \
    --min_samples $MIN_SAMPLES \
    --dataset $DATASET \
    --base_model_name $BASE_MODEL_NAME \
    --mask_filling_model_name $MASK_FILLING_MODEL_NAME \
    --scoring_model_name $SCORING_MODEL_NAME \
    --n_perturbation_rounds $N_PERTURBATION_ROUNDS \
    --pct_words_masked $PCT_WORDS_MASKED \
    --span_length $SPAN_LENGTH \
    --output_dir $OUTPUT_DIR \
    --$DEBUG

echo "============================================================"
echo "运行完成！"
echo "结果保存在: $OUTPUT_DIR"
echo "============================================================"
