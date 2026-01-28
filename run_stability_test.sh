#!/bin/bash
# 实验稳定性测试脚本
# 运行多次实验并统计结果，减少随机性影响

# ========================================
# 配置参数
# ========================================

NUM_RUNS=5              # 运行次数
MAX_RAW_DATA=50         # 样本数（多次运行以节省时间）
N_PERTURBATION_ROUNDS=5 # 扰动轮数
BASE_MODEL="gpt2"       # 基础模型
DEVICE="cpu"              # 设备

# ========================================
# 创建结果目录
# ========================================

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_DIR="./stability_results_${TIMESTAMP}"
mkdir -p "$RESULTS_DIR"

echo "============================================================"
echo "实验稳定性测试"
echo "============================================================"
echo "配置:"
echo "  - 运行次数: $NUM_RUNS"
echo "  - 每次样本数: $MAX_RAW_DATA"
echo "  - 扰动轮数: $N_PERTURBATION_ROUNDS"
echo "  - 基础模型: $BASE_MODEL"
echo "  - 设备: $DEVICE"
echo "============================================================"

# 存储所有运行的结果
declare -a roc_auc_array
declare -a pr_auc_array

# ========================================
# 运行多次实验
# ========================================

for i in $(seq 1 $NUM_RUNS); do
    echo ""
    echo "------------------------------------------------------------"
    echo "运行第 ${i}/${NUM_RUNS} 次"
    echo "------------------------------------------------------------"

    # 运行实验
    python run.py \
        --DEVICE "$DEVICE" \
        --max_raw_data "$MAX_RAW_DATA" \
        --min_samples 10 \
        --dataset builtin \
        --base_model_name "$BASE_MODEL" \
        --mask_filling_model_name t5-small \
        --n_perturbation_rounds "$N_PERTURBATION_ROUNDS" \
        --pct_words_masked 0.15 \
        --span_length 3 \
        --output_dir "$RESULTS_DIR/run_${i}"

    # 检查是否成功
    if [ $? -eq 0 ]; then
        echo "[OK] 第 ${i}/${NUM_RUNS} 次运行成功"

        # 提取结果
        ROC_AUC=$(grep -oP '(?<="roc_auc": )([0-9.]+\.?[0-9]*)' "$RESULTS_DIR/run_${i}"/*_results.json | head -1)
        PR_AUC=$(grep -oP '(?<="pr_auc": )([0-9.]+\.?[0-9]*)' "$RESULTS_DIR/run_${i}"/*_results.json | head -1)

        # 添加到数组
        if [ ! -z "$ROC_AUC" ]; then
            roc_auc_array+=("$ROC_AUC")
        fi

        if [ ! -z "$PR_AUC" ]; then
            pr_auc_array+=("$PR_AUC")
        fi
    else
        echo "[ERROR] 第 ${i}/${NUM_RUNS} 次运行失败"
    fi
done

# ========================================
# 统计结果
# ========================================

echo ""
echo "============================================================"
echo "稳定性测试完成！"
echo "============================================================"

# 统计运行成功次数
SUCCESS_COUNT=0
for auc in "${roc_auc_array[@]}"; do
    SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
done

echo "成功运行: ${SUCCESS_COUNT}/${NUM_RUNS} 次"

# 计算 ROC AUC 统计
if [ $SUCCESS_COUNT -gt 0 ]; then
    echo ""
    echo "ROC AUC 统计:"
    echo "${roc_auc_array[@]}" | tr ' ' '\n' | sort -n

    # 计算统计量
    MIN_AUC=$(echo "${roc_auc_array[@]}" | tr ' ' '\n' | sort -n | head -1)
    MAX_AUC=$(echo "${roc_auc_array[@]}" | tr ' ' '\n' | sort -n | tail -1)
    AVG_AUC=$(echo "(${roc_auc_array[@]})" | tr ' ' '\n' | awk '{s=0; for(i=1;i<=NF;i++) s+=$1; print s/NF}')

    echo "  最小值: $MIN_AUC"
    echo "  最大值: $MAX_AUC"
    echo "  平均值: $AVG_AUC"
    echo "  标准差: $(echo "${roc_auc_array[@]}" | tr ' ' '\n' | awk '{sum+=$1; sum2+=$1*$1} END {print sqrt((sum2-sum*sum/NF)/NF)}')"

    # 生成汇总报告
    SUMMARY_FILE="$RESULTS_DIR/stability_summary.txt"
    cat > "$SUMMARY_FILE" << EOF
实验稳定性测试汇总报告
====================================
配置:
  - 运行次数: $NUM_RUNS
  - 样本数: $MAX_RAW_DATA
  - 扰动轮数: $N_PERTURBATION_ROUNDS
  - 基础模型: $BASE_MODEL
  - 设备: $DEVICE

结果:
  - 成功运行: ${SUCCESS_COUNT}/${NUM_RUNS} 次

ROC AUC 统计:
  - 最小值: $MIN_AUC
  - 最大值: $MAX_AUC
  - 平均值: $AVG_AUC
  - 标准差: $(echo "${roc_auc_array[@]}" | tr ' ' '\n' | awk '{sum+=$1; sum2+=$1*$1} END {print sqrt((sum2-sum*sum/NF)/NF)}')

所有 ROC AUC 值:
EOF

    for auc in "${roc_auc_array[@]}"; do
        echo "  - $auc" >> "$SUMMARY_FILE"
    done

    echo ""
    echo "[OK] 汇总报告已保存到: $SUMMARY_FILE"
    echo "[OK] 详细结果保存在: $RESULTS_DIR/"
else
    echo ""
    echo "[ERROR] 所有运行均失败，无法生成统计"
fi

echo ""
echo "============================================================"
echo "测试完成"
echo "============================================================"
