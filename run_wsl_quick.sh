#!/bin/bash
# Jittor DetectGPT 快速测试脚本 (WSL2 环境)
# 用于验证环境和代码是否正常工作

# ========================================
# 快速测试配置（小样本，少扰动）
# ========================================

echo "============================================================"
echo "Jittor DetectGPT 快速测试"
echo "============================================================"
echo "配置说明:"
echo "  - 样本数: 20 (10人类 + 10AI)"
echo "  - 扰动轮数: 3 (默认5轮)"
echo "  - 批次大小: 1"
echo "  - 设备: CPU"
echo "  - 调试模式: 启用"
echo "============================================================"

python run.py \
    --DEVICE cpu \
    --batch_size 1 \
    --max_raw_data 20 \
    --min_samples 10 \
    --dataset builtin \
    --base_model_name gpt2 \
    --mask_filling_model_name t5-small \
    --n_perturbation_rounds 3 \
    --pct_words_masked 0.15 \
    --span_length 3 \
    --output_dir ./tmp_quick_results \
    --debug

echo ""
echo "============================================================"
echo "快速测试完成！"
echo "如果看到上面的输出没有错误，说明环境配置正常。"
echo "============================================================"
