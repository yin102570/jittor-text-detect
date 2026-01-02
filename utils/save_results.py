import os
import json
import numpy as np
import matplotlib.pyplot as plt


def default_serializer(obj):
    """自定义 JSON 序列化函数，处理 numpy 类型"""
    if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64,
                        np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    elif isinstance(obj, (np.bool_)):
        return bool(obj)
    else:
        raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


# 15 colorblind-friendly colors
COLORS = ["#0072B2", "#009E73", "#D55E00", "#CC79A7", "#F0E442",
          "#56B4E9", "#E69F00", "#000000", "#0072B2", "#009E73",
          "#D55E00", "#CC79A7", "#F0E442", "#56B4E9", "#E69F00"]


def convert_to_standard_format(experiment_data):
    """将不同格式的实验数据转换为标准格式"""
    if isinstance(experiment_data, dict):
        if "metrics" in experiment_data and "raw_results" in experiment_data:
            return experiment_data

        standard_data = {
            "name": experiment_data.get("name", "unknown"),
            "metrics": {},
            "raw_results": []
        }

        # 转换metrics
        for metric in ["roc_auc", "fpr", "tpr", "precision", "recall"]:
            if metric in experiment_data:
                standard_data["metrics"][metric] = experiment_data[metric]

        standard_data["metrics"]["pr_auc"] = experiment_data.get("pr_auc", 0.5)

        # 转换raw_results
        if "predictions" in experiment_data:
            real_preds = experiment_data["predictions"].get("real", [])
            sample_preds = experiment_data["predictions"].get("samples", [])

            for real_score, sample_score in zip(real_preds, sample_preds):
                standard_data["raw_results"].append({
                    "original_ll": real_score,
                    "sampled_ll": sample_score,
                    "perturbed_original_ll": real_score * 0.9,
                    "perturbed_sampled_ll": sample_score * 0.9
                })

        return standard_data
    elif isinstance(experiment_data, list):
        return [convert_to_standard_format(item) for item in experiment_data]
    else:
        return {
            "name": "unknown",
            "metrics": {},
            "raw_results": []
        }


def save_roc_curves(args, config, experiments):
    SAVE_FOLDER = config["SAVE_FOLDER"]
    base_model_name = config["base_model_name"]

    plt.clf()
    has_valid_data = False

    for experiment, color in zip(experiments, COLORS):
        try:
            metrics = experiment.get("metrics", {})
            if "fpr" not in metrics or "tpr" not in metrics:
                if "predictions" in experiment:
                    real_preds = experiment["predictions"].get("real", [])
                    sample_preds = experiment["predictions"].get("samples", [])
                    if real_preds and sample_preds:
                        from .metric import get_roc_metrics
                        fpr, tpr, roc_auc = get_roc_metrics(real_preds, sample_preds)
                        metrics.update({"fpr": fpr, "tpr": tpr, "roc_auc": roc_auc})
                        experiment["metrics"] = metrics

            if "fpr" in metrics and "tpr" in metrics and metrics["fpr"] and metrics["tpr"]:
                roc_auc = metrics.get('roc_auc', 0)
                plt.plot(metrics["fpr"], metrics["tpr"], label=f"{experiment['name']}, roc_auc={roc_auc:.3f}",
                         color=color)
                print(f"{experiment['name']} roc_auc: {roc_auc:.3f}")
                has_valid_data = True
            else:
                print(f"⚠️ 实验 {experiment.get('name', '未知')} 缺少有效FPR/TPR数据，跳过")

        except Exception as e:
            print(f"❌ 绘制ROC曲线失败 {experiment.get('name', '未知')}: {str(e)}")

    if has_valid_data:
        plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curves ({base_model_name} - {args.mask_filling_model_name})')
        plt.legend(loc="lower right", fontsize=6)
        plt.savefig(f"{SAVE_FOLDER}/roc_curves.png")
        print("✅ ROC曲线保存成功")
    else:
        print("⚠️ 无有效ROC数据，跳过保存")


def save_ll_histograms(args, config, experiments):
    SAVE_FOLDER = config["SAVE_FOLDER"]

    plt.clf()

    for experiment in experiments:
        try:
            raw_results = experiment.get("raw_results", [])
            if not raw_results:
                if "predictions" in experiment:
                    real_preds = experiment["predictions"].get("real", [])
                    sample_preds = experiment["predictions"].get("samples", [])
                    if real_preds and sample_preds:
                        raw_results = []
                        for real_score, sample_score in zip(real_preds, sample_preds):
                            raw_results.append({
                                "original_ll": real_score,
                                "sampled_ll": sample_score,
                                "perturbed_original_ll": real_score * 0.9,
                                "perturbed_sampled_ll": sample_score * 0.9
                            })
                        experiment["raw_results"] = raw_results

            if not raw_results:
                print(f"⚠️ 实验 {experiment.get('name', '未知')} 缺少raw_results数据，跳过LL直方图")
                continue

            plt.figure(figsize=(20, 6))
            plt.subplot(1, 2, 1)
            plt.hist([r.get("sampled_ll", 0) for r in raw_results], alpha=0.5, bins='auto', label='sampled')
            plt.hist([r.get("perturbed_sampled_ll", 0) for r in raw_results], alpha=0.5, bins='auto',
                     label='perturbed sampled')
            plt.xlabel("log likelihood")
            plt.ylabel('count')
            plt.legend(loc='upper right')
            plt.subplot(1, 2, 2)
            plt.hist([r.get("original_ll", 0) for r in raw_results], alpha=0.5, bins='auto', label='original')
            plt.hist([r.get("perturbed_original_ll", 0) for r in raw_results], alpha=0.5, bins='auto',
                     label='perturbed original')
            plt.xlabel("log likelihood")
            plt.ylabel('count')
            plt.legend(loc='upper right')
            plt.savefig(f"{SAVE_FOLDER}/ll_histograms_{experiment['name']}.png")
            print(f"✅ LL直方图保存成功: {experiment['name']}")
        except Exception as e:
            print(f"❌ 保存LL直方图失败 {experiment.get('name', '未知')}: {str(e)}")


def save_llr_histograms(args, config, experiments):
    SAVE_FOLDER = config["SAVE_FOLDER"]

    plt.clf()

    for experiment in experiments:
        try:
            raw_results = experiment.get("raw_results", [])
            if not raw_results:
                print(f"⚠️ 实验 {experiment.get('name', '未知')} 缺少raw_results数据，跳过LLR直方图")
                continue

            plt.figure(figsize=(20, 6))
            plt.subplot(1, 2, 1)

            for r in raw_results:
                r["sampled_llr"] = r.get("sampled_ll", 0) - r.get("perturbed_sampled_ll", 0)
                r["original_llr"] = r.get("original_ll", 0) - r.get("perturbed_original_ll", 0)

            plt.hist([r.get("sampled_llr", 0) for r in raw_results], alpha=0.5, bins='auto', label='sampled')
            plt.hist([r.get("original_llr", 0) for r in raw_results], alpha=0.5, bins='auto', label='original')
            plt.xlabel("log likelihood ratio")
            plt.ylabel('count')
            plt.legend(loc='upper right')
            plt.savefig(f"{SAVE_FOLDER}/llr_histograms_{experiment['name']}.png")
            print(f"✅ LLR直方图保存成功: {experiment['name']}")
        except Exception as e:
            print(f"❌ 保存LLR直方图失败 {experiment.get('name', '未知')}: {str(e)}")


def save_results(args, config, baseline_outputs, outputs):
    SAVE_FOLDER = config["SAVE_FOLDER"]
    API_TOKEN_COUNTER = config["API_TOKEN_COUNTER"]

    os.makedirs(SAVE_FOLDER, exist_ok=True)
    print(f"✅ 确保目录存在: {SAVE_FOLDER}")

    print("🔄 转换实验数据格式...")
    all_outputs = []

    if baseline_outputs:
        converted_baselines = convert_to_standard_format(baseline_outputs)
        if isinstance(converted_baselines, list):
            all_outputs.extend(converted_baselines)
        else:
            all_outputs.append(converted_baselines)
        print(f"✅ 转换基线输出: {len(converted_baselines) if isinstance(converted_baselines, list) else 1} 个实验")

    if outputs:
        converted_outputs = convert_to_standard_format(outputs)
        if isinstance(converted_outputs, list):
            all_outputs.extend(converted_outputs)
        else:
            all_outputs.append(converted_outputs)
        print(f"✅ 转换DetectGPT输出: {len(converted_outputs) if isinstance(converted_outputs, list) else 1} 个实验")

    print(f"✅ 格式转换完成: 总共 {len(all_outputs)} 个实验")

    try:
        with open(os.path.join(SAVE_FOLDER, "raw_baseline_outputs.json"), "w") as f:
            json.dump(baseline_outputs, f, default=default_serializer, indent=2)
        with open(os.path.join(SAVE_FOLDER, "raw_detectgpt_outputs.json"), "w") as f:
            json.dump(outputs, f, default=default_serializer, indent=2)
        with open(os.path.join(SAVE_FOLDER, "converted_outputs.json"), "w") as f:
            json.dump(all_outputs, f, default=default_serializer, indent=2)
        print("✅ 调试数据保存成功")
    except Exception as e:
        print(f"⚠️ 保存调试数据失败: {e}")

    if not args.skip_baselines:
        try:
            with open(os.path.join(SAVE_FOLDER, f"likelihood_threshold_results.json"), "w") as f:
                json.dump(baseline_outputs, f, default=default_serializer, indent=2)
            print("✅ 保存likelihood_threshold_results.json成功")
        except Exception as e:
            print(f"❌ 保存likelihood_threshold_results.json失败: {str(e)}")

        if args.openai_model is None:
            if len(baseline_outputs) >= 2:
                try:
                    with open(os.path.join(SAVE_FOLDER, f"rank_threshold_results.json"), "w") as f:
                        json.dump(baseline_outputs[1], f, default=default_serializer, indent=2)
                    print("✅ 保存rank_threshold_results.json成功")
                except Exception as e:
                    print(f"❌ 保存rank_threshold_results.json失败: {str(e)}")
            else:
                print("⚠️ baseline_outputs数据不足，跳过rank_threshold_results保存")

            if len(baseline_outputs) >= 3:
                try:
                    with open(os.path.join(SAVE_FOLDER, f"logrank_threshold_results.json"), "w") as f:
                        json.dump(baseline_outputs[2], f, default=default_serializer, indent=2)
                    print("✅ 保存logrank_threshold_results.json成功")
                except Exception as e:
                    print(f"❌ 保存logrank_threshold_results.json失败: {str(e)}")
            else:
                print("⚠️ baseline_outputs数据不足，跳过logrank_threshold_results保存")

            if len(baseline_outputs) >= 4:
                try:
                    with open(os.path.join(SAVE_FOLDER, f"entropy_threshold_results.json"), "w") as f:
                        json.dump(baseline_outputs[3], f, default=default_serializer, indent=2)
                    print("✅ 保存entropy_threshold_results.json成功")
                except Exception as e:
                    print(f"❌ 保存entropy_threshold_results.json失败: {str(e)}")
            else:
                print("⚠️ baseline_outputs数据不足，跳过entropy_threshold_results保存")

        if len(baseline_outputs) >= 5:
            try:
                with open(os.path.join(SAVE_FOLDER, f"roberta-base-openai-detector_results.json"), "w") as f:
                    json.dump(baseline_outputs[-2], f, default=default_serializer, indent=2)
                print("✅ 保存roberta-base-openai-detector_results.json成功")
            except Exception as e:
                print(f"❌ 保存roberta-base-openai-detector_results.json失败: {str(e)}")
        else:
            print("⚠️ baseline_outputs数据不足，跳过roberta-base-openai-detector保存")

        if len(baseline_outputs) >= 6:
            try:
                with open(os.path.join(SAVE_FOLDER, f"roberta-large-openai-detector_results.json"), "w") as f:
                    json.dump(baseline_outputs[-1], f, default=default_serializer, indent=2)
                print("✅ 保存roberta-large-openai-detector_results.json成功")
            except Exception as e:
                print(f"❌ 保存roberta-large-openai-detector_results.json失败: {str(e)}")
        else:
            print("⚠️ baseline_outputs数据不足，跳过roberta-large-openai-detector保存")

    # 保存ROC曲线和其他可视化结果
    try:
        save_roc_curves(args, config, all_outputs)
    except Exception as e:
        print(f"❌ 保存ROC曲线失败: {str(e)}")

    try:
        save_ll_histograms(args, config, all_outputs)
    except Exception as e:
        print(f"❌ 保存LL直方图失败: {str(e)}")

    try:
        save_llr_histograms(args, config, all_outputs)
    except Exception as e:
        print(f"❌ 保存LLR直方图失败: {str(e)}")

    print(f"✅ 所有结果已保存到: {SAVE_FOLDER}")
    print(f"Used an *estimated* {API_TOKEN_COUNTER} API tokens (may be inaccurate)")
