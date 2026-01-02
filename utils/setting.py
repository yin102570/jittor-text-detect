import os
import json
import datetime


def initial_setup(args, config):
    API_TOKEN_COUNTER = 0  # 保留该变量，保持config兼容性，无实际OpenAI用途

    # 彻底删除所有OpenAI相关判断和导入逻辑
    # 移除原有的 if args.openai_model is not None 分支

    START_DATE = datetime.datetime.now().strftime('%Y-%m-%d')
    START_TIME = datetime.datetime.now().strftime('%H-%M-%S-%f')

    # 关键修复：使用args.output_dir作为基础目录
    base_output_dir = args.output_dir if hasattr(args, 'output_dir') and args.output_dir else "./results"

    # define SAVE_FOLDER as the timestamp - base model name - mask filling model name
    # create it if it doesn't exist
    # 补充：若args无int8/half属性，兜底赋值，避免属性不存在报错
    int8_flag = args.int8 if hasattr(args, 'int8') else False
    half_flag = args.half if hasattr(args, 'half') else False
    precision_string = "int8" if int8_flag else ("fp16" if half_flag else "fp32")

    # 补充：若args无do_top_k/do_top_p属性，兜底赋值，避免属性不存在报错
    do_top_k_flag = args.do_top_k if hasattr(args, 'do_top_k') else False
    do_top_p_flag = args.do_top_p if hasattr(args, 'do_top_p') else False
    sampling_string = "top_k" if do_top_k_flag else ("top_p" if do_top_p_flag else "temp")

    # 补充：若args无output_name属性，兜底赋值，避免属性不存在报错
    output_name = args.output_name if hasattr(args, 'output_name') else ""
    output_subfolder = f"{output_name}/" if output_name else ""

    # 直接使用本地基础模型名称，删除OpenAI相关分支判断
    base_model_name = args.base_model_name.replace('/', '_') if hasattr(args, 'base_model_name') else "gpt2"

    # 补充：若args无scoring_model_name属性，兜底赋值，避免属性不存在报错
    scoring_model_name = args.scoring_model_name if hasattr(args, 'scoring_model_name') else ""
    scoring_model_string = (f"-{scoring_model_name}" if scoring_model_name else "").replace('/', '_')

    # 补充：若args无相关属性，兜底赋值，避免属性不存在报错
    pct_words_masked = args.pct_words_masked if hasattr(args, 'pct_words_masked') else 0.15
    n_perturbation_rounds = args.n_perturbation_rounds if hasattr(args, 'n_perturbation_rounds') else 5
    dataset = args.dataset if hasattr(args, 'dataset') else "WritingPrompts"
    n_samples = args.n_samples if hasattr(args, 'n_samples') else 100

    # 关键修复：使用base_output_dir而不是硬编码的tmp_results
    experiment_folder = f"{output_subfolder}{base_model_name}{scoring_model_string}-{args.mask_filling_model_name}-{sampling_string}/{START_DATE}-{START_TIME}-{precision_string}-{pct_words_masked}-{n_perturbation_rounds}-{dataset}-{n_samples}"
    SAVE_FOLDER = os.path.join(base_output_dir, experiment_folder)

    if not os.path.exists(SAVE_FOLDER):
        os.makedirs(SAVE_FOLDER)
    print(f"📁 保存结果到: {os.path.abspath(SAVE_FOLDER)}")
    print(f"📁 基础输出目录: {base_output_dir}")

    # write args to file
    # 兼容args为Namespace或字典类型
    args_dict = vars(args) if hasattr(args, '__dict__') else args
    with open(os.path.join(SAVE_FOLDER, "args.json"), "w") as f:
        json.dump(args_dict, f, indent=4)

    config["START_DATE"] = START_DATE
    config["START_TIME"] = START_TIME
    config["base_model_name"] = base_model_name
    config["SAVE_FOLDER"] = SAVE_FOLDER
    config["API_TOKEN_COUNTER"] = API_TOKEN_COUNTER
    # 关键修复：确保config中有output_dir
    config["output_dir"] = base_output_dir


def set_experiment_config(args, config):
    """
    Parses the runtime arguments for setting the experiment configuration.
    """
    # 补充：若args无cache_dir属性，兜底赋值，避免属性不存在报错
    cache_dir = args.cache_dir if hasattr(args, 'cache_dir') else "./cache"
    os.environ["XDG_CACHE_HOME"] = cache_dir
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    print(f"Using cache dir {cache_dir}")

    # 补充：所有args属性均添加兜底判断，避免属性不存在报错
    mask_filling_model_name = args.mask_filling_model_name if hasattr(args, 'mask_filling_model_name') else "t5-small"
    n_samples = args.n_samples if hasattr(args, 'n_samples') else 100
    batch_size = args.batch_size if hasattr(args, 'batch_size') else 1
    # 补充：若args无n_perturbation_list属性，兜底赋值
    n_perturbation_list = args.n_perturbation_list if hasattr(args, 'n_perturbation_list') else "5"
    n_perturbation_rounds = args.n_perturbation_rounds if hasattr(args, 'n_perturbation_rounds') else 5
    n_similarity_samples = args.n_similarity_samples if hasattr(args, 'n_similarity_samples') else 10

    config["mask_filling_model_name"] = mask_filling_model_name
    config["n_samples"] = n_samples
    config["batch_size"] = batch_size
    config["n_perturbation_list"] = [int(x) for x in n_perturbation_list.split(",")]
    config["n_perturbation_rounds"] = n_perturbation_rounds
    config["n_similarity_samples"] = n_similarity_samples
    config["cache_dir"] = cache_dir