# Jittor 在 Windows 上的安装和配置指南

## 问题说明
Jittor 在 Windows 上首次运行时会编译 C++ 代码，如果缺少必要的编译工具会导致错误。

## 解决方案

### 方案1：安装 Microsoft C++ Build Tools（推荐）

1. **下载并安装 Microsoft C++ Build Tools**
   - 访问：https://visualstudio.microsoft.com/visual-cpp-build-tools/
   - 下载并运行安装程序
   - 在安装界面勾选：
     - **Desktop development with C++** (使用 C++ 的桌面开发)
     - 确保 Windows 10/11 SDK 已勾选
   - 点击安装

2. **设置环境变量**
   ```powershell
   # 添加 MSVC 编译器到 PATH
   setx PATH "%PATH%;C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Tools\MSVC\14.xx.xxxxx\bin\Hostx64\x64"
   ```

3. **重新安装 Jittor**
   ```bash
   pip uninstall jittor
   pip install jittor==1.3.10
   ```

### 方案2：使用 WSL2（Windows Subsystem for Linux）

1. **启用 WSL2**
   ```powershell
   wsl --install
   ```

2. **在 WSL2 中安装 Jittor**
   ```bash
   # 更新系统
   sudo apt update && sudo apt upgrade -y

   # 安装 Python 和 pip
   sudo apt install python3 python3-pip -y

   # 安装 Jittor
   pip install jittor==1.3.10

   # 安装项目依赖
   pip install -r requirements.txt
   ```

3. **在 WSL2 中运行项目**
   ```bash
   cd /mnt/d/HuaweiMoveData/Users/asdf1/Desktop/jittor-text-detect
   python run.py --dataset builtin --max_raw_data 50 --min_samples 20 --DEVICE cpu
   ```

### 方案3：使用预编译的 Jittor

Jittor 官方可能提供预编译版本，尝试安装：

```bash
pip install jittor==1.3.10 --prefer-binary
```

## 当前项目样本数量问题分析

根据 `run.py` 代码分析，当前已经内置了大量数据：

### 数据规模（第17-239行）
- **人类文本**：50 条（第23-74行）+ 50 条（第134-184行）= **100 条**
- **AI 文本**：50 条（第78-129行）+ 50 条（第188-238行）= **100 条**
- **总计**：**200 条样本**

### 数据加载逻辑（第246-256行）
```python
n_samples = min(args.max_raw_data // 2, len(all_human_texts), len(all_ai_texts))
n_samples = max(n_samples, 20)  # 至少20个样本
```

**结论**：代码已经内置了充足的数据（200条），样本数量不足的问题可能是：
1. 参数 `max_raw_data` 设置过小
2. 数据验证步骤过滤掉了大量文本（<50字符的文本）

## 解决样本数量不足的方法

### 方法1：调整运行参数
```bash
# 使用更大的样本数量
python run.py --dataset builtin --max_raw_data 200 --min_samples 20 --DEVICE cpu
```

### 方法2：降低文本长度过滤要求
检查 `detectGPT.py` 第36行：
```python
valid_o = isinstance(o, str) and o.strip() and len(o.strip()) > 50
```
将 `> 50` 改为 `> 20` 或 `> 10` 可以接受更多短文本。

### 方法3：检查数据格式
确保 `load_builtin_data_with_labels` 函数返回的数据格式正确：
- `original`: 人类文本列表
- `samples`: AI 文本列表
- `labels`: 标签列表（0=人类，1=AI）

## 快速测试命令

```bash
# 在 WSL2 或解决编译问题后运行
cd /path/to/jittor-text-detect
python run.py --dataset builtin --max_raw_data 100 --min_samples 20 --DEVICE cpu --debug
```

## 预期输出

成功运行后应该看到：
```
✅ 加载带标签数据：人类文本 50 条，AI文本 50 条
📊 总样本数：100 条
✅ 数据格式有效: 包含 50 条人类文本，50 条AI文本
✅ Jittor自动适配设备: CPU
```

## 常见问题

### Q: 编译错误持续出现？
A: 使用 WSL2 方案，在 Linux 环境中运行更稳定。

### Q: 样本数量仍然不足？
A: 检查 `--max_raw_data` 参数，确保值 ≥ 40（因为需要除以2分配给人类和AI文本）。

### Q: 运行速度慢？
A: 如果有 GPU，使用 `--DEVICE gpu` 可以大幅加速。
