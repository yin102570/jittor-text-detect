# AUC性能优化报告

## 问题诊断

### 原始问题
- **ROC AUC**: 0.57（仅略高于随机猜测0.5）
- **PR AUC**: 0.60
- **分数分布重叠度高**：人类文本和AI文本的分数无法区分

### 根本原因分析

#### 1. **致命错误：扰动分数造假**
**位置**: `utils/baselines/detectGPT.py:154-155`

```python
# ❌ 错误代码
"perturbed_original_ll": orig_score * 0.9,
"perturbed_sampled_ll": samp_score * 0.9
```

**问题**：扰动似然被简化为原分数乘以0.9，导致：
- 人类和AI文本的扰动分数完全相关
- 曲率分数 = 原始分数 - 扰动分数 = 原始分数 * 0.1
- 人类和AI文本的曲率分数分布完全重叠
- AUC接近随机猜测（0.5）

#### 2. **特征工程不足**
- 仅使用6个基础特征
- 未进行特征归一化
- 模型容量不足（100棵树）

#### 3. **扰动策略不优**
- 掩码比例15%可能过小
- 扰动轮数5次不足
- 扰动跨度3可能过大

---

## 优化方案

### 优化1：真实计算扰动分数（核心修复）

**位置**: `utils/baselines/detectGPT.py:130-173`

```python
# ✅ 修复代码：真实计算扰动分数
perturbed_original_scores = []
perturbed_sampled_scores = []

# 计算人类文本的扰动分数
for i, text in enumerate(cleaned_original):
    perturbed_lls = []
    for _ in range(n_perturbations):
        perturbed_text = scorer._perturb_text(text)
        if perturbed_text != text:
            perturbed_ll = get_ll(args, config, [perturbed_text])[0]
            perturbed_lls.append(perturbed_ll)
    avg_perturbed_ll = np.mean(perturbed_lls) if perturbed_lls else original_scores[i]
    perturbed_original_scores.append(avg_perturbed_ll)

# 计算AI文本的扰动分数
for i, text in enumerate(cleaned_samples):
    perturbed_lls = []
    for _ in range(n_perturbations):
        perturbed_text = scorer._perturb_text(text)
        if perturbed_text != text:
            perturbed_ll = get_ll(args, config, [perturbed_text])[0]
            perturbed_lls.append(perturbed_ll)
    avg_perturbed_ll = np.mean(perturbed_lls) if perturbed_lls else sampled_scores[i]
    perturbed_sampled_scores.append(avg_perturbed_ll)

# 使用曲率分数作为检测分数
curvature_original = [orig - pert for orig, pert in zip(original_scores, perturbed_original_scores)]
curvature_sampled = [samp - pert for samp, pert in zip(sampled_scores, perturbed_sampled_scores)]
```

**预期提升**：ROC AUC 从 0.57 → **0.75+**

---

### 优化2：归一化曲率分数

**位置**: `utils/baselines/model.py:464-470`

```python
# ✅ 修复代码：使用归一化曲率
avg_perturbed_ll = np.mean(perturbed_lls)
std_perturbed_ll = np.std(perturbed_lls) if len(perturbed_lls) > 1 else 0

# 归一化曲率 = (原始似然 - 平均扰动似然) / (标准差 + 小常数)
score = (original_ll - avg_perturbed_ll) / (std_perturbed_ll + 1e-8)
```

**原理**：归一化使得不同文本的分数具有可比性，增强区分度

---

### 优化3：增强特征工程

**位置**: `utils/baselines/ensemble.py:23-72`

**新增8维特征**（原6维）：

| 特征 | 说明 | 优化点 |
|------|------|--------|
| 原始似然值（归一化） | 标准化处理 | 消除量纲影响 |
| 平均扰动似然值（归一化） | 标准化处理 | 增强可比性 |
| 似然值方差 | 扰动稳定性 | 保留 |
| 文本长度（归一化） | 标准化处理 | 消除量纲影响 |
| 曲率 | 原始-扰动差值 | 保留 |
| 相对曲率 | 曲率/长度 | 保留 |
| **熵特征** | **-ll * log\|ll\|** | **新增：文本复杂度** |
| **扰动似然比率** | **原始/扰动** | **新增：相对变化** |

---

### 优化4：提升模型容量

**位置**: `utils/baselines/ensemble.py:17-21`

```python
# ❌ 原配置
self.rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
self.gb = GradientBoostingClassifier(n_estimators=100, random_state=42)

# ✅ 优化后
self.rf = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
self.gb = GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42)
```

**提升**：
- 树数量：100 → 200（提升2倍）
- 最大深度：无限制 → 15（避免过拟合）
- 学习率：默认 → 0.1（优化收敛速度）

---

### 优化5：优化扰动参数

**位置**: `run.py:369-374`

```python
# ❌ 原参数
--pct_words_masked 0.15      # 掩码比例15%
--span_length 3              # 跨度3
--n_perturbation_rounds 5   # 扰动轮数5

# ✅ 优化后
--pct_words_masked 0.20      # 掩码比例20%（增强扰动）
--span_length 2              # 跨度2（更精细）
--n_perturbation_rounds 10   # 扰动轮数10（更稳定）
```

**原理**：
- 增大掩码比例：产生更多变化
- 减小跨度：更精细的扰动
- 增加轮数：更稳定的估计

---

### 优化6：修复集成分类器Bug

**Bug 1**: `ensemble.py:137` - 变量名错误
```python
# ❌ 错误
final_prob = (rf_pred * rf_weight + gb_prob * gb_weight) / total_weight

# ✅ 修复
final_prob = (rf_prob * rf_weight + gb_prob * gb_weight) / total_weight
```

**Bug 2**: `ensemble.py:180` - 扰动数据维度问题
```python
# ✅ 修复：确保2维数组
if perturbed_lls_arr.ndim == 1:
    perturbed_lls_arr = perturbed_lls_arr.reshape(-1, 1)
```

**Bug 3**: `ensemble.py:179` - 键名错误
```python
# ❌ 错误
perturbed_lls.append(result.get("perturbed_sampled_ll", ...))

# ✅ 修复
pert_ll = result.get("perturbed_original_ll", ...)
perturbed_lls.append(pert_ll)
```

---

## 预期性能提升

### DetectGPT核心算法
- **ROC AUC**: 0.57 → **0.75-0.82**
- **PR AUC**: 0.60 → **0.78-0.85**

### 集成分类器
- **ROC AUC**: 预期 **0.85-0.92**
- **PR AUC**: 预期 **0.88-0.95**

### RoBERTa基线
- **ROC AUC**: 预期 **0.70-0.78**
- **PR AUC**: 预期 **0.72-0.80**

---

## 运行命令

### 快速测试（验证修复）
```bash
conda activate jittor-cpu-wsl
cd /mnt/d/HuaweiMoveData/Users/asdf1/Desktop/jittor-text-detect

python run.py --DEVICE cpu --max_raw_data 50 --debug
```

### 完整实验（DetectGPT + 集成分类器）
```bash
python run.py --DEVICE cpu --max_raw_data 200 --ensemble --n_perturbation_rounds 10
```

### 对比实验（RoBERTa + DetectGPT）
```bash
python run.py --DEVICE cpu --max_raw_data 100 --roberta --ensemble
```

### 稳定性测试
```bash
bash run_stability_test.sh
```

---

## 技术要点总结

### 1. 为什么曲率分数有效？
- **人类文本**：扰动后似然下降**更多** → 曲率**更大**
- **AI文本**：扰动后似然下降**较少** → 曲率**更小**
- 原因：AI生成文本对局部扰动更鲁棒（来自低熵分布）

### 2. 为什么归一化重要？
- 消除文本长度影响
- 使不同文本分数可比
- 提升模型泛化能力

### 3. 为什么增加特征有效？
- 熵特征：捕获文本复杂度
- 比率特征：捕获相对变化
- 多维度：减少信息损失

### 4. 为什么增大扰动轮数有效？
- 减少随机性
- 更稳定的曲率估计
- 提升AUC一致性

---

## 下一步优化方向

1. **使用更大的模型**：GPT-2 Large / Bloomz-560m
2. **调整生成温度**：尝试0.5-0.9范围
3. **增加样本量**：500 → 1000+
4. **交叉验证**：5-fold CV验证稳定性
5. **特征选择**：分析特征重要性

---

## 修改文件清单

| 文件 | 修改内容 | 行数 |
|------|----------|------|
| `utils/baselines/detectGPT.py` | 真实计算扰动分数 | +50 |
| `utils/baselines/model.py` | 归一化曲率分数 | +5 |
| `utils/baselines/ensemble.py` | 8维特征 + 模型优化 | +40 |
| `run.py` | 优化扰动参数 | - |

**总计**：约95行新增/修改代码

---

## 验证检查清单

- [x] 修复扰动分数造假问题
- [x] 添加归一化处理
- [x] 增强特征工程（6→8维）
- [x] 提升模型容量（100→200树）
- [x] 优化扰动参数
- [x] 修复集成分类器Bug
- [x] 通过语法检查
- [ ] 实际运行验证AUC提升
- [ ] 稳定性测试（多次运行）
