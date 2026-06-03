# VulPathFusion 主流程说明

本文档说明 GitHub 版本中每个核心代码文件的作用、输入输出以及推荐执行顺序。该版本只保留主流程代码和数据集，不包含本地自动化脚本、训练结果、模型检查点和论文中间文件。

## 目录结构

```text
VulPathFusion_github/
  README.md
  requirements.txt
  .env.example
  src/
    generate_dfg_paths.py
    generate_security_description.py
    train_codebert.py
    valid_tuned_ensemble.py
    build_screened_path_inputs.py
    analyze_prediction_subsets.py
    path_perturbation.py
  data/
    lin_et_al/
    devign/
    reveal/
```

## 总体流程

```text
data/{dataset}
  -> generate_dfg_paths.py
  -> outputs/{dataset}/full_dfg
  -> outputs/{dataset}/vul_paths
  -> generate_security_description.py
  -> outputs/{dataset}/descriptions_llm
  -> build_screened_path_inputs.py
  -> outputs/{dataset}/experiment_inputs_screened
  -> train_codebert.py
  -> outputs/{dataset}/checkpoints_screened_thr
  -> valid_tuned_ensemble.py
  -> outputs/{dataset}/ensemble_screened
  -> analyze_prediction_subsets.py
  -> outputs/{dataset}/subset_screened
  -> path_perturbation.py
  -> outputs/{dataset}/perturbation_screened_thr
```

`{dataset}` 可以取 `lin_et_al`、`devign` 或 `reveal`。

## 1. 数据输入

每个数据集目录下包含：

```text
train.jsonl
valid.jsonl
test.jsonl
```

每行是一个 JSON 样本，主要字段如下：

| 字段 | 含义 |
| --- | --- |
| `func` | C/C++ 函数代码 |
| `target` | 漏洞标签，`1` 表示漏洞，`0` 表示非漏洞 |
| `idx` | 样本编号 |
| `func_name` | 函数名或文件名 |
| `project` | 所属项目 |

## 2. 提取 DFG 和漏洞路径

代码文件：

```text
src/generate_dfg_paths.py
```

作用：

1. 读取原始函数代码。
2. 使用 tree-sitter 解析 C/C++ 代码。
3. 构造数据流证据。
4. 从潜在 sink、边界检查、指针访问、算术操作等位置筛选漏洞相关路径。

输出：

| 输出目录 | 作用 |
| --- | --- |
| `outputs/{dataset}/full_dfg` | 保存完整 DFG 证据 |
| `outputs/{dataset}/vul_paths` | 保存筛选后的漏洞路径证据 |

## 3. 生成安全描述

代码文件：

```text
src/generate_security_description.py
```

作用：

1. 读取 `vul_paths` 中的路径证据。
2. 对成功提取到路径的样本调用 LLM。
3. 生成保守的安全语义描述，避免直接把候选证据写成确定漏洞结论。

环境变量：

```text
SILICONFLOW_API_KEY
```

输出：

```text
outputs/{dataset}/descriptions_llm/*.desc.jsonl
```

## 4. 构造多视角输入

代码文件：

```text
src/build_screened_path_inputs.py
```

作用：

把原始代码、Top-1/Top-3 漏洞路径、安全描述组合成多种 CodeBERT 输入视角。

生成的视角：

| 视角 | 输入内容 |
| --- | --- |
| `code_only` | 仅源代码 |
| `code_vul_path_top1_screened` | 源代码 + Top-1 漏洞路径 |
| `code_vul_path_top3_screened` | 源代码 + Top-3 漏洞路径 |
| `code_desc_screened` | 源代码 + 安全描述 |
| `code_vul_path_desc_top3_screened` | 源代码 + Top-3 漏洞路径 + 安全描述 |
| `path_desc_code_top1_screened` | Top-1 漏洞路径 + 安全描述 + 源代码 |
| `path_desc_code_top3_screened` | Top-3 漏洞路径 + 安全描述 + 源代码 |

输出：

```text
outputs/{dataset}/experiment_inputs_screened/{view}/train.jsonl
outputs/{dataset}/experiment_inputs_screened/{view}/valid.jsonl
outputs/{dataset}/experiment_inputs_screened/{view}/test.jsonl
```

## 5. 训练 CodeBERT 分类器

代码文件：

```text
src/train_codebert.py
```

作用：

1. 对每个输入视角分别训练一个 CodeBERT 二分类模型。
2. 在验证集上选择最佳模型。
3. 在验证集上搜索最佳分类阈值。
4. 输出验证集和测试集预测概率。

输出：

| 文件或目录 | 作用 |
| --- | --- |
| `best_model/` | 验证集选择的最佳模型 |
| `valid_predictions.jsonl` | 验证集预测概率 |
| `test_predictions.jsonl` | 测试集预测概率 |
| `test_metrics.json` | 测试集指标 |
| `threshold_tuning.json` | 阈值搜索记录 |

## 6. 融合模型

代码文件：

```text
src/valid_tuned_ensemble.py
```

作用：

1. 读取多个输入视角模型的 `valid_predictions.jsonl` 和 `test_predictions.jsonl`。
2. 在验证集上搜索模型组合、融合权重和分类阈值。
3. 用验证集选出的组合在测试集上报告最终结果。

融合方式是概率级加权平均，例如：

```text
final_prob =
  0.2 * path_top1_prob
+ 0.1 * path_top3_prob
+ 0.2 * desc_prob
+ 0.2 * path_desc_code_top1_prob
+ 0.3 * path_desc_code_top3_prob
```

这里融合的是每个模型对“该样本为漏洞”的预测概率 `prob_vulnerable`，不是直接融合 F1。

输出：

| 文件 | 作用 |
| --- | --- |
| `best_ensemble_metrics.json` | 验证集选择的正式融合结果 |
| `combination_summary.csv` | 所有候选组合的排名 |
| `test_predictions.jsonl` | 融合后的测试集预测概率 |

## 7. has_path / no_path 子集分析

代码文件：

```text
src/analyze_prediction_subsets.py
```

作用：

把测试样本分成两组：

1. `has_path`：成功提取到漏洞路径证据的样本。
2. `no_path`：没有提取到漏洞路径证据的样本。

然后分别计算 Precision、Recall、F1、AUC、AP，用来分析路径证据是否对模型更有帮助。

输出：

```text
outputs/{dataset}/subset_screened/subset_metrics.json
```

## 8. 路径扰动解释分析

代码文件：

```text
src/path_perturbation.py
```

作用：

对输入中的漏洞路径证据做遮挡或删除，例如删除 `risk`、`sink`、`checks`、`arithmetic`、`dereference` 等字段，再观察模型预测概率是否下降。

如果删除路径证据后漏洞概率明显下降，说明模型确实利用了这些路径证据，而不是只依赖原始代码文本。

输出：

```text
outputs/{dataset}/perturbation_screened_thr/{method}/perturbation_metrics.json
```

## 9. 上传 GitHub 时保留和忽略

应该保留：

| 内容 | 原因 |
| --- | --- |
| `src/` | 主流程代码 |
| `data/` | 可复现实验数据 |
| `README.md` | 快速开始说明 |
| `requirements.txt` | 依赖环境 |
| `.env.example` | API key 示例，不包含真实 key |

不应该上传：

| 内容 | 原因 |
| --- | --- |
| `outputs/` | 运行结果，体积大且可重新生成 |
| `checkpoints/` | 模型权重，体积大 |
| `scripts/` | 本地自动化脚本，不属于最小主流程 |
| `.env` | 包含私有 API key |
