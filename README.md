# VulEviFusion: Validation-Selected Fusion of Path and Semantic Evidence

VulEviFusion is a C/C++ function-level vulnerability detection workflow that
constructs vulnerability-oriented path evidence, rewrites that evidence into
conservative security descriptions, and trains CodeT5 classifiers on formal
evidence views.

The key idea is to avoid treating raw source code, static path evidence, and
LLM-generated descriptions as one fixed input. VulEviFusion first constructs
screened Top-3 vulnerability paths, uses the LLM only as a constrained evidence
rewriter, and then trains CodeT5 classifiers on several formal views. In the full
paper, view subsets, weights, and thresholds are selected only on the validation
set before final test reporting.

This GitHub release is a compact reproducibility package. It contains the data
preparation, five-view input construction, CodeT5 single-view training, and
validation-selected VulEviFusion fusion script. It intentionally excludes
trained checkpoints, prediction files, local automation scripts, and `outputs`.

## Overview

VulEviFusion addresses three practical issues in neural vulnerability detection:

1. **Source tokens alone hide security-relevant relations.**  
   Raw C/C++ code often does not explicitly expose how variables flow into
   security-sensitive operations, checks, pointer dereferences, or arithmetic
   expressions.

2. **Full program graphs can be noisy under a 512-token model budget.**  
   VulEviFusion therefore extracts compact, vulnerability-oriented evidence
   paths instead of appending all available graph context.

3. **LLM descriptions should not become label leakage.**  
   The LLM sees only screened path evidence and source context. It is prompted
   to summarize observable security evidence and does not output the final
   vulnerability label.

The released workflow has four executable stages:

1. **Evidence-path extraction**  
   Build approximate intra-procedural dependencies and screened vulnerability
   paths from C/C++ functions.

2. **Constrained security-description generation**  
   Generate conservative descriptions only for samples with real path evidence.

3. **Five formal CodeT5 input views**  
   Construct the five views used after RQ1 selects CodeT5 + Top-3 paths.

4. **CodeT5 training, validation-tuned fusion, and analysis**  
   Train each view independently, then search view subsets, weights, and
   thresholds on the validation set before reporting fixed test results.

## Design of VulEviFusion

![VulEviFusion framework](./vulevifusion_framework_01.png)

The figure shows the full paper-level design. This repository releases the
evidence construction, description generation, five-view input construction,
single-view CodeT5 training utilities, and the validation-selected fusion script
used to produce VulEviFusion's final result.

## Main Features

- Function-level vulnerability detection for C/C++ code.
- Tree-sitter based lightweight C/C++ parsing.
- Approximate intra-procedural dependency and event extraction.
- Screened Top-3 vulnerability-oriented path evidence.
- Conservative Qwen3-Coder security-description generation.
- Five formal CodeT5 input views:
  - `code_only`
  - `code_vul_path_top3_screened`
  - `code_desc_screened`
  - `code_vul_path_desc_top3_screened`
  - `path_desc_code_top3_screened`
- Validation-selected checkpoint and threshold for each single-view classifier.
- Validation-selected fusion over view subsets, weights, and thresholds.
- Subset analysis for samples with and without extracted path evidence.
- Path-field perturbation analysis for explanation support.
- Public classification metrics: Accuracy, Precision, Recall, and F1.

## Repository Structure

```text
VulEviFusion_github/
  README.md
  requirements.txt
  .env.example
  VulEviFusion_framework.png
  docs/
    workflow_zh.md
  src/
    generate_dfg_paths.py
    generate_security_description.py
    build_screened_path_inputs.py
    train_codet5.py
    run_VulEviFusion_fusion.py
    analyze_prediction_subsets.py
    path_perturbation.py
  data/
    lin_et_al/
      train.jsonl
      valid.jsonl
      test.jsonl
    devign/
      train.jsonl
      valid.jsonl
      test.jsonl
      dataset_summary.json
    reveal/
      train.jsonl
      valid.jsonl
      test.jsonl
      dataset_summary.json
```

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

The default training script uses Hugging Face `Salesforce/codet5-base`. A
CUDA-enabled GPU is recommended for training. For the default batch sizes, a
24GB to 32GB GPU is recommended.

If you run the LLM description generation step, set the SiliconFlow API key:

```powershell
$env:SILICONFLOW_API_KEY="your_key_here"
```

or on Linux:

```bash
export SILICONFLOW_API_KEY="your_key_here"
```

## Dataset Format

Each dataset split is stored as a JSONL file. Each line represents one
function-level sample.

```json
{"idx": 1, "project": "FFmpeg", "func_name": "example.c_foo", "func": "int foo(...) { ... }", "target": 0}
```

Required fields:

| Field | Description |
|---|---|
| `func` | Source code of the C/C++ function. |
| `target` | Binary label, where `1` means vulnerable and `0` means non-vulnerable. |

Optional but recommended fields:

| Field | Description |
|---|---|
| `idx` | Sample ID used for alignment across intermediate files. |
| `project` | Project name. |
| `func_name` | Function or file-level identifier. |

## Running the Pipeline

The examples below use `data/lin_et_al`. To run Devign or ReVeal, replace
`lin_et_al` with `devign` or `reveal` in the input and output paths.

### Step 1: Generate DFG and Vulnerability Paths

```bash
python src/generate_dfg_paths.py \
  --input_dir data/lin_et_al \
  --output_dir outputs/lin_et_al \
  --code_key func \
  --splits train,valid,test \
  --max_hops 6 \
  --max_paths_per_sample 5
```

Expected outputs:

```text
outputs/lin_et_al/full_dfg/train.full_dfg.jsonl
outputs/lin_et_al/full_dfg/valid.full_dfg.jsonl
outputs/lin_et_al/full_dfg/test.full_dfg.jsonl
outputs/lin_et_al/vul_paths/train.vul_paths.jsonl
outputs/lin_et_al/vul_paths/valid.vul_paths.jsonl
outputs/lin_et_al/vul_paths/test.vul_paths.jsonl
```

### Step 2: Generate Conservative Security Descriptions

```bash
python src/generate_security_description.py \
  --input_dir outputs/lin_et_al/vul_paths \
  --output_dir outputs/lin_et_al/descriptions_llm \
  --code_key func \
  --splits train,valid,test
```

Expected outputs:

```text
outputs/lin_et_al/descriptions_llm/train.desc.jsonl
outputs/lin_et_al/descriptions_llm/valid.desc.jsonl
outputs/lin_et_al/descriptions_llm/test.desc.jsonl
```

The LLM is used as a constrained evidence rewriter. It receives no class label
and does not directly produce the final vulnerability decision.

### Step 3: Build the Five Formal Input Views

```bash
python src/build_screened_path_inputs.py \
  --data_dir data/lin_et_al \
  --vul_path_dir outputs/lin_et_al/vul_paths \
  --desc_dir outputs/lin_et_al/descriptions_llm \
  --output_dir outputs/lin_et_al/experiment_inputs_screened \
  --code_key func \
  --target_key target \
  --splits train,valid,test \
  --top_k 3 \
  --max_code_chars 2400 \
  --max_path_chars 900 \
  --max_desc_chars 450
```

Expected output layout:

```text
outputs/lin_et_al/experiment_inputs_screened/
  code_only/
  code_vul_path_top3_screened/
  code_desc_screened/
  code_vul_path_desc_top3_screened/
  path_desc_code_top3_screened/
```

Each view directory contains `train.jsonl`, `valid.jsonl`, and `test.jsonl`.

### Step 4: Train a CodeT5 Classifier

Train one model per input view. Example for `code_vul_path_top3_screened`:

```bash
python src/train_codet5.py \
  --train_file outputs/lin_et_al/experiment_inputs_screened/code_vul_path_top3_screened/train.jsonl \
  --valid_file outputs/lin_et_al/experiment_inputs_screened/code_vul_path_top3_screened/valid.jsonl \
  --test_file outputs/lin_et_al/experiment_inputs_screened/code_vul_path_top3_screened/test.jsonl \
  --model_name Salesforce/codet5-base \
  --output_dir outputs/lin_et_al/checkpoints_screened_codet5/code_vul_path_top3_screened \
  --max_length 512 \
  --batch_size 16 \
  --eval_batch_size 32 \
  --learning_rate 2e-5 \
  --epochs 20 \
  --class_weight balanced \
  --metric_for_best f1 \
  --threshold_metric f1 \
  --fp16 \
  --save_predictions
```

Expected outputs:

```text
outputs/lin_et_al/checkpoints_screened_codet5/code_vul_path_top3_screened/best_model/
outputs/lin_et_al/checkpoints_screened_codet5/code_vul_path_top3_screened/valid_predictions.jsonl
outputs/lin_et_al/checkpoints_screened_codet5/code_vul_path_top3_screened/test_predictions.jsonl
outputs/lin_et_al/checkpoints_screened_codet5/code_vul_path_top3_screened/test_metrics.json
outputs/lin_et_al/checkpoints_screened_codet5/code_vul_path_top3_screened/threshold_tuning.json
```

The best checkpoint and threshold are selected on the validation set. The test
set is used only for final reporting.

### Step 5: Run Validation-Tuned VulEviFusion Fusion

After training all five formal views with `--save_predictions`, run the final
VulEviFusion fusion search. The script searches view subsets, fusion weights,
and the classification threshold on the validation set, then reports the fixed
configuration on the test set:

```bash
python src/run_VulEviFusion_fusion.py \
  --checkpoint_root outputs/lin_et_al/checkpoints_screened_codet5 \
  --methods code_only,code_vul_path_top3_screened,code_desc_screened,code_vul_path_desc_top3_screened,path_desc_code_top3_screened \
  --output_dir outputs/lin_et_al/VulEviFusion_fusion \
  --auto_combinations \
  --combo_min_size 2 \
  --combo_max_size 5 \
  --weight_step 0.05 \
  --max_weight_configs 60000 \
  --selection_metric f1
```

Expected outputs:

```text
outputs/lin_et_al/VulEviFusion_fusion/best_ensemble_metrics.json
outputs/lin_et_al/VulEviFusion_fusion/ensemble_metrics.json
outputs/lin_et_al/VulEviFusion_fusion/combination_summary.csv
outputs/lin_et_al/VulEviFusion_fusion/combination_summary.json
outputs/lin_et_al/VulEviFusion_fusion/combination_summary.md
outputs/lin_et_al/VulEviFusion_fusion/valid_predictions.jsonl
outputs/lin_et_al/VulEviFusion_fusion/test_predictions.jsonl
```

The official VulEviFusion result should be taken from the validation-selected
entry in `best_ensemble_metrics.json`. The test split is evaluated only after
the view subset, weights, and threshold have been fixed by validation data.

### Step 6: Analyze Path-Availability Subsets

```bash
python src/analyze_prediction_subsets.py \
  --predictions outputs/lin_et_al/checkpoints_screened_codet5/code_vul_path_top3_screened/test_predictions.jsonl \
  --features outputs/lin_et_al/experiment_inputs_screened/code_vul_path_top3_screened/test.jsonl \
  --output_dir outputs/lin_et_al/subset_screened/code_vul_path_top3
```

Expected output:

```text
outputs/lin_et_al/subset_screened/code_vul_path_top3/subset_metrics.json
```

This step compares samples with extracted vulnerability paths against samples
without extracted paths.

### Step 7: Run Path-Field Perturbation

```bash
python src/path_perturbation.py \
  --checkpoint_dir outputs/lin_et_al/checkpoints_screened_codet5/code_vul_path_top3_screened \
  --input_file outputs/lin_et_al/experiment_inputs_screened/code_vul_path_top3_screened/test.jsonl \
  --output_dir outputs/lin_et_al/perturbation_screened_codet5/code_vul_path_top3 \
  --batch_size 32 \
  --max_length 512 \
  --save_predictions
```

Expected outputs:

```text
outputs/lin_et_al/perturbation_screened_codet5/code_vul_path_top3/perturbation_summary.json
outputs/lin_et_al/perturbation_screened_codet5/code_vul_path_top3/perturbation_summary.md
```

This analysis removes or masks fields such as `risk`, `sink`, `checks`,
`arithmetic`, and `dereference`, then measures probability changes. It provides
behavioral support for evidence use, not proof of causal program reasoning.

## Evaluation Metrics

For manuscript reporting, use only the public classification metrics used by the
current VulEviFusion protocol:

- Accuracy
- Precision
- Recall
- F1

Checkpoints, view subsets, fusion weights, and thresholds are selected on the
validation set. Do not use test results to choose models, thresholds, views,
weights, or hyperparameters.

## Key Design Choices

### Path Evidence Before LLM Text

The LLM is not asked to inspect arbitrary code and produce a vulnerability
verdict. It receives screened evidence paths and summarizes observable risk
signals under a conservative prompt.

### Five Formal Views

After CodeT5 + Top-3 is selected, this release constructs only the formal input
views used by the current VulEviFusion protocol. Exploratory input variants are
not generated by default.

### No Test-Set Selection

Training selects the checkpoint and classification threshold on validation data.
The test split is evaluated after selection is frozen.

### Compact GitHub Release

This repository includes the validation-selected fusion script needed to produce
the final VulEviFusion result, while excluding trained checkpoints, generated
predictions, local automation scripts, and bulky `outputs` directories.

## Reproducibility Tips

- Keep `idx`, `project`, and `func_name` stable across raw data, path files,
  description files, and constructed input views.
- Use the same `train/valid/test` split throughout one run.
- Keep generated `screened_input_summary.json`, `test_metrics.json`,
  `threshold_tuning.json`, and `best_ensemble_metrics.json` files for later
  auditing.
- Do not mix outputs from different datasets or different `top_k` settings.
- Do not commit `.env`, checkpoints, model weights, or `outputs`.

## Suggested `.gitignore`

```gitignore
.env
__pycache__/
*.py[cod]
*.log
outputs/
outputs_*/
checkpoints/
checkpoints_*/
*.bin
*.pt
*.pth
*.safetensors
.idea/
.vscode/
.DS_Store
Thumbs.db
```

## Acknowledgement

This project uses PyTorch, Hugging Face Transformers, scikit-learn, tree-sitter,
and common Python data-processing utilities.
