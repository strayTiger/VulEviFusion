# VulPathFusion

VulPathFusion is a C/C++ vulnerability detection workflow that augments CodeBERT with screened vulnerability-path evidence and security descriptions, then performs validation-selected probability-level fusion.

This GitHub package keeps only the main reproducible workflow:

- `src/`: core Python programs.
- `data/lin_et_al/`: main dataset used in the paper.
- `data/devign/` and `data/reveal/`: cross-dataset evaluation data.
- `docs/workflow_zh.md`: Chinese workflow notes.

It intentionally excludes local automation scripts, trained checkpoints, prediction files, `outputs`, and other runtime results.

## Environment

Install dependencies:

```bash
pip install -r requirements.txt
```

If you use the LLM description generation step, set the SiliconFlow API key before running step 2:

```powershell
$env:SILICONFLOW_API_KEY="your_key_here"
```

or on Linux:

```bash
export SILICONFLOW_API_KEY="your_key_here"
```

## Dataset Layout

Each dataset directory contains `train.jsonl`, `valid.jsonl`, and `test.jsonl`.

```text
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

Rows use `func` as source-code field and `target` as binary label.

## Main Workflow

The example below uses `data/lin_et_al`. To run Devign or ReVeal, replace `lin_et_al` with `devign` or `reveal` in the input and output paths.

### 1. Generate DFG and Vulnerability Paths

```bash
python src/generate_dfg_paths.py \
  --input_dir data/lin_et_al \
  --output_dir outputs/lin_et_al \
  --code_key func \
  --splits train,valid,test \
  --max_hops 6 \
  --max_paths_per_sample 5
```

Main outputs:

- `outputs/lin_et_al/full_dfg/*.full_dfg.jsonl`: full data-flow evidence.
- `outputs/lin_et_al/vul_paths/*.vul_paths.jsonl`: screened vulnerability-oriented paths.

### 2. Generate Security Descriptions

```bash
python src/generate_security_description.py \
  --input_dir outputs/lin_et_al/vul_paths \
  --output_dir outputs/lin_et_al/descriptions_llm \
  --code_key func \
  --splits train,valid,test
```

Main outputs:

- `outputs/lin_et_al/descriptions_llm/train.desc.jsonl`
- `outputs/lin_et_al/descriptions_llm/valid.desc.jsonl`
- `outputs/lin_et_al/descriptions_llm/test.desc.jsonl`

### 3. Build Screened Model Inputs

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

This generates seven input views:

- `code_only`
- `code_vul_path_top1_screened`
- `code_vul_path_top3_screened`
- `code_desc_screened`
- `code_vul_path_desc_top3_screened`
- `path_desc_code_top1_screened`
- `path_desc_code_top3_screened`

Each view contains `train.jsonl`, `valid.jsonl`, and `test.jsonl`.

### 4. Train CodeBERT Classifiers

Train one model per input view. Example for `code_vul_path_top3_screened`:

```bash
python src/train_codebert.py \
  --train_file outputs/lin_et_al/experiment_inputs_screened/code_vul_path_top3_screened/train.jsonl \
  --valid_file outputs/lin_et_al/experiment_inputs_screened/code_vul_path_top3_screened/valid.jsonl \
  --test_file outputs/lin_et_al/experiment_inputs_screened/code_vul_path_top3_screened/test.jsonl \
  --model_name microsoft/codebert-base \
  --output_dir outputs/lin_et_al/checkpoints_screened_thr/code_vul_path_top3_screened \
  --max_length 512 \
  --batch_size 16 \
  --eval_batch_size 32 \
  --learning_rate 2e-5 \
  --epochs 20 \
  --class_weight balanced \
  --metric_for_best f1 \
  --threshold_metric f1 \
  --save_predictions
```

Repeat this command for all views you want to include in fusion, replacing the view name in both the input path and output path.

Main outputs for each view:

- `best_model/`: validation-selected model.
- `valid_predictions.jsonl` and `test_predictions.jsonl`: probability predictions.
- `test_metrics.json`: final test metrics.
- `threshold_tuning.json`: validation threshold-search record.

### 5. Validation-Selected Fusion

```bash
python src/valid_tuned_ensemble.py \
  --checkpoint_root outputs/lin_et_al/checkpoints_screened_thr \
  --methods code_only,code_vul_path_top1_screened,code_vul_path_top3_screened,code_desc_screened,code_vul_path_desc_top3_screened,path_desc_code_top1_screened,path_desc_code_top3_screened \
  --output_dir outputs/lin_et_al/ensemble_screened/auto_search_k2_7 \
  --auto_combinations \
  --combo_min_size 2 \
  --combo_max_size 7 \
  --weight_step 0.1 \
  --max_weight_configs 20000 \
  --selection_metric f1
```

This script searches model subsets, fusion weights, and thresholds on the validation set, then reports the selected result on the test set.

Main outputs:

- `best_ensemble_metrics.json`: official validation-selected fusion result.
- `combination_summary.csv` / `.md`: ranked candidate combinations.
- `valid_predictions.jsonl` and `test_predictions.jsonl`: fused probabilities.

### 6. Subset Analysis

```bash
python src/analyze_prediction_subsets.py \
  --predictions outputs/lin_et_al/ensemble_screened/auto_search_k2_7/test_predictions.jsonl \
  --features outputs/lin_et_al/experiment_inputs_screened/code_vul_path_top3_screened/test.jsonl \
  --output_dir outputs/lin_et_al/subset_screened/auto_search_k2_7
```

This compares performance on samples with extracted vulnerability paths and samples without extracted paths.

### 7. Path Perturbation Analysis

```bash
python src/path_perturbation.py \
  --checkpoint_dir outputs/lin_et_al/checkpoints_screened_thr/code_vul_path_top3_screened \
  --input_file outputs/lin_et_al/experiment_inputs_screened/code_vul_path_top3_screened/test.jsonl \
  --output_dir outputs/lin_et_al/perturbation_screened_thr/code_vul_path_top3 \
  --batch_size 32 \
  --max_length 512 \
  --save_predictions
```

This removes or masks different vulnerability-path fields and measures how the model probability changes, which is used as an explanation-supporting perturbation experiment.

## Notes

- Do not commit `.env`, trained checkpoints, model weights, or `outputs`.
- `best_by_test_diagnostic` in fusion outputs is diagnostic only. Use `test_metrics` under the validation-selected result for paper reporting.
- The workflow is designed for GPU training. A 24GB to 32GB GPU is recommended for the default CodeBERT settings.
