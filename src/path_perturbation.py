#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
path_perturbation.py

Evaluate whether screened vulnerability-path evidence affects model decisions.
The script predicts the original input, then creates controlled perturbations
inside <VUL_PATH_SCREENED> and measures probability changes.

Example:

python src/path_perturbation.py ^
  --checkpoint_dir outputs/lin_et_al/checkpoints_screened_thr/code_vul_path_top3_screened ^
  --input_file outputs/lin_et_al/experiment_inputs_screened/code_vul_path_top3_screened/test.jsonl ^
  --output_dir outputs/lin_et_al/perturbation_screened_thr/code_vul_path_top3 ^
  --batch_size 32 ^
  --max_length 512 ^
  --save_predictions
"""

import argparse
import json
import os
import re
import statistics
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = lambda x, **kwargs: x


PATH_BLOCK_RE = re.compile(r"\s*<VUL_PATH_SCREENED>.*?</VUL_PATH_SCREENED>\s*", re.DOTALL)
PATH_INNER_RE = re.compile(r"(<VUL_PATH_SCREENED>)(.*?)(</VUL_PATH_SCREENED>)", re.DOTALL)
FIELD_NAMES = [
    "type",
    "sink_expr",
    "sink_vars",
    "sink",
    "risk",
    "checks",
    "arithmetic",
    "dereference",
]
FIELD_RE = re.compile(r"(?<![A-Za-z0-9_])(" + "|".join(FIELD_NAMES) + r")\s*:", re.DOTALL)

PERTURBATION_FIELDS = {
    "remove_sink": {"sink", "sink_expr", "sink_vars"},
    "remove_risk": {"risk"},
    "remove_checks": {"checks"},
    "remove_arithmetic": {"arithmetic"},
    "remove_dereference": {"dereference"},
}
DEFAULT_VARIANTS = [
    "remove_path",
    "remove_sink",
    "remove_risk",
    "remove_checks",
    "remove_arithmetic",
    "remove_dereference",
]


def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def write_json(path: str, obj: Any) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def split_csv(text: str) -> List[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def clean_text(text: Any) -> str:
    if text is None:
        return ""
    return str(text)


def normalize_spaces(text: str) -> str:
    return " ".join(text.split())


def remove_path_block(text: str) -> str:
    return PATH_BLOCK_RE.sub(" ", text).strip()


def has_path_block(text: str) -> bool:
    return bool(PATH_INNER_RE.search(text))


def strip_field_value(value: str) -> str:
    return value.strip(" ;|\n\r\t")


def remove_path_fields_from_inner(inner: str, fields_to_remove: Sequence[str]) -> str:
    fields = {field.lower() for field in fields_to_remove}
    matches = list(FIELD_RE.finditer(inner))
    if not matches:
        return inner

    kept_parts: List[str] = []
    for i, match in enumerate(matches):
        label = match.group(1).lower()
        value_start = match.end()
        value_end = matches[i + 1].start() if i + 1 < len(matches) else len(inner)
        if label in fields:
            continue
        value = strip_field_value(inner[value_start:value_end])
        if value:
            kept_parts.append(f"{label}: {value}")
        else:
            kept_parts.append(f"{label}:")

    return " ; ".join(kept_parts)


def remove_path_fields(text: str, fields_to_remove: Sequence[str]) -> str:
    def repl(match: re.Match) -> str:
        prefix, inner, suffix = match.groups()
        perturbed_inner = remove_path_fields_from_inner(inner, fields_to_remove)
        if not perturbed_inner.strip():
            return ""
        return f"{prefix} {perturbed_inner.strip()} {suffix}"

    return PATH_INNER_RE.sub(repl, text)


def perturb_text(text: str, perturbation: str) -> str:
    if perturbation == "original":
        return text
    if perturbation == "remove_path":
        return remove_path_block(text)
    if perturbation in PERTURBATION_FIELDS:
        return remove_path_fields(text, sorted(PERTURBATION_FIELDS[perturbation]))
    raise ValueError(f"Unknown perturbation: {perturbation}")


def row_label(row: Dict[str, Any], label_key: str = "target") -> int:
    value = row.get(label_key, row.get("label", 0))
    return int(value)


def row_has_real_path(row: Dict[str, Any], text_key: str = "input_text") -> bool:
    components = row.get("components")
    if isinstance(components, dict) and "vul_path_is_real" in components:
        return bool(components.get("vul_path_is_real"))
    return has_path_block(clean_text(row.get(text_key, "")))


def row_meta(row: Dict[str, Any], index: int) -> Dict[str, Any]:
    meta = {
        "row_index": index,
        "idx": row.get("idx", index),
        "project": row.get("project", ""),
        "func_name": row.get("func_name", ""),
        "experiment_name": row.get("experiment_name", ""),
    }
    components = row.get("components")
    if isinstance(components, dict):
        meta["screened_path_count"] = components.get("screened_path_count")
        meta["path_first"] = components.get("path_first")
    return meta


def mean(values: Sequence[float]) -> Optional[float]:
    return float(sum(values) / len(values)) if values else None


def median(values: Sequence[float]) -> Optional[float]:
    return float(statistics.median(values)) if values else None


def binary_metrics(labels: Sequence[int], probs: Sequence[float], threshold: float) -> Dict[str, Any]:
    if not labels:
        return {"n": 0}

    tp = fp = tn = fn = 0
    for label, prob in zip(labels, probs):
        pred = int(prob >= threshold)
        if label == 1 and pred == 1:
            tp += 1
        elif label == 0 and pred == 1:
            fp += 1
        elif label == 0 and pred == 0:
            tn += 1
        elif label == 1 and pred == 0:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    acc = (tp + tn) / len(labels)
    return {
        "n": len(labels),
        "acc": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "threshold": float(threshold),
    }


def summarize_group(rows: List[Dict[str, Any]], threshold: float) -> Dict[str, Any]:
    if not rows:
        return {"n": 0}

    original_probs = [float(row["original_prob"]) for row in rows]
    perturbed_probs = [float(row["perturbed_prob"]) for row in rows]
    labels = [int(row["label"]) for row in rows]
    deltas = [float(row["original_prob"]) - float(row["perturbed_prob"]) for row in rows]
    positive_delta_count = sum(1 for delta in deltas if delta > 0)

    return {
        "n": len(rows),
        "positives": int(sum(labels)),
        "negatives": int(len(labels) - sum(labels)),
        "mean_original_prob": mean(original_probs),
        "mean_perturbed_prob": mean(perturbed_probs),
        "mean_delta": mean(deltas),
        "median_delta": median(deltas),
        "mean_abs_delta": mean([abs(delta) for delta in deltas]),
        "positive_delta_rate": positive_delta_count / len(rows),
        "changed_rows": int(sum(1 for row in rows if row.get("changed"))),
        "flipped_positive_to_negative": int(
            sum(
                1
                for row in rows
                if int(row["label"]) == 1
                and float(row["original_prob"]) >= threshold
                and float(row["perturbed_prob"]) < threshold
            )
        ),
        "flipped_negative_to_positive": int(
            sum(
                1
                for row in rows
                if int(row["label"]) == 0
                and float(row["original_prob"]) < threshold
                and float(row["perturbed_prob"]) >= threshold
            )
        ),
        "original_metrics": binary_metrics(labels, original_probs, threshold),
        "perturbed_metrics": binary_metrics(labels, perturbed_probs, threshold),
    }


def summarize_perturbation_rows(rows: List[Dict[str, Any]], threshold: float) -> Dict[str, Any]:
    variants = sorted({str(row["perturbation"]) for row in rows})
    summary: Dict[str, Any] = {}
    for variant in variants:
        subset = [row for row in rows if row["perturbation"] == variant]
        summary[variant] = {
            "all": summarize_group(subset, threshold),
            "changed_only": summarize_group([row for row in subset if row.get("changed")], threshold),
            "positives": summarize_group([row for row in subset if int(row["label"]) == 1], threshold),
            "negatives": summarize_group([row for row in subset if int(row["label"]) == 0], threshold),
            "has_path": summarize_group([row for row in subset if row.get("has_path")], threshold),
            "no_path": summarize_group([row for row in subset if not row.get("has_path")], threshold),
            "positive_has_path": summarize_group(
                [row for row in subset if int(row["label"]) == 1 and row.get("has_path")],
                threshold,
            ),
            "positive_no_path": summarize_group(
                [row for row in subset if int(row["label"]) == 1 and not row.get("has_path")],
                threshold,
            ),
        }
    return summary


def load_json_if_exists(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return json.load(f)


def load_threshold(checkpoint_dir: str, explicit_threshold: Optional[float]) -> float:
    if explicit_threshold is not None:
        return float(explicit_threshold)

    candidates = [
        os.path.join(checkpoint_dir, "test_metrics.json"),
        os.path.join(checkpoint_dir, "threshold_tuning.json"),
        os.path.join(checkpoint_dir, "best_threshold_table.json"),
        os.path.join(checkpoint_dir, "best_valid_metrics.json"),
    ]
    for path in candidates:
        obj = load_json_if_exists(path)
        if not obj:
            continue
        for key in ["final_threshold_selected_on_valid", "best_threshold", "best_threshold_during_training"]:
            if key in obj:
                return float(obj[key])
        for key in ["test_metrics_at_selected_threshold", "valid_metrics_at_selected_threshold"]:
            nested = obj.get(key)
            if isinstance(nested, dict) and "threshold" in nested:
                return float(nested["threshold"])
    return 0.5


def resolve_model_dir(checkpoint_dir: str) -> str:
    best_model = os.path.join(checkpoint_dir, "best_model")
    if os.path.isdir(best_model):
        return best_model
    return checkpoint_dir


def predict_texts(
    texts: List[str],
    checkpoint_dir: str,
    model_name: Optional[str],
    batch_size: int,
    max_length: int,
    device_name: str,
) -> List[float]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model_dir = resolve_model_dir(checkpoint_dir)
    tokenizer_source = model_name or model_dir
    model_source = model_dir
    device = torch.device(device_name if device_name else ("cuda" if torch.cuda.is_available() else "cpu"))

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    model = AutoModelForSequenceClassification.from_pretrained(model_source)
    model.to(device)
    model.eval()

    probs: List[float] = []
    for start in tqdm(range(0, len(texts), batch_size), desc="Predicting"):
        batch_texts = texts[start : start + batch_size]
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            logits = model(**encoded).logits
            batch_probs = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().tolist()
        probs.extend(float(prob) for prob in batch_probs)
    return probs


def write_summary_markdown(path: str, summary: Dict[str, Any], threshold: float) -> None:
    ensure_dir(os.path.dirname(path))
    lines = [
        "# Path Perturbation Summary",
        "",
        f"Threshold: `{threshold}`",
        "",
        "| Perturbation | Group | N | Mean delta | Positive drop rate | Flip pos->neg | Perturbed F1 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for perturbation, groups in summary.items():
        for group_name in ["all", "has_path", "positive_has_path", "changed_only"]:
            group = groups.get(group_name, {"n": 0})
            perturbed_metrics = group.get("perturbed_metrics", {}) if isinstance(group, dict) else {}
            lines.append(
                "| "
                + " | ".join(
                    [
                        perturbation,
                        group_name,
                        str(group.get("n", 0)),
                        format_optional(group.get("mean_delta")),
                        format_optional(group.get("positive_delta_rate")),
                        str(group.get("flipped_positive_to_negative", 0)),
                        format_optional(perturbed_metrics.get("f1")),
                    ]
                )
                + " |"
            )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def format_optional(value: Any) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.6f}"
    except Exception:
        return str(value)


def run(args: argparse.Namespace) -> None:
    rows = read_jsonl(args.input_file)
    if args.max_samples > 0:
        rows = rows[: args.max_samples]
    if not rows:
        raise ValueError(f"No rows loaded from {args.input_file}")

    variants = split_csv(args.variants)
    if not variants:
        raise ValueError("No perturbation variants selected.")
    for variant in variants:
        if variant != "remove_path" and variant not in PERTURBATION_FIELDS:
            raise ValueError(f"Unknown perturbation variant: {variant}")

    threshold = load_threshold(args.checkpoint_dir, args.threshold)
    original_texts = [clean_text(row.get(args.text_key, "")) for row in rows]
    original_probs = predict_texts(
        texts=original_texts,
        checkpoint_dir=args.checkpoint_dir,
        model_name=args.model_name,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device_name=args.device,
    )

    all_prediction_rows: List[Dict[str, Any]] = []
    for variant in variants:
        perturbed_texts = [perturb_text(text, variant) for text in original_texts]
        changed_indices = [i for i, (orig, pert) in enumerate(zip(original_texts, perturbed_texts)) if orig != pert]
        perturbed_probs = list(original_probs)

        if changed_indices:
            changed_texts = [perturbed_texts[i] for i in changed_indices]
            changed_probs = predict_texts(
                texts=changed_texts,
                checkpoint_dir=args.checkpoint_dir,
                model_name=args.model_name,
                batch_size=args.batch_size,
                max_length=args.max_length,
                device_name=args.device,
            )
            for index, prob in zip(changed_indices, changed_probs):
                perturbed_probs[index] = prob

        for i, (row, original_prob, perturbed_prob, original_text, perturbed_text) in enumerate(
            zip(rows, original_probs, perturbed_probs, original_texts, perturbed_texts)
        ):
            meta = row_meta(row, i)
            has_path = row_has_real_path(row, args.text_key)
            changed = original_text != perturbed_text
            label = row_label(row, args.label_key)
            result = {
                **meta,
                "perturbation": variant,
                "label": int(label),
                "has_path": bool(has_path),
                "changed": bool(changed),
                "original_prob": float(original_prob),
                "perturbed_prob": float(perturbed_prob),
                "delta": float(original_prob - perturbed_prob),
                "original_pred": int(original_prob >= threshold),
                "perturbed_pred": int(perturbed_prob >= threshold),
                "threshold": float(threshold),
            }
            if args.save_text_preview:
                result["original_text_preview"] = normalize_spaces(original_text)[:500]
                result["perturbed_text_preview"] = normalize_spaces(perturbed_text)[:500]
            all_prediction_rows.append(result)

    summary = summarize_perturbation_rows(all_prediction_rows, threshold)
    output = {
        "checkpoint_dir": args.checkpoint_dir,
        "input_file": args.input_file,
        "threshold": threshold,
        "variants": variants,
        "rows": len(rows),
        "summary": summary,
        "args": vars(args),
    }

    ensure_dir(args.output_dir)
    write_json(os.path.join(args.output_dir, "perturbation_summary.json"), output)
    write_summary_markdown(os.path.join(args.output_dir, "perturbation_summary.md"), summary, threshold)
    if args.save_predictions:
        write_jsonl(os.path.join(args.output_dir, "perturbation_predictions.jsonl"), all_prediction_rows)

    print(json.dumps(output, ensure_ascii=False, indent=2))
    print(f"[Done] perturbation eval -> {args.output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name", default=None, help="Optional tokenizer source; defaults to checkpoint best_model.")
    parser.add_argument("--text_key", default="input_text")
    parser.add_argument("--label_key", default="target")
    parser.add_argument("--variants", default=",".join(DEFAULT_VARIANTS))
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--device", default="", help="cuda, cpu, or empty for auto.")
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--save_predictions", action="store_true")
    parser.add_argument("--save_text_preview", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
