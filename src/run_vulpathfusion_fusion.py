#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import math
import os
from itertools import combinations, product
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)


def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
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


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    ensure_dir(os.path.dirname(path))
    if not rows:
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write("")
        return

    fieldnames = list(rows[0].keys())
    for row in rows[1:]:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_key(row: Dict[str, Any]) -> str:
    parts = []
    for key in ["project", "func_name", "idx"]:
        if key in row and row[key] is not None:
            parts.append(str(row[key]))
    if parts:
        return "||".join(parts)
    if "input_text_preview" in row:
        return str(hash(row["input_text_preview"]))
    raise ValueError("Prediction row has no stable key fields.")


def parse_methods(text: str) -> List[str]:
    methods = [x.strip() for x in text.split(",") if x.strip()]
    seen = set()
    out = []
    for method in methods:
        if method in seen:
            continue
        seen.add(method)
        out.append(method)
    return out


def prediction_path(checkpoint_root: str, method: str, split: str) -> str:
    return os.path.join(checkpoint_root, method, f"{split}_predictions.jsonl")


def load_predictions(
    checkpoint_root: str,
    methods: List[str],
    split: str,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    out = {}
    for method in methods:
        path = prediction_path(checkpoint_root, method, split)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing {split} predictions for {method}: {path}")
        keyed = {}
        for row in read_jsonl(path):
            keyed[make_key(row)] = row
        out[method] = keyed
    return out


def aligned_arrays(
    predictions: Dict[str, Dict[str, Dict[str, Any]]],
    methods: List[str],
) -> Tuple[List[str], np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    common_keys = set(predictions[methods[0]].keys())
    for method in methods[1:]:
        common_keys &= set(predictions[method].keys())
    keys = sorted(common_keys)
    if not keys:
        raise ValueError(f"No common prediction keys across methods: {methods}")

    labels = []
    probs = []
    metas = []
    for key in keys:
        first = predictions[methods[0]][key]
        label = int(first.get("label", first.get("target")))
        labels.append(label)
        metas.append({
            "project": first.get("project"),
            "func_name": first.get("func_name"),
            "idx": first.get("idx"),
            "key": key,
        })
        row_probs = []
        for method in methods:
            row = predictions[method][key]
            other_label = int(row.get("label", row.get("target")))
            if other_label != label:
                raise ValueError(f"Label mismatch for key {key} in method {method}")
            row_probs.append(float(row["prob_vulnerable"]))
        probs.append(row_probs)

    return keys, np.asarray(labels, dtype=np.int64), np.asarray(probs, dtype=np.float64), metas


def compute_metrics(labels: np.ndarray, probs: np.ndarray, threshold: float) -> Dict[str, Any]:
    preds = (probs >= threshold).astype(np.int64)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="binary",
        zero_division=0,
    )
    try:
        auc = roc_auc_score(labels, probs)
    except ValueError:
        auc = None
    try:
        ap = average_precision_score(labels, probs)
    except ValueError:
        ap = None
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    return {
        "acc": float(accuracy_score(labels, preds)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": None if auc is None else float(auc),
        "ap": None if ap is None else float(ap),
        "threshold": float(threshold),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def compute_fixed_threshold_reports(
    result: Dict[str, Any],
    thresholds: List[float],
) -> Dict[str, Dict[str, Any]]:
    reports: Dict[str, Dict[str, Any]] = {}
    for threshold in thresholds:
        key = f"{float(threshold):g}"
        reports[key] = {
            "threshold": float(threshold),
            "valid_metrics": compute_metrics(
                result["valid_labels"],
                result["valid_probs"],
                float(threshold),
            ),
            "test_metrics": compute_metrics(
                result["test_labels"],
                result["test_probs"],
                float(threshold),
            ),
        }
    return reports


def parse_float_csv(text: str) -> List[float]:
    values: List[float] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        value = float(item)
        if not 0.0 <= value <= 1.0:
            raise ValueError("Fixed thresholds must be in [0, 1].")
        values.append(value)
    return values


def compute_metrics_fast(
    labels: np.ndarray,
    probs: np.ndarray,
    threshold: float,
    auc: Optional[float],
    ap: Optional[float],
) -> Dict[str, Any]:
    preds = probs >= threshold
    positives = labels == 1
    negatives = ~positives

    tp = int(np.sum(preds & positives))
    fp = int(np.sum(preds & negatives))
    fn = int(np.sum((~preds) & positives))
    tn = int(np.sum((~preds) & negatives))

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    acc = (tp + tn) / len(labels) if len(labels) else 0.0

    return {
        "acc": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": auc,
        "ap": ap,
        "threshold": float(threshold),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def safe_metric(metrics: Dict[str, Any], name: str) -> float:
    value = metrics.get(name)
    if value is None:
        return -1.0
    return float(value)


def is_better(
    candidate: Dict[str, Any],
    best: Optional[Dict[str, Any]],
    selection_metric: str,
    threshold: Optional[float] = None,
    best_threshold: Optional[float] = None,
) -> bool:
    if best is None:
        return True

    c_score = safe_metric(candidate, selection_metric)
    b_score = safe_metric(best, selection_metric)
    if c_score > b_score + 1e-12:
        return True
    if abs(c_score - b_score) > 1e-12:
        return False

    # Tie-breakers favor F1, then AP, then a threshold closer to 0.5.
    for metric_name in ["f1", "ap"]:
        c_value = safe_metric(candidate, metric_name)
        b_value = safe_metric(best, metric_name)
        if c_value > b_value + 1e-12:
            return True
        if abs(c_value - b_value) > 1e-12:
            return False

    if threshold is not None and best_threshold is not None:
        return abs(threshold - 0.5) < abs(best_threshold - 0.5)
    return False


def make_thresholds(t_min: float, t_max: float, step: float) -> List[float]:
    if step <= 0:
        raise ValueError("--threshold_step must be > 0")
    vals = []
    t = t_min
    while t <= t_max + 1e-12:
        vals.append(round(float(t), 6))
        t += step
    if t_min <= 0.5 <= t_max and 0.5 not in vals:
        vals.append(0.5)
    return sorted(set(vals))


def weight_units(weight_step: float) -> int:
    if weight_step <= 0 or weight_step > 1:
        raise ValueError("--weight_step must be in (0, 1].")
    units = int(round(1.0 / weight_step))
    if abs(units * weight_step - 1.0) > 1e-8:
        raise ValueError("--weight_step must divide 1.0 exactly, e.g. 0.1, 0.05, 0.025.")
    return units


def count_weight_configs(n_methods: int, weight_step: float) -> int:
    units = weight_units(weight_step)
    if n_methods <= 1:
        return 1
    return math.comb(units + n_methods - 1, n_methods - 1)


def make_weight_grid(n_methods: int, weight_step: float) -> List[np.ndarray]:
    if n_methods < 2:
        return [np.ones(1, dtype=np.float64)]

    units = weight_units(weight_step)
    grids = []
    for combo in product(range(units + 1), repeat=n_methods - 1):
        used = sum(combo)
        if used > units:
            continue
        last = units - used
        weights = np.asarray(list(combo) + [last], dtype=np.float64) / float(units)
        grids.append(weights)
    return grids


def tune(
    labels: np.ndarray,
    probs_matrix: np.ndarray,
    methods: List[str],
    thresholds: List[float],
    weight_step: float,
    selection_metric: str,
    keep_rows: bool = True,
) -> Tuple[np.ndarray, float, Dict[str, Any], List[Dict[str, Any]]]:
    best_weights = None
    best_threshold = 0.5
    best_metrics = None
    rows = []

    for weights in make_weight_grid(len(methods), weight_step):
        probs = probs_matrix @ weights
        try:
            auc = float(roc_auc_score(labels, probs))
        except ValueError:
            auc = None
        try:
            ap = float(average_precision_score(labels, probs))
        except ValueError:
            ap = None

        for threshold in thresholds:
            metrics = compute_metrics_fast(labels, probs, threshold, auc=auc, ap=ap)
            if keep_rows:
                row = {
                    "threshold": threshold,
                    "acc": metrics["acc"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                    "ap": metrics["ap"],
                    "auc": metrics["auc"],
                    "tp": metrics["tp"],
                    "fp": metrics["fp"],
                    "fn": metrics["fn"],
                    "tn": metrics["tn"],
                }
                for method, weight in zip(methods, weights):
                    row[f"weight_{method}"] = float(weight)
                rows.append(row)

            if is_better(metrics, best_metrics, selection_metric, threshold, best_threshold):
                best_weights = weights.copy()
                best_threshold = threshold
                best_metrics = metrics

    if best_weights is None or best_metrics is None:
        raise ValueError("Failed to tune ensemble.")
    return best_weights, best_threshold, best_metrics, rows


def make_prediction_rows(
    metas: List[Dict[str, Any]],
    labels: np.ndarray,
    probs: np.ndarray,
    threshold: float,
) -> List[Dict[str, Any]]:
    preds = (probs >= threshold).astype(np.int64)
    rows = []
    for meta, label, pred, prob in zip(metas, labels, preds, probs):
        rows.append({
            **meta,
            "label": int(label),
            "pred": int(pred),
            "prob_vulnerable": float(prob),
            "threshold": float(threshold),
        })
    return rows


def run_ensemble(
    valid_predictions: Dict[str, Dict[str, Dict[str, Any]]],
    test_predictions: Dict[str, Dict[str, Dict[str, Any]]],
    methods: List[str],
    thresholds: List[float],
    weight_step: float,
    selection_metric: str,
    keep_tuning_rows: bool,
) -> Dict[str, Any]:
    _, valid_labels, valid_probs_matrix, valid_metas = aligned_arrays(valid_predictions, methods)
    _, test_labels, test_probs_matrix, test_metas = aligned_arrays(test_predictions, methods)

    weights, threshold, valid_metrics, tuning_rows = tune(
        labels=valid_labels,
        probs_matrix=valid_probs_matrix,
        methods=methods,
        thresholds=thresholds,
        weight_step=weight_step,
        selection_metric=selection_metric,
        keep_rows=keep_tuning_rows,
    )

    valid_probs = valid_probs_matrix @ weights
    test_probs = test_probs_matrix @ weights
    test_metrics = compute_metrics(test_labels, test_probs, threshold)

    return {
        "methods": methods,
        "weights_array": weights,
        "weights": {method: float(weight) for method, weight in zip(methods, weights)},
        "threshold": float(threshold),
        "valid_metrics": valid_metrics,
        "test_metrics": test_metrics,
        "valid_labels": valid_labels,
        "test_labels": test_labels,
        "valid_probs": valid_probs,
        "test_probs": test_probs,
        "valid_metas": valid_metas,
        "test_metas": test_metas,
        "tuning_rows": tuning_rows,
        "common_valid_rows": int(len(valid_labels)),
        "common_test_rows": int(len(test_labels)),
    }


def result_to_public_dict(result: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    public = {
        "methods": result["methods"],
        "weights": result["weights"],
        "threshold_selected_on_valid": result["threshold"],
        "valid_metrics": result["valid_metrics"],
        "test_metrics": result["test_metrics"],
        "common_valid_rows": result["common_valid_rows"],
        "common_test_rows": result["common_test_rows"],
        "args": vars(args),
    }
    fixed_thresholds = parse_float_csv(getattr(args, "fixed_thresholds", ""))
    if fixed_thresholds:
        public["fixed_threshold_reports"] = compute_fixed_threshold_reports(
            result,
            fixed_thresholds,
        )
    return public


def save_ensemble_result(result: Dict[str, Any], output_dir: str, args: argparse.Namespace) -> None:
    ensure_dir(output_dir)
    write_csv(os.path.join(output_dir, "ensemble_tuning_table.csv"), result["tuning_rows"])
    write_jsonl(
        os.path.join(output_dir, "valid_predictions.jsonl"),
        make_prediction_rows(result["valid_metas"], result["valid_labels"], result["valid_probs"], result["threshold"]),
    )
    write_jsonl(
        os.path.join(output_dir, "test_predictions.jsonl"),
        make_prediction_rows(result["test_metas"], result["test_labels"], result["test_probs"], result["threshold"]),
    )
    public = result_to_public_dict(result, args)
    write_json(os.path.join(output_dir, "ensemble_metrics.json"), public)
    if "fixed_threshold_reports" in public:
        write_json(
            os.path.join(output_dir, "fixed_threshold_metrics.json"),
            public["fixed_threshold_reports"],
        )


def combo_summary_row(
    combo_id: str,
    result: Dict[str, Any],
    status: str = "ok",
    weight_config_count: Optional[int] = None,
) -> Dict[str, Any]:
    valid_metrics = result.get("valid_metrics", {})
    test_metrics = result.get("test_metrics", {})
    return {
        "combo_id": combo_id,
        "status": status,
        "method_count": len(result.get("methods", [])),
        "methods": ",".join(result.get("methods", [])),
        "weights_json": json.dumps(result.get("weights", {}), ensure_ascii=False, sort_keys=True),
        "threshold": result.get("threshold"),
        "valid_f1": valid_metrics.get("f1"),
        "valid_precision": valid_metrics.get("precision"),
        "valid_recall": valid_metrics.get("recall"),
        "valid_ap": valid_metrics.get("ap"),
        "valid_auc": valid_metrics.get("auc"),
        "test_f1": test_metrics.get("f1"),
        "test_precision": test_metrics.get("precision"),
        "test_recall": test_metrics.get("recall"),
        "test_ap": test_metrics.get("ap"),
        "test_auc": test_metrics.get("auc"),
        "test_tp": test_metrics.get("tp"),
        "test_fp": test_metrics.get("fp"),
        "test_fn": test_metrics.get("fn"),
        "test_tn": test_metrics.get("tn"),
        "weight_config_count": weight_config_count,
    }


def write_combo_markdown(path: str, rows: List[Dict[str, Any]], limit: int) -> None:
    ensure_dir(os.path.dirname(path))
    headers = [
        "Rank",
        "Methods",
        "Valid F1",
        "Test F1",
        "Test P",
        "Test R",
        "Threshold",
        "Weights",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for rank, row in enumerate(rows[:limit], start=1):
        lines.append(
            "| "
            + " | ".join([
                str(rank),
                str(row.get("methods", "")),
                f"{float(row.get('valid_f1') or 0.0):.4f}",
                f"{float(row.get('test_f1') or 0.0):.4f}",
                f"{float(row.get('test_precision') or 0.0):.4f}",
                f"{float(row.get('test_recall') or 0.0):.4f}",
                f"{float(row.get('threshold') or 0.0):.4f}",
                str(row.get("weights_json", "")),
            ])
            + " |"
        )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def sorted_summary_rows(rows: List[Dict[str, Any]], metric: str) -> List[Dict[str, Any]]:
    key_name = f"valid_{metric}"
    return sorted(rows, key=lambda r: (r.get(key_name) is not None, float(r.get(key_name) or -1.0), float(r.get("valid_f1") or -1.0)), reverse=True)


def auto_search(
    valid_predictions: Dict[str, Dict[str, Dict[str, Any]]],
    test_predictions: Dict[str, Dict[str, Dict[str, Any]]],
    methods: List[str],
    thresholds: List[float],
    args: argparse.Namespace,
) -> None:
    min_size = max(2, args.combo_min_size)
    max_size = args.combo_max_size if args.combo_max_size > 0 else len(methods)
    max_size = min(max_size, len(methods))
    if min_size > max_size:
        raise ValueError("--combo_min_size cannot be larger than --combo_max_size.")

    summary_rows: List[Dict[str, Any]] = []
    skipped_rows: List[Dict[str, Any]] = []
    best_result: Optional[Dict[str, Any]] = None
    best_combo_id = ""
    best_by_test: Optional[Dict[str, Any]] = None
    best_by_test_combo_id = ""

    combo_index = 0
    for size in range(min_size, max_size + 1):
        for combo in combinations(methods, size):
            combo_methods = list(combo)
            combo_index += 1
            combo_id = f"combo_{combo_index:04d}_k{size}"
            n_weight_configs = count_weight_configs(size, args.weight_step)
            if n_weight_configs > args.max_weight_configs:
                skipped_rows.append({
                    "combo_id": combo_id,
                    "status": "skipped_too_many_weight_configs",
                    "method_count": size,
                    "methods": ",".join(combo_methods),
                    "weight_config_count": n_weight_configs,
                    "max_weight_configs": args.max_weight_configs,
                })
                continue

            print(f"[Combo] {combo_id}: {', '.join(combo_methods)} ({n_weight_configs} weight configs)", flush=True)
            result = run_ensemble(
                valid_predictions=valid_predictions,
                test_predictions=test_predictions,
                methods=combo_methods,
                thresholds=thresholds,
                weight_step=args.weight_step,
                selection_metric=args.selection_metric,
                keep_tuning_rows=False,
            )
            row = combo_summary_row(combo_id, result, weight_config_count=n_weight_configs)
            summary_rows.append(row)

            if best_result is None or is_better(result["valid_metrics"], best_result["valid_metrics"], args.selection_metric):
                best_result = result
                best_combo_id = combo_id

            if best_by_test is None or is_better(result["test_metrics"], best_by_test["test_metrics"], args.selection_metric):
                best_by_test = result
                best_by_test_combo_id = combo_id

    ranked_rows = sorted_summary_rows(summary_rows, args.selection_metric)
    ensure_dir(args.output_dir)
    write_csv(os.path.join(args.output_dir, "combination_summary.csv"), ranked_rows)
    write_json(
        os.path.join(args.output_dir, "combination_summary.json"),
        {
            "ranked_by": f"valid_{args.selection_metric}",
            "summary": ranked_rows,
            "skipped": skipped_rows,
            "args": vars(args),
        },
    )
    write_combo_markdown(os.path.join(args.output_dir, "combination_summary.md"), ranked_rows, args.top_k)
    if skipped_rows:
        write_csv(os.path.join(args.output_dir, "skipped_combinations.csv"), skipped_rows)

    if best_result is None:
        raise ValueError("No ensemble combination was evaluated. Relax --max_weight_configs or combo size settings.")

    # Re-run only the best valid-selected combo with full tuning rows for detailed output.
    best_result = run_ensemble(
        valid_predictions=valid_predictions,
        test_predictions=test_predictions,
        methods=best_result["methods"],
        thresholds=thresholds,
        weight_step=args.weight_step,
        selection_metric=args.selection_metric,
        keep_tuning_rows=True,
    )
    best_result["combo_id"] = best_combo_id
    save_ensemble_result(best_result, args.output_dir, args)

    official = result_to_public_dict(best_result, args)
    official["combo_id"] = best_combo_id
    official["selected_by"] = f"valid_{args.selection_metric}"
    official["best_by_test_diagnostic"] = None
    if best_by_test is not None:
        official["best_by_test_diagnostic"] = {
            "combo_id": best_by_test_combo_id,
            "methods": best_by_test["methods"],
            "weights": best_by_test["weights"],
            "threshold": best_by_test["threshold"],
            "valid_metrics": best_by_test["valid_metrics"],
            "test_metrics": best_by_test["test_metrics"],
            "note": "Diagnostic only. Do not report this as the official result because it is selected by test performance.",
        }
    write_json(os.path.join(args.output_dir, "best_ensemble_metrics.json"), official)

    print("\nBest valid-selected combo:")
    print(json.dumps({
        "combo_id": best_combo_id,
        "methods": best_result["methods"],
        "weights": best_result["weights"],
        "threshold": best_result["threshold"],
        "valid_metrics": best_result["valid_metrics"],
        "test_metrics": best_result["test_metrics"],
    }, ensure_ascii=False, indent=2))
    print(f"[Done] auto ensemble search -> {args.output_dir}")


def single_run(
    valid_predictions: Dict[str, Dict[str, Dict[str, Any]]],
    test_predictions: Dict[str, Dict[str, Dict[str, Any]]],
    methods: List[str],
    thresholds: List[float],
    args: argparse.Namespace,
) -> None:
    n_weight_configs = count_weight_configs(len(methods), args.weight_step)
    if n_weight_configs > args.max_weight_configs:
        raise ValueError(
            f"This combo has {n_weight_configs} weight configs, above --max_weight_configs={args.max_weight_configs}. "
            f"Increase --weight_step or --max_weight_configs."
        )

    result = run_ensemble(
        valid_predictions=valid_predictions,
        test_predictions=test_predictions,
        methods=methods,
        thresholds=thresholds,
        weight_step=args.weight_step,
        selection_metric=args.selection_metric,
        keep_tuning_rows=True,
    )
    save_ensemble_result(result, args.output_dir, args)

    print("Selected weights:")
    print(json.dumps(result["weights"], ensure_ascii=False, indent=2))
    print("Selected threshold:")
    print(result["threshold"])
    print("Valid metrics:")
    print(json.dumps(result["valid_metrics"], ensure_ascii=False, indent=2))
    print("Test metrics:")
    print(json.dumps(result["test_metrics"], ensure_ascii=False, indent=2))
    print(f"[Done] ensemble -> {args.output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_root", type=str, required=True)
    parser.add_argument("--methods", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--threshold_min", type=float, default=0.05)
    parser.add_argument("--threshold_max", type=float, default=0.95)
    parser.add_argument("--threshold_step", type=float, default=0.01)
    parser.add_argument("--weight_step", type=float, default=0.05)
    parser.add_argument(
        "--selection_metric",
        type=str,
        default="f1",
        choices=["f1", "ap", "auc", "recall", "precision", "acc"],
        help="Metric used on valid to select weights, threshold, and best combination.",
    )
    parser.add_argument("--auto_combinations", action="store_true")
    parser.add_argument("--combo_min_size", type=int, default=2)
    parser.add_argument(
        "--combo_max_size",
        type=int,
        default=4,
        help="Largest subset size to evaluate. Use 0 to allow all selected methods.",
    )
    parser.add_argument(
        "--max_weight_configs",
        type=int,
        default=60000,
        help="Skip combinations whose simplex weight grid is larger than this value.",
    )
    parser.add_argument("--top_k", type=int, default=20, help="Rows shown in combination_summary.md.")
    parser.add_argument(
        "--fixed_thresholds",
        default="",
        help=(
            "Optional comma-separated thresholds, for example 0.5. "
            "They are evaluated with the validation-selected model subset and weights unchanged."
        ),
    )
    args = parser.parse_args()

    methods = parse_methods(args.methods)
    if len(methods) < 2:
        raise ValueError("At least two methods are required.")

    thresholds = make_thresholds(args.threshold_min, args.threshold_max, args.threshold_step)
    valid_predictions = load_predictions(args.checkpoint_root, methods, "valid")
    test_predictions = load_predictions(args.checkpoint_root, methods, "test")

    if args.auto_combinations:
        auto_search(
            valid_predictions=valid_predictions,
            test_predictions=test_predictions,
            methods=methods,
            thresholds=thresholds,
            args=args,
        )
    else:
        single_run(
            valid_predictions=valid_predictions,
            test_predictions=test_predictions,
            methods=methods,
            thresholds=thresholds,
            args=args,
        )


if __name__ == "__main__":
    main()
