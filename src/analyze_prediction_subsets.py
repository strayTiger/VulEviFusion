#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate saved predictions on path/description availability subsets."""

import argparse
import json
import os
from typing import Any, Dict, Iterable, List, Optional

try:
    from sklearn.metrics import average_precision_score, roc_auc_score
except Exception:  # pragma: no cover
    average_precision_score = None
    roc_auc_score = None


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
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def make_key(row: Dict[str, Any]) -> str:
    parts = []
    for key in ["project", "func_name", "idx"]:
        if row.get(key) is not None:
            parts.append(str(row[key]))
    if parts:
        return "||".join(parts)
    return str(row.get("input_text_preview", ""))


def to_bool(value: Any) -> bool:
    return bool(value)


def feature_map(feature_rows: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for row in feature_rows:
        comps = row.get("components", {}) or {}
        out[make_key(row)] = {
            "has_path": to_bool(comps.get("vul_path_is_real") or comps.get("vul_path_text")),
            "has_desc": to_bool(comps.get("security_desc_is_real") or comps.get("security_desc")),
            "path_first": to_bool(comps.get("path_first")),
            "screened_path_count": int(comps.get("screened_path_count") or 0),
        }
    return out


def metric_counts(labels: List[int], preds: List[int]) -> Dict[str, int]:
    tp = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 1)
    fp = sum(1 for y, p in zip(labels, preds) if y == 0 and p == 1)
    fn = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 0)
    tn = sum(1 for y, p in zip(labels, preds) if y == 0 and p == 0)
    return {"tn": tn, "fp": fp, "fn": fn, "tp": tp}


def compute_metrics(rows: List[Dict[str, Any]], threshold: Optional[float]) -> Dict[str, Any]:
    labels = [int(row.get("label", row.get("target"))) for row in rows]
    probs = [float(row["prob_vulnerable"]) for row in rows]
    if threshold is None and all("pred" in row for row in rows):
        preds = [int(row["pred"]) for row in rows]
        used_threshold = None
    else:
        used_threshold = 0.5 if threshold is None else float(threshold)
        preds = [1 if prob >= used_threshold else 0 for prob in probs]

    counts = metric_counts(labels, preds)
    total = len(rows)
    precision = counts["tp"] / (counts["tp"] + counts["fp"]) if counts["tp"] + counts["fp"] else 0.0
    recall = counts["tp"] / (counts["tp"] + counts["fn"]) if counts["tp"] + counts["fn"] else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    acc = (counts["tp"] + counts["tn"]) / total if total else 0.0

    auc = None
    ap = None
    if total and len(set(labels)) > 1:
        if roc_auc_score is not None:
            auc = float(roc_auc_score(labels, probs))
        if average_precision_score is not None:
            ap = float(average_precision_score(labels, probs))

    return {
        "n": total,
        "positives": sum(labels),
        "negatives": total - sum(labels),
        "acc": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": auc,
        "ap": ap,
        "threshold": used_threshold,
        **counts,
    }


def compute_subset_report(
    prediction_rows: List[Dict[str, Any]],
    feature_rows: List[Dict[str, Any]],
    threshold: Optional[float] = None,
) -> Dict[str, Any]:
    features = feature_map(feature_rows)
    enriched = []
    missing_features = 0
    for row in prediction_rows:
        feat = features.get(make_key(row))
        if feat is None:
            missing_features += 1
            feat = {"has_path": False, "has_desc": False, "path_first": False, "screened_path_count": 0}
        enriched.append({**row, **feat})

    subsets = {
        "all": enriched,
        "has_path": [row for row in enriched if row["has_path"]],
        "no_path": [row for row in enriched if not row["has_path"]],
        "has_desc": [row for row in enriched if row["has_desc"]],
        "has_path_and_desc": [row for row in enriched if row["has_path"] and row["has_desc"]],
        "positive_has_path": [row for row in enriched if int(row.get("label", row.get("target"))) == 1 and row["has_path"]],
        "positive_no_path": [row for row in enriched if int(row.get("label", row.get("target"))) == 1 and not row["has_path"]],
    }

    report = {name: compute_metrics(rows, threshold=threshold) if rows else {"n": 0} for name, rows in subsets.items()}
    report["_meta"] = {
        "prediction_rows": len(prediction_rows),
        "feature_rows": len(feature_rows),
        "missing_features": missing_features,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()

    report = compute_subset_report(
        prediction_rows=read_jsonl(args.predictions),
        feature_rows=read_jsonl(args.features),
        threshold=args.threshold,
    )
    out_path = os.path.join(args.output_dir, "subset_metrics.json")
    write_json(out_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"[Done] subset metrics -> {out_path}")


if __name__ == "__main__":
    main()
