#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_codebert.py

Train a CodeBERT binary vulnerability classifier for one input view. The best
model and the classification threshold are selected on the validation set, then
applied once to the test set.

Example:

python src/train_codebert.py --train_file outputs/lin_et_al/experiment_inputs_screened/code_vul_path_top3_screened/train.jsonl --valid_file outputs/lin_et_al/experiment_inputs_screened/code_vul_path_top3_screened/valid.jsonl --test_file outputs/lin_et_al/experiment_inputs_screened/code_vul_path_top3_screened/test.jsonl --model_name microsoft/codebert-base --output_dir outputs/lin_et_al/checkpoints_screened_thr/code_vul_path_top3_screened --max_length 512 --batch_size 16 --eval_batch_size 32 --learning_rate 2e-5 --epochs 20 --class_weight balanced --metric_for_best f1 --threshold_metric f1 --save_predictions
"""

import argparse
import json
import os
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x


# ============================================================
# Utils
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
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


def write_json(path: str, obj: Dict[str, Any]):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_jsonl(path: str, rows: List[Dict[str, Any]]):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        if np.isnan(x):
            return None
    except Exception:
        pass
    try:
        return float(x)
    except Exception:
        return None


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Dataset
# ============================================================

class JsonlTextDataset(Dataset):
    def __init__(
        self,
        path: str,
        text_key: str = "input_text",
        label_key: str = "target",
        max_samples: int = -1,
    ):
        self.path = path
        self.text_key = text_key
        self.label_key = label_key
        rows = read_jsonl(path)
        if max_samples > 0:
            rows = rows[:max_samples]

        self.examples = []
        for i, row in enumerate(rows):
            text = row.get(text_key, "")
            label = row.get(label_key, None)
            if text is None:
                text = ""
            text = str(text)
            if label is None:
                continue
            try:
                label = int(label)
            except Exception:
                continue
            if label not in [0, 1]:
                continue

            meta = {
                "idx": row.get("idx", i),
                "project": row.get("project", ""),
                "func_name": row.get("func_name", ""),
                "experiment_name": row.get("experiment_name", ""),
            }
            self.examples.append({"text": text, "label": label, "meta": meta})

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx: int):
        return self.examples[idx]


class Collator:
    def __init__(self, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        texts = [x["text"] for x in batch]
        labels = torch.tensor([x["label"] for x in batch], dtype=torch.long)
        metas = [x["meta"] for x in batch]

        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded["labels"] = labels
        encoded["metas"] = metas
        encoded["texts"] = texts
        return encoded


# ============================================================
# Metrics and threshold tuning
# ============================================================

def compute_metrics(labels: List[int], preds: List[int], probs: List[float]) -> Dict[str, Any]:
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="binary",
        zero_division=0,
    )

    auc = None
    ap = None
    if len(set(labels)) == 2:
        try:
            auc = roc_auc_score(labels, probs)
        except Exception:
            auc = None
        try:
            ap = average_precision_score(labels, probs)
        except Exception:
            ap = None

    try:
        tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    except Exception:
        tn, fp, fn, tp = 0, 0, 0, 0

    return {
        "acc": safe_float(acc),
        "precision": safe_float(precision),
        "recall": safe_float(recall),
        "f1": safe_float(f1),
        "auc": safe_float(auc),
        "ap": safe_float(ap),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def get_best_metric(metrics: Dict[str, Any], metric_name: str) -> float:
    value = metrics.get(metric_name, None)
    if value is None:
        return -1.0
    return float(value)


def metrics_at_threshold(labels: List[int], probs: List[float], threshold: float) -> Dict[str, Any]:
    preds = [1 if p >= threshold else 0 for p in probs]
    metrics = compute_metrics(labels, preds, probs)
    metrics["threshold"] = float(threshold)
    return metrics


def make_thresholds(t_min: float, t_max: float, step: float) -> List[float]:
    if step <= 0:
        raise ValueError("--threshold_step must be > 0")
    thresholds = []
    t = t_min
    while t <= t_max + 1e-12:
        thresholds.append(round(float(t), 6))
        t += step
    if t_min <= 0.5 <= t_max and 0.5 not in thresholds:
        thresholds.append(0.5)
    return sorted(set(thresholds))


def find_best_threshold(
    labels: List[int],
    probs: List[float],
    metric_name: str = "f1",
    threshold_min: float = 0.05,
    threshold_max: float = 0.95,
    threshold_step: float = 0.01,
) -> Tuple[float, Dict[str, Any], List[Dict[str, Any]]]:
    thresholds = make_thresholds(threshold_min, threshold_max, threshold_step)
    table = []

    best_threshold = 0.5
    best_metrics = metrics_at_threshold(labels, probs, 0.5)
    best_score = get_best_metric(best_metrics, metric_name)

    for threshold in thresholds:
        m = metrics_at_threshold(labels, probs, threshold)
        table.append(m)
        score = get_best_metric(m, metric_name)

        # Primary: selected metric. Secondary: F1. Tie: closer to 0.5.
        if score > best_score:
            best_threshold = threshold
            best_metrics = m
            best_score = score
        elif abs(score - best_score) <= 1e-12:
            this_f1 = get_best_metric(m, "f1")
            best_f1 = get_best_metric(best_metrics, "f1")
            if this_f1 > best_f1:
                best_threshold = threshold
                best_metrics = m
                best_score = score
            elif abs(this_f1 - best_f1) <= 1e-12:
                if abs(threshold - 0.5) < abs(best_threshold - 0.5):
                    best_threshold = threshold
                    best_metrics = m
                    best_score = score

    return float(best_threshold), best_metrics, table


def labels_probs_from_rows(rows: List[Dict[str, Any]]) -> Tuple[List[int], List[float]]:
    labels = [int(r["label"]) for r in rows]
    probs = [float(r["prob_vulnerable"]) for r in rows]
    return labels, probs


def apply_threshold_to_rows(rows: List[Dict[str, Any]], threshold: float) -> List[Dict[str, Any]]:
    out = []
    for r in rows:
        prob = float(r.get("prob_vulnerable", 0.0))
        nr = dict(r)
        nr["pred"] = int(prob >= threshold)
        nr["threshold"] = float(threshold)
        out.append(nr)
    return out


# ============================================================
# Class weights
# ============================================================

def compute_class_weights(dataset: JsonlTextDataset, device: torch.device) -> Optional[torch.Tensor]:
    labels = [ex["label"] for ex in dataset.examples]
    n_total = len(labels)
    n_pos = sum(labels)
    n_neg = n_total - n_pos
    if n_total == 0 or n_pos == 0 or n_neg == 0:
        return None

    w0 = n_total / (2.0 * n_neg)
    w1 = n_total / (2.0 * n_pos)
    return torch.tensor([w0, w1], dtype=torch.float, device=device)


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    dataloader: DataLoader,
    device: torch.device,
    loss_fn: Optional[nn.Module] = None,
    desc: str = "Evaluating",
    threshold: float = 0.5,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []
    all_losses = []
    prediction_rows = []

    for batch in tqdm(dataloader, desc=desc):
        metas = batch.pop("metas")
        texts = batch.pop("texts")
        labels = batch["labels"].to(device)

        model_inputs = {
            k: v.to(device)
            for k, v in batch.items()
            if k in ["input_ids", "attention_mask", "token_type_ids"]
        }

        outputs = model(**model_inputs)
        logits = outputs.logits

        if loss_fn is not None:
            loss = loss_fn(logits, labels)
            all_losses.append(loss.item())

        probs = torch.softmax(logits, dim=-1)[:, 1]
        preds = (probs >= threshold).long()

        labels_np = labels.detach().cpu().numpy().tolist()
        preds_np = preds.detach().cpu().numpy().tolist()
        probs_np = probs.detach().cpu().numpy().tolist()

        all_labels.extend(labels_np)
        all_preds.extend(preds_np)
        all_probs.extend(probs_np)

        for meta, text, gold, pred, prob in zip(metas, texts, labels_np, preds_np, probs_np):
            prediction_rows.append(
                {
                    **meta,
                    "label": int(gold),
                    "pred": int(pred),
                    "prob_vulnerable": float(prob),
                    "threshold": float(threshold),
                    "input_text_preview": text[:500],
                }
            )

    metrics = compute_metrics(all_labels, all_preds, all_probs)
    metrics["threshold"] = float(threshold)
    metrics["loss"] = float(np.mean(all_losses)) if all_losses else None
    return metrics, prediction_rows


# ============================================================
# Training
# ============================================================

def train(args):
    set_seed(args.seed)

    device = get_device()
    ensure_dir(args.output_dir)

    print("=" * 80)
    print("Training CodeBERT classifier with validation threshold tuning")
    print("=" * 80)
    print(f"Device      : {device}")
    print(f"Model       : {args.model_name}")
    print(f"Train file  : {args.train_file}")
    print(f"Valid file  : {args.valid_file}")
    print(f"Test file   : {args.test_file}")
    print(f"Output dir  : {args.output_dir}")
    print(f"Tune threshold on valid: {not args.no_threshold_tuning}")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    train_dataset = JsonlTextDataset(
        args.train_file,
        text_key=args.text_key,
        label_key=args.label_key,
        max_samples=args.max_train_samples,
    )
    valid_dataset = JsonlTextDataset(
        args.valid_file,
        text_key=args.text_key,
        label_key=args.label_key,
        max_samples=args.max_valid_samples,
    )
    test_dataset = JsonlTextDataset(
        args.test_file,
        text_key=args.text_key,
        label_key=args.label_key,
        max_samples=args.max_test_samples,
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Valid samples: {len(valid_dataset)}")
    print(f"Test samples : {len(test_dataset)}")

    if len(train_dataset) == 0:
        raise ValueError("Training set is empty. Please check train_file, text_key, and label_key.")

    collator = Collator(tokenizer, max_length=args.max_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collator,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collator,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collator,
    )

    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)
    model.to(device)

    class_weights = None
    if args.class_weight == "balanced":
        class_weights = compute_class_weights(train_dataset, device=device)
        if class_weights is not None:
            print(f"Using balanced class weights: {class_weights.detach().cpu().tolist()}")
        else:
            print("Class weights are not used because one class is missing.")

    loss_fn = nn.CrossEntropyLoss(weight=class_weights) if class_weights is not None else nn.CrossEntropyLoss()

    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=args.learning_rate,
        eps=args.adam_epsilon,
    )

    total_update_steps = (
        len(train_loader) // args.gradient_accumulation_steps
        + int(len(train_loader) % args.gradient_accumulation_steps != 0)
    ) * args.epochs

    warmup_steps = int(total_update_steps * args.warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_update_steps,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16 and device.type == "cuda")

    best_score = -1.0
    best_epoch = -1
    best_threshold = 0.5
    best_valid_metrics_default = None
    best_valid_metrics_tuned = None
    best_threshold_table = []
    patience_counter = 0
    training_log = []

    best_model_dir = os.path.join(args.output_dir, "best_model")
    last_model_dir = os.path.join(args.output_dir, "last_model")
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses = []
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        optimizer.zero_grad()

        for step, batch in enumerate(progress, start=1):
            batch.pop("metas")
            batch.pop("texts")
            labels = batch["labels"].to(device)

            model_inputs = {
                k: v.to(device)
                for k, v in batch.items()
                if k in ["input_ids", "attention_mask", "token_type_ids"]
            }

            with torch.cuda.amp.autocast(enabled=args.fp16 and device.type == "cuda"):
                outputs = model(**model_inputs)
                logits = outputs.logits
                loss = loss_fn(logits, labels)
                loss = loss / args.gradient_accumulation_steps

            scaler.scale(loss).backward()
            epoch_losses.append(loss.item() * args.gradient_accumulation_steps)

            if step % args.gradient_accumulation_steps == 0 or step == len(train_loader):
                if args.max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            progress.set_postfix(
                {
                    "loss": f"{np.mean(epoch_losses):.4f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                }
            )

        train_loss = float(np.mean(epoch_losses)) if epoch_losses else None

        # Evaluate valid at threshold 0.5 first to get probabilities.
        valid_metrics_default, valid_predictions_default = evaluate(
            model=model,
            dataloader=valid_loader,
            device=device,
            loss_fn=loss_fn,
            desc=f"Valid epoch {epoch} threshold 0.5",
            threshold=0.5,
        )

        valid_labels, valid_probs = labels_probs_from_rows(valid_predictions_default)

        if args.no_threshold_tuning:
            epoch_threshold = 0.5
            valid_metrics_tuned = dict(valid_metrics_default)
            threshold_table = [valid_metrics_default]
        else:
            epoch_threshold, valid_metrics_tuned, threshold_table = find_best_threshold(
                labels=valid_labels,
                probs=valid_probs,
                metric_name=args.threshold_metric,
                threshold_min=args.threshold_min,
                threshold_max=args.threshold_max,
                threshold_step=args.threshold_step,
            )
            valid_metrics_tuned["loss"] = valid_metrics_default.get("loss", None)

        score = get_best_metric(valid_metrics_tuned, args.metric_for_best)

        epoch_log = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": train_loss,
            "valid_metrics_default_threshold_0_5": valid_metrics_default,
            "valid_best_threshold": float(epoch_threshold),
            "valid_metrics_at_best_threshold": valid_metrics_tuned,
            "metric_for_best": args.metric_for_best,
            "threshold_metric": args.threshold_metric,
            "best_metric_value_this_epoch": score,
        }
        training_log.append(epoch_log)

        print("\n" + "-" * 80)
        print(f"Epoch {epoch} finished")
        print(f"Train loss: {train_loss}")
        print("Valid metrics at threshold 0.5:")
        print(json.dumps(valid_metrics_default, ensure_ascii=False, indent=2))
        print(f"Best valid threshold by {args.threshold_metric}: {epoch_threshold}")
        print("Valid metrics at best threshold:")
        print(json.dumps(valid_metrics_tuned, ensure_ascii=False, indent=2))
        print("-" * 80 + "\n")

        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_threshold = float(epoch_threshold)
            best_valid_metrics_default = dict(valid_metrics_default)
            best_valid_metrics_tuned = dict(valid_metrics_tuned)
            best_threshold_table = list(threshold_table)
            patience_counter = 0

            ensure_dir(best_model_dir)
            model.save_pretrained(best_model_dir)
            tokenizer.save_pretrained(best_model_dir)

            write_json(
                os.path.join(args.output_dir, "best_valid_metrics.json"),
                {
                    "best_epoch": best_epoch,
                    "best_score": best_score,
                    "metric_for_best": args.metric_for_best,
                    "threshold_metric": args.threshold_metric,
                    "best_threshold": best_threshold,
                    "valid_metrics_default_threshold_0_5": best_valid_metrics_default,
                    "valid_metrics_at_best_threshold": best_valid_metrics_tuned,
                },
            )
            write_json(
                os.path.join(args.output_dir, "best_threshold_table.json"),
                {
                    "best_epoch": best_epoch,
                    "threshold_metric": args.threshold_metric,
                    "best_threshold": best_threshold,
                    "threshold_table": best_threshold_table,
                },
            )
            print(f"[Best] Saved best model to {best_model_dir}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{args.early_stop_patience}")
            if args.early_stop_patience > 0 and patience_counter >= args.early_stop_patience:
                print("Early stopping triggered.")
                break

        write_json(os.path.join(args.output_dir, "training_log.json"), {"log": training_log})

    ensure_dir(last_model_dir)
    model.save_pretrained(last_model_dir)
    tokenizer.save_pretrained(last_model_dir)

    print("=" * 80)
    print(f"Training finished. Best epoch: {best_epoch}, best {args.metric_for_best}: {best_score}")
    print(f"Best threshold during training: {best_threshold}")
    print("=" * 80)

    # Load best model for final validation and testing.
    print(f"Loading best model from {best_model_dir} for final testing...")
    best_model = AutoModelForSequenceClassification.from_pretrained(best_model_dir)
    best_model.to(device)

    # Recompute valid threshold using the loaded best checkpoint.
    valid_metrics_default_final, valid_predictions_default = evaluate(
        model=best_model,
        dataloader=valid_loader,
        device=device,
        loss_fn=loss_fn,
        desc="Valid best model threshold 0.5",
        threshold=0.5,
    )
    valid_labels, valid_probs = labels_probs_from_rows(valid_predictions_default)

    if args.no_threshold_tuning:
        final_threshold = 0.5
        valid_metrics_tuned_final = dict(valid_metrics_default_final)
        final_threshold_table = [valid_metrics_default_final]
    else:
        final_threshold, valid_metrics_tuned_final, final_threshold_table = find_best_threshold(
            labels=valid_labels,
            probs=valid_probs,
            metric_name=args.threshold_metric,
            threshold_min=args.threshold_min,
            threshold_max=args.threshold_max,
            threshold_step=args.threshold_step,
        )
        valid_metrics_tuned_final["loss"] = valid_metrics_default_final.get("loss", None)

    valid_predictions_tuned = apply_threshold_to_rows(valid_predictions_default, final_threshold)

    # Test at 0.5 for reference.
    test_metrics_default, test_predictions_default = evaluate(
        model=best_model,
        dataloader=test_loader,
        device=device,
        loss_fn=loss_fn,
        desc="Testing threshold 0.5",
        threshold=0.5,
    )

    # Test at validation-selected threshold.
    test_metrics_tuned, test_predictions_tuned = evaluate(
        model=best_model,
        dataloader=test_loader,
        device=device,
        loss_fn=loss_fn,
        desc=f"Testing tuned threshold {final_threshold}",
        threshold=final_threshold,
    )

    final_metrics = {
        "best_epoch": best_epoch,
        "metric_for_best": args.metric_for_best,
        "threshold_metric": args.threshold_metric,
        "best_valid_score_during_training": best_score,
        "best_threshold_during_training": best_threshold,
        "final_threshold_selected_on_valid": final_threshold,
        "valid_metrics_default_threshold_0_5": valid_metrics_default_final,
        "valid_metrics_at_selected_threshold": valid_metrics_tuned_final,
        "test_metrics_default_threshold_0_5": test_metrics_default,
        "test_metrics_at_selected_threshold": test_metrics_tuned,
        "args": vars(args),
    }

    write_json(os.path.join(args.output_dir, "test_metrics.json"), final_metrics)

    write_json(
        os.path.join(args.output_dir, "threshold_tuning.json"),
        {
            "threshold_metric": args.threshold_metric,
            "final_threshold_selected_on_valid": final_threshold,
            "valid_metrics_default_threshold_0_5": valid_metrics_default_final,
            "valid_metrics_at_selected_threshold": valid_metrics_tuned_final,
            "test_metrics_default_threshold_0_5": test_metrics_default,
            "test_metrics_at_selected_threshold": test_metrics_tuned,
            "threshold_table": final_threshold_table,
        },
    )

    if args.save_predictions:
        write_jsonl(os.path.join(args.output_dir, "valid_predictions_default_threshold_0_5.jsonl"), valid_predictions_default)
        write_jsonl(os.path.join(args.output_dir, "valid_predictions.jsonl"), valid_predictions_tuned)
        write_jsonl(os.path.join(args.output_dir, "test_predictions_default_threshold_0_5.jsonl"), test_predictions_default)
        write_jsonl(os.path.join(args.output_dir, "test_predictions.jsonl"), test_predictions_tuned)

    print("\nFinal validation-selected threshold:")
    print(final_threshold)

    print("\nFinal test metrics at threshold 0.5:")
    print(json.dumps(test_metrics_default, ensure_ascii=False, indent=2))

    print("\nFinal test metrics at validation-selected threshold:")
    print(json.dumps(test_metrics_tuned, ensure_ascii=False, indent=2))

    print(f"\nSaved to: {args.output_dir}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--valid_file", type=str, required=True)
    parser.add_argument("--test_file", type=str, required=True)

    parser.add_argument(
        "--model_name",
        type=str,
        default="microsoft/codebert-base",
        help="HuggingFace model name or local path.",
    )

    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--text_key", type=str, default="input_text")
    parser.add_argument("--label_key", type=str, default="target")

    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=32)

    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)

    parser.add_argument(
        "--class_weight",
        type=str,
        default="none",
        choices=["none", "balanced"],
        help="Use balanced class weights for imbalanced datasets.",
    )

    parser.add_argument(
        "--metric_for_best",
        type=str,
        default="f1",
        choices=["f1", "auc", "ap", "acc", "precision", "recall"],
        help="Metric used to select best checkpoint. With threshold tuning, this metric is computed at the valid-selected threshold.",
    )

    parser.add_argument(
        "--threshold_metric",
        type=str,
        default="f1",
        choices=["f1", "acc", "precision", "recall"],
        help="Metric used to select the best threshold on the validation set.",
    )
    parser.add_argument("--threshold_min", type=float, default=0.05)
    parser.add_argument("--threshold_max", type=float, default=0.95)
    parser.add_argument("--threshold_step", type=float, default=0.01)
    parser.add_argument(
        "--no_threshold_tuning",
        action="store_true",
        help="Disable validation threshold tuning and use threshold 0.5.",
    )

    parser.add_argument("--early_stop_patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--fp16", action="store_true")

    parser.add_argument(
        "--save_predictions",
        action="store_true",
        help="Save valid/test predictions. test_predictions.jsonl uses the validation-selected threshold.",
    )

    parser.add_argument("--max_train_samples", type=int, default=-1)
    parser.add_argument("--max_valid_samples", type=int, default=-1)
    parser.add_argument("--max_test_samples", type=int, default=-1)

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
