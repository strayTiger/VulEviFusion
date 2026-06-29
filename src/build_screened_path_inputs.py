#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build screened vulnerability-path input variants for CodeT5 experiments.

The script reuses existing outputs/{dataset}/vul_paths and optional LLM descriptions.
It does not regenerate DFG/path extraction. The goal is to keep only compact,
high-signal path evidence before training. It emits the five formal VulPathFusion
candidate views used after validation selects CodeT5 + Top-3 paths.
"""

import argparse
import json
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = lambda x, **kwargs: x


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path or not os.path.exists(path):
        return rows
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


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> int:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def clean_text(text: Any, max_chars: int = -1) -> str:
    if text is None:
        return ""
    text = str(text).replace("\r", " ").replace("\n", " ").replace("\t", " ")
    text = " ".join(text.split())
    if max_chars > 0 and len(text) > max_chars:
        return text[:max_chars].rstrip() + " ..."
    return text


def make_key(row: Dict[str, Any]) -> str:
    idx = row.get("idx")
    project = row.get("project")
    func_name = row.get("func_name")
    if idx is not None and project is not None:
        return f"idx::{project}::{idx}"
    if idx is not None:
        return f"idx::{idx}"
    if project is not None and func_name is not None:
        return f"func::{project}::{func_name}"
    if func_name is not None:
        return f"func::{func_name}"
    return json.dumps(row, sort_keys=True, ensure_ascii=False)


def make_map(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        out.setdefault(make_key(row), row)
    return out


def split_csv(text: str) -> List[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def has_real_security_desc(text: Any) -> bool:
    value = clean_text(text).strip()
    if not value:
        return False
    low = value.lower()
    if low in {"none", "null", "n/a", "na"}:
        return False
    if "no security description is available" in low:
        return False
    return True


def short_list(values: Any, limit: int = 5) -> str:
    if not isinstance(values, list):
        return "none"
    vals = [clean_text(v) for v in values if clean_text(v)]
    vals = vals[:limit]
    return ", ".join(vals) if vals else "none"


def expr_list(items: Any, key: str, limit: int = 2) -> str:
    if not isinstance(items, list):
        return "none"
    vals = []
    seen = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        value = clean_text(item.get(key, ""))
        if not value or value in seen:
            continue
        vals.append(value)
        seen.add(value)
        if len(vals) >= limit:
            break
    return " | ".join(vals) if vals else "none"


def path_score(path: Dict[str, Any]) -> float:
    risk = str(path.get("risk_level", "")).lower()
    path_type = str(path.get("path_type", "")).lower()
    sink_kind = str(path.get("sink_kind", "")).lower()
    sink_name = str(path.get("sink_name", "")).lower()

    score = {
        "high": 5.0,
        "medium": 3.5,
        "validation_related": 2.2,
        "checked_or_lower_risk": 1.4,
        "low_or_context": 0.2,
    }.get(risk, 1.0)

    if any(x in path_type for x in ["overflow", "format", "use_after", "double_free", "resource_leak", "lifetime"]):
        score += 2.0
    if any(x in path_type for x in ["array", "pointer", "dereference", "memory"]):
        score += 1.0
    if sink_kind == "call":
        score += 1.2
    if sink_kind in {"array_subscript", "pointer_dereference", "deallocation"}:
        score += 1.0
    if sink_name in {"memcpy", "strcpy", "strncpy", "sprintf", "snprintf", "strcat", "gets", "scanf"}:
        score += 1.4
    if path.get("flow_edges"):
        score += 0.8
    if path.get("dereferences"):
        score += 0.5
    if path.get("arithmetic_ops"):
        score += 0.4
    if path.get("checks"):
        score += 0.2
    return score


def path_identity(path: Dict[str, Any]) -> Tuple[str, str, str]:
    return (
        clean_text(path.get("path_type", "")),
        clean_text(path.get("sink_name", "")),
        clean_text(path.get("sink_expr", "")),
    )


def screen_paths(paths: List[Dict[str, Any]], top_k: int = 3) -> List[Dict[str, Any]]:
    if not paths:
        return []

    candidates = [p for p in paths if isinstance(p, dict)]
    stronger = [p for p in candidates if str(p.get("risk_level", "")).lower() != "low_or_context"]
    if stronger:
        candidates = stronger

    deduped: List[Dict[str, Any]] = []
    seen = set()
    for path in sorted(candidates, key=path_score, reverse=True):
        ident = path_identity(path)
        if ident in seen:
            continue
        seen.add(ident)
        deduped.append(path)
        if top_k > 0 and len(deduped) >= top_k:
            break
    return deduped


def path_to_screened_text(path: Dict[str, Any], max_chars: int = 320) -> str:
    parts = [
        f"type: {clean_text(path.get('path_type', 'unknown'))}",
        f"sink: {clean_text(path.get('sink_name') or path.get('sink_expr') or 'unknown')}",
        f"sink_expr: {clean_text(path.get('sink_expr', ''))}",
        f"sink_vars: {short_list(path.get('sink_vars'), limit=5)}",
        f"risk: {clean_text(path.get('risk_level', 'unknown'))}",
        f"checks: {expr_list(path.get('checks'), 'condition', limit=2)}",
        f"arithmetic: {expr_list(path.get('arithmetic_ops'), 'expr', limit=2)}",
        f"dereference: {expr_list(path.get('dereferences'), 'expr', limit=2)}",
    ]
    text = " ; ".join(parts)
    return clean_text(text, max_chars=max_chars)


def paths_to_block(paths: List[Dict[str, Any]], max_chars: int) -> str:
    if not paths:
        return ""
    per_path_chars = max(160, max_chars // max(1, len(paths)))
    chunks = [path_to_screened_text(path, max_chars=per_path_chars) for path in paths]
    return "<VUL_PATH_SCREENED> " + " || ".join(chunks) + " </VUL_PATH_SCREENED>"


def desc_to_block(desc: str, max_chars: int) -> str:
    desc = clean_text(desc, max_chars=max_chars)
    if not has_real_security_desc(desc):
        return ""
    return "<SECURITY_DESC_SCREENED> " + desc + " </SECURITY_DESC_SCREENED>"


def code_to_block(raw_row: Dict[str, Any], code_key: str, max_code_chars: int) -> str:
    return "<CODE> " + clean_text(raw_row.get(code_key, ""), max_chars=max_code_chars) + " </CODE>"


def make_variant(
    raw_row: Dict[str, Any],
    experiment_name: str,
    input_text: str,
    target_key: str,
    components: Dict[str, Any],
) -> Dict[str, Any]:
    row = {
        "target": raw_row.get(target_key),
        "experiment_target_key": target_key,
        "experiment_name": experiment_name,
        "input_text": input_text,
        "components": components,
    }
    for key in ["idx", "project", "func_name"]:
        if key in raw_row:
            row[key] = raw_row[key]
    return row


def build_variants_for_row(
    raw_row: Dict[str, Any],
    vul_row: Optional[Dict[str, Any]],
    desc_row: Optional[Dict[str, Any]],
    code_key: str,
    target_key: str,
    top_k: int,
    max_code_chars: int,
    max_path_chars: int,
    max_desc_chars: int,
) -> Dict[str, Dict[str, Any]]:
    code_block = code_to_block(raw_row, code_key=code_key, max_code_chars=max_code_chars)
    paths = []
    if vul_row is not None:
        paths = screen_paths(vul_row.get("vul_paths", []), top_k=top_k)
    path_block_topk = paths_to_block(paths, max_chars=max_path_chars)

    desc = ""
    if desc_row is not None:
        desc = clean_text(desc_row.get("security_desc", ""))
    desc_block = desc_to_block(desc, max_chars=max_desc_chars)

    has_path = bool(paths)
    has_desc = bool(desc_block)
    variants: Dict[str, Dict[str, Any]] = {}

    variants["code_only"] = make_variant(
        raw_row,
        "code_only",
        code_block,
        target_key,
        {
            "code": True,
            "vul_path_text": False,
            "security_desc": False,
            "vul_path_is_real": False,
            "security_desc_is_real": False,
            "screened": True,
            "screened_path_count": 0,
            "path_first": False,
        },
    )

    selected_blocks = [code_block]
    if path_block_topk:
        selected_blocks.append(path_block_topk)
    variants[f"code_vul_path_top{top_k}_screened"] = make_variant(
        raw_row,
        f"code_vul_path_top{top_k}_screened",
        "\n".join(selected_blocks),
        target_key,
        {
            "code": True,
            "vul_path_text": has_path,
            "security_desc": False,
            "vul_path_is_real": has_path,
            "security_desc_is_real": False,
            "screened": True,
            "screened_path_count": len(paths),
            "path_first": False,
        },
    )

    selected_blocks = [code_block]
    if desc_block:
        selected_blocks.append(desc_block)
    variants["code_desc_screened"] = make_variant(
        raw_row,
        "code_desc_screened",
        "\n".join(selected_blocks),
        target_key,
        {
            "code": True,
            "vul_path_text": False,
            "security_desc": has_desc,
            "vul_path_is_real": False,
            "security_desc_is_real": has_desc,
            "screened": True,
            "screened_path_count": 0,
            "path_first": False,
        },
    )

    selected_blocks = [code_block]
    if path_block_topk:
        selected_blocks.append(path_block_topk)
    if desc_block:
        selected_blocks.append(desc_block)
    variants[f"code_vul_path_desc_top{top_k}_screened"] = make_variant(
        raw_row,
        f"code_vul_path_desc_top{top_k}_screened",
        "\n".join(selected_blocks),
        target_key,
        {
            "code": True,
            "vul_path_text": has_path,
            "security_desc": has_desc,
            "vul_path_is_real": has_path,
            "security_desc_is_real": has_desc,
            "screened": True,
            "screened_path_count": len(paths),
            "path_first": False,
        },
    )

    selected_blocks = []
    if path_block_topk:
        selected_blocks.append(path_block_topk)
    if desc_block:
        selected_blocks.append(desc_block)
    selected_blocks.append(code_block)
    variants[f"path_desc_code_top{top_k}_screened"] = make_variant(
        raw_row,
        f"path_desc_code_top{top_k}_screened",
        "\n".join(selected_blocks),
        target_key,
        {
            "code": True,
            "vul_path_text": has_path,
            "security_desc": has_desc,
            "vul_path_is_real": has_path,
            "security_desc_is_real": has_desc,
            "screened": True,
            "screened_path_count": len(paths),
            "path_first": has_path or has_desc,
        },
    )

    return variants


def find_split_file(base_dir: str, split: str, suffix: str) -> str:
    candidates = [
        os.path.join(base_dir, f"{split}.{suffix}.jsonl"),
        os.path.join(base_dir, f"{split}.jsonl"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0]


def process_split(args: argparse.Namespace, split: str) -> Dict[str, Any]:
    raw_path = os.path.join(args.data_dir, f"{split}.jsonl")
    vul_path = find_split_file(args.vul_path_dir, split, "vul_paths")
    desc_path = find_split_file(args.desc_dir, split, "desc") if args.desc_dir else ""

    raw_rows = read_jsonl(raw_path)
    vul_map = make_map(read_jsonl(vul_path))
    desc_map = make_map(read_jsonl(desc_path)) if desc_path else {}
    if args.max_samples > 0:
        raw_rows = raw_rows[: args.max_samples]

    variant_rows: Dict[str, List[Dict[str, Any]]] = {}
    real_path_count = 0
    real_desc_count = 0

    for raw_row in tqdm(raw_rows, desc=f"build {split}"):
        key = make_key(raw_row)
        variants = build_variants_for_row(
            raw_row=raw_row,
            vul_row=vul_map.get(key),
            desc_row=desc_map.get(key),
            code_key=args.code_key,
            target_key=args.target_key,
            top_k=args.top_k,
            max_code_chars=args.max_code_chars,
            max_path_chars=args.max_path_chars,
            max_desc_chars=args.max_desc_chars,
        )
        if any(v["components"].get("vul_path_is_real") for v in variants.values()):
            real_path_count += 1
        if any(v["components"].get("security_desc_is_real") for v in variants.values()):
            real_desc_count += 1
        for name, row in variants.items():
            variant_rows.setdefault(name, []).append(row)

    written = {}
    for name, rows in variant_rows.items():
        out_path = os.path.join(args.output_dir, name, f"{split}.jsonl")
        written[name] = write_jsonl(out_path, rows)

    return {
        "split": split,
        "raw_rows": len(raw_rows),
        "real_path_rows": real_path_count,
        "real_desc_rows": real_desc_count,
        "written": written,
        "raw_path": raw_path,
        "vul_path": vul_path,
        "desc_path": desc_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--vul_path_dir", required=True)
    parser.add_argument("--desc_dir", default="")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--code_key", default="func")
    parser.add_argument("--target_key", default="target")
    parser.add_argument("--splits", default="train,valid,test")
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--max_code_chars", type=int, default=2400)
    parser.add_argument("--max_path_chars", type=int, default=900)
    parser.add_argument("--max_desc_chars", type=int, default=450)
    parser.add_argument("--max_samples", type=int, default=-1)
    args = parser.parse_args()

    summaries = [process_split(args, split) for split in split_csv(args.splits)]
    write_json(os.path.join(args.output_dir, "screened_input_summary.json"), {"splits": summaries, "args": vars(args)})
    print(json.dumps({"splits": summaries, "output_dir": args.output_dir}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
