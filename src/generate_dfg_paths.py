#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_dfg_paths.py

Generate two files for each split:
1) outputs/{dataset}/full_dfg/{split}.full_dfg.jsonl
2) outputs/{dataset}/vul_paths/{split}.vul_paths.jsonl

Input JSONL example:
{"func": "...", "func_name": "CVE-xxx.c", "target": 1, "idx": 123, "project": "LibPNG"}

This is a practical C/C++ data-flow and vulnerability-path extractor. It is not
the official GraphCodeBERT extractor.

Example:

python src/generate_dfg_paths.py --input_dir data/lin_et_al --output_dir outputs/lin_et_al --code_key func --max_hops 6 --max_paths_per_sample 5
"""

import argparse
import json
import os
import re
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Tuple, Set

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x


# ============================================================
# Parser loading
# ============================================================

def get_parser(lang: str):
    """
    Load tree-sitter parser.

    V3 prioritizes tree_sitter_c / tree_sitter_cpp because they are stable in
    the current Windows py311 environment. tree_sitter_language_pack is not used
    first because it may hang in some environments.
    """
    try:
        from tree_sitter import Language, Parser
        if lang == "c":
            import tree_sitter_c
            language = Language(tree_sitter_c.language())
        else:
            import tree_sitter_cpp
            language = Language(tree_sitter_cpp.language())
        parser = Parser()
        try:
            parser.language = language
        except Exception:
            parser.set_language(language)
        return parser
    except Exception:
        pass

    try:
        from tree_sitter_languages import get_parser as tsl_get_parser
        return tsl_get_parser(lang)
    except Exception:
        pass

    # Intentionally try language_pack last.
    try:
        from tree_sitter_language_pack import get_parser as lp_get_parser
        return lp_get_parser(lang)
    except Exception:
        pass

    return None


PARSER_C = None
PARSER_CPP = None


def choose_lang(func_name: str, code: str) -> str:
    name = (func_name or "").lower()
    if name.endswith((".cpp", ".cc", ".cxx", ".hpp", ".hxx", ".hh")):
        return "cpp"
    if re.search(r"\bstd::|template\s*<|class\s+\w+|namespace\s+\w+|new\s+\w+|delete\s+", code):
        return "cpp"
    return "c"


def parse_code(code: str, lang: str):
    global PARSER_C, PARSER_CPP
    if lang == "cpp":
        if PARSER_CPP is None:
            PARSER_CPP = get_parser("cpp")
        parser = PARSER_CPP
    else:
        if PARSER_C is None:
            PARSER_C = get_parser("c")
        parser = PARSER_C

    if parser is None:
        return None
    try:
        return parser.parse(code.encode("utf-8", errors="ignore"))
    except Exception:
        return None


# ============================================================
# Tree helpers
# ============================================================

def node_text(node, code_bytes: bytes) -> str:
    try:
        return code_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="ignore")
    except Exception:
        return ""


def node_line(node) -> int:
    try:
        return node.start_point[0] + 1
    except Exception:
        return -1


def walk(node):
    if node is None:
        return
    yield node
    for child in getattr(node, "children", []):
        yield from walk(child)


def get_child_by_field(node, field_name: str):
    try:
        return node.child_by_field_name(field_name)
    except Exception:
        return None


# V3: exclude type_identifier / namespace_identifier to reduce type-name noise.
IDENT_TYPES = {"identifier", "field_identifier"}

C_KEYWORDS = {
    "if", "else", "for", "while", "do", "switch", "case", "break", "continue",
    "return", "sizeof", "typedef", "struct", "union", "enum", "static", "const",
    "volatile", "unsigned", "signed", "int", "char", "short", "long", "float",
    "double", "void", "bool", "true", "false", "NULL", "nullptr", "class",
    "public", "private", "protected", "template", "typename", "namespace",
    "using", "include", "define", "ifdef", "ifndef", "endif", "elif",
}


def clean_var(v: str) -> str:
    return re.sub(r"\s+", " ", str(v).strip())


def is_literal_text(s: str) -> bool:
    s = str(s).strip()
    if not s:
        return True
    if re.fullmatch(r"[-+]?\d+(\.\d+)?([uUlLfF]+)?", s):
        return True
    if s.startswith('"') and s.endswith('"'):
        return True
    if s.startswith("'") and s.endswith("'"):
        return True
    return False


def unique_keep_order(xs: List[str]) -> List[str]:
    seen, out = set(), []
    for x in xs:
        x = str(x).strip()
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


def collect_identifiers(node, code_bytes: bytes, skip_function_name_in_calls: bool = True) -> List[str]:
    if node is None:
        return []
    vars_found = []

    def rec(n):
        if n is None:
            return
        if skip_function_name_in_calls and getattr(n, "type", "") == "call_expression":
            args = get_child_by_field(n, "arguments")
            if args is not None:
                rec(args)
            return
        if getattr(n, "type", "") in IDENT_TYPES:
            txt = clean_var(node_text(n, code_bytes))
            if txt and txt not in C_KEYWORDS and not is_literal_text(txt):
                vars_found.append(txt)
            return
        for ch in getattr(n, "children", []):
            rec(ch)

    rec(node)
    return unique_keep_order(vars_found)


def collect_full_expr_identifiers(expr_text: str) -> List[str]:
    tokens = re.findall(r"\b[A-Za-z_]\w*\b", str(expr_text))
    return unique_keep_order([t for t in tokens if t not in C_KEYWORDS])


def extract_lhs_variables(lhs_node, code_bytes: bytes) -> List[str]:
    if lhs_node is None:
        return []
    txt = node_text(lhs_node, code_bytes).strip()
    ids = collect_identifiers(lhs_node, code_bytes, skip_function_name_in_calls=True)
    if ids:
        return ids[:1]
    return collect_full_expr_identifiers(txt)[:1]


def get_call_name(call_node, code_bytes: bytes) -> str:
    fn = get_child_by_field(call_node, "function")
    if fn is None:
        return ""
    txt = node_text(fn, code_bytes).strip().replace(" ", "")
    if "::" in txt:
        txt = txt.split("::")[-1]
    if "." in txt:
        txt = txt.split(".")[-1]
    if "->" in txt:
        txt = txt.split("->")[-1]
    return txt


def get_call_args(call_node, code_bytes: bytes) -> List[Tuple[str, List[str]]]:
    args_node = get_child_by_field(call_node, "arguments")
    if args_node is None:
        return []
    args = []
    for ch in args_node.children:
        if ch.type in {"(", ")", ","}:
            continue
        txt = node_text(ch, code_bytes).strip()
        if txt:
            args.append((txt, collect_identifiers(ch, code_bytes, skip_function_name_in_calls=True)))
    return args


# ============================================================
# Vulnerability rules
# ============================================================

DANGEROUS_COPY_FUNCS = {"strcpy", "strcat", "sprintf", "vsprintf", "gets", "scanf", "sscanf", "fscanf"}
BOUNDED_BUT_RISKY_COPY_FUNCS = {"strncpy", "strncat", "snprintf", "vsnprintf"}
MEMORY_COPY_FUNCS = {"memcpy", "memmove", "memset", "bcopy", "read", "recv", "recvfrom", "fread"}
ALLOC_FUNCS = {"malloc", "calloc", "realloc", "alloca", "operator new", "new"}
DEALLOC_FUNCS = {"free", "delete", "operator delete"}
FORMAT_FUNCS = {"printf", "fprintf", "sprintf", "snprintf", "vprintf", "vfprintf", "vsprintf", "vsnprintf", "syslog"}
COMMAND_FUNCS = {"system", "popen", "execl", "execle", "execlp", "execv", "execve", "execvp", "WinExec", "ShellExecute", "ShellExecuteA", "ShellExecuteW"}
PATH_FUNCS = {"open", "fopen", "freopen", "creat", "remove", "unlink", "rename", "mkdir", "rmdir", "opendir", "stat", "lstat", "access", "chmod", "chown"}
RESOURCE_OPEN_FUNCS = {"fopen", "freopen", "open", "socket", "accept", "opendir"}
RESOURCE_CLOSE_FUNCS = {"fclose", "close", "closesocket", "closedir"}
INPUT_FUNCS = {"getenv", "gets", "fgets", "scanf", "sscanf", "fscanf", "read", "recv", "recvfrom", "fread"}
INTEGER_RISK_FUNCS = {"malloc", "calloc", "realloc", "memcpy", "memmove", "memset", "read", "recv", "fread", "strncpy", "snprintf"}
SANITIZER_HINTS = {"validate", "sanitize", "escape", "check", "is_valid", "safe", "realpath", "canonicalize", "strnlen", "bounds", "limit"}
ARITH_OPS = {"+", "-", "*", "/", "%", "<<", ">>"}

# V3: variable-name hints for boundary/check-only paths.
SECURITY_VAR_HINTS = {
    "len", "length", "size", "limit", "max", "min", "count", "cnt", "idx", "index",
    "offset", "pos", "height", "width", "row", "col", "chunk", "alloc", "buffer",
    "buf", "cap", "capacity", "num", "total", "bound", "boundary",
}


def is_security_related_name(name: str) -> bool:
    name = str(name).lower()
    return any(h in name for h in SECURITY_VAR_HINTS)


def is_simple_or_std_callee(callee_expr: str) -> bool:
    """
    Return True for direct library-like function calls:
        memcpy(...)
        ::memcpy(...)
        std::memcpy(...)

    Return False for member/function-pointer calls:
        ctx->ops->remove(...)
        obj.remove(...)
        a.b.remove(...)

    This prevents many false positives where a project-defined method happens
    to have the same name as a risky C library function.
    """
    callee_expr = str(callee_expr or "").strip().replace(" ", "")

    if not callee_expr:
        return True

    if "->" in callee_expr or "." in callee_expr:
        return False

    if re.fullmatch(r"[A-Za-z_]\w*", callee_expr):
        return True

    if re.fullmatch(r"::[A-Za-z_]\w*", callee_expr):
        return True

    if re.fullmatch(r"(std|__gnu_cxx|boost)::[A-Za-z_]\w*", callee_expr):
        return True

    return False


def is_constant_small_array_access(expr: str, max_idx: int = 3) -> bool:
    """
    Filter low-value layout accesses such as:
        buf[0], arr[1], slotbuf[2]

    These fixed small constant accesses are very common in safe code and caused
    many false high-risk OOB evidence paths.
    """
    expr = str(expr or "").strip().replace(" ", "")

    m = re.fullmatch(r"[A-Za-z_]\w*(?:->\w+|\.\w+)*\[(0x[0-9a-fA-F]+|\d+)\]", expr)
    if not m:
        return False

    try:
        idx = int(m.group(1), 0)
    except Exception:
        return False

    return 0 <= idx <= max_idx


def has_variable_array_index(expr: str) -> bool:
    """
    Return True if array index contains a variable-like token:
        arr[i], arr[len-1], arr[offset + k]

    Return False for constant index:
        arr[0], arr[16]
    """
    expr = str(expr or "")
    m = re.search(r"\[(.*?)\]", expr)

    if not m:
        return False

    index_text = m.group(1)
    vars_ = collect_full_expr_identifiers(index_text)
    vars_ = [v for v in vars_ if v not in {"sizeof"}]

    return len(vars_) > 0


def classify_call_sink(
    call_name: str,
    args: List[Tuple[str, List[str]]],
    callee_expr: str = "",
) -> Optional[str]:
    if not call_name:
        return None

    # V3: only treat library-like direct calls as risky library sinks.
    # This avoids false positives such as ctx->ops->remove(ctx, e).
    if not is_simple_or_std_callee(callee_expr or call_name):
        return None

    if call_name in DANGEROUS_COPY_FUNCS:
        return "buffer_boundary"
    if call_name in BOUNDED_BUT_RISKY_COPY_FUNCS:
        return "buffer_boundary_bounded"
    if call_name in MEMORY_COPY_FUNCS:
        return "memory_boundary"
    if call_name in FORMAT_FUNCS:
        return "format_string"
    if call_name in COMMAND_FUNCS:
        return "command_injection"
    if call_name in PATH_FUNCS:
        return "path_traversal_or_file_access"
    if call_name in INTEGER_RISK_FUNCS:
        return "integer_to_memory"
    return None


def likely_format_arg_index(call_name: str) -> int:
    if call_name in {"printf", "vprintf"}:
        return 0
    if call_name in {"fprintf", "sprintf", "vfprintf", "vsprintf", "syslog"}:
        return 1
    if call_name in {"snprintf", "vsnprintf"}:
        return 2
    return 0


def sink_argument_variables(call_name: str, vuln_type: str, args: List[Tuple[str, List[str]]]) -> List[str]:
    selected = []
    if vuln_type == "format_string":
        idx = likely_format_arg_index(call_name)
        if idx < len(args):
            fmt_text, fmt_vars = args[idx]
            if is_literal_text(fmt_text):
                for _, vs in args[idx + 1:]:
                    selected.extend(vs)
            else:
                selected.extend(fmt_vars)
        return unique_keep_order(selected)

    for _, vs in args:
        selected.extend(vs)
    return unique_keep_order(selected)


# ============================================================
# Full DFG extraction
# ============================================================

def make_edge(src: str, dst: str, kind: str, line: int, evidence: str) -> Dict[str, Any]:
    return {"src": clean_var(src), "dst": clean_var(dst), "kind": kind, "line": line, "evidence": str(evidence).strip()[:300]}


def find_function_parameters(root, code_bytes: bytes) -> List[str]:
    params = []
    for n in walk(root):
        if n.type in {"function_definition", "function_declarator"}:
            params_node = get_child_by_field(n, "parameters")
            if params_node is not None:
                for p in walk(params_node):
                    if p.type == "parameter_declaration":
                        ids = collect_identifiers(p, code_bytes)
                        if ids:
                            params.append(ids[-1])
                break
    return unique_keep_order(params)


def is_likely_security_check(cond: str) -> bool:
    c = str(cond).lower()
    if any(op in c for op in ["<", "<=", ">", ">=", "!=", "=="]):
        return True
    if any(x in c for x in ["sizeof", "strlen", "strnlen", "null", "nullptr", "limit", "max", "min", "len", "length", "size", "bound", "index", "idx", "offset"]):
        return True
    if any(h in c for h in SANITIZER_HINTS):
        return True
    return False


def extract_checks(root, code_bytes: bytes) -> List[Dict[str, Any]]:
    checks = []
    for n in walk(root):
        if n.type in {"if_statement", "while_statement", "for_statement"}:
            cond = get_child_by_field(n, "condition")
            if cond is None:
                txt = node_text(n, code_bytes)
                m = re.search(r"\((.*?)\)", txt, flags=re.S)
                cond_txt = m.group(1).strip() if m else txt[:200]
                vars_ = collect_full_expr_identifiers(cond_txt)
            else:
                cond_txt = node_text(cond, code_bytes).strip()
                vars_ = collect_identifiers(cond, code_bytes)
            checks.append({"line": node_line(n), "kind": n.type.replace("_statement", ""), "condition": cond_txt[:300], "vars": vars_, "is_likely_security_check": is_likely_security_check(cond_txt)})
    return checks


def extract_arithmetic_ops(root, code_bytes: bytes) -> List[Dict[str, Any]]:
    ops = []
    for n in walk(root):
        if n.type == "binary_expression":
            txt = node_text(n, code_bytes).strip()
            op_hit = None
            for op in sorted(ARITH_OPS, key=len, reverse=True):
                if op in txt:
                    op_hit = op
                    break
            if op_hit:
                ops.append({"line": node_line(n), "op": op_hit, "expr": txt[:300], "vars": collect_identifiers(n, code_bytes)})
    return ops


def extract_dereferences(root, code_bytes: bytes) -> List[Dict[str, Any]]:
    derefs = []
    for n in walk(root):
        if n.type in {"subscript_expression", "pointer_expression", "field_expression"}:
            txt = node_text(n, code_bytes).strip()
            vars_ = collect_identifiers(n, code_bytes)
            if not vars_:
                continue
            kind = "array_subscript" if n.type == "subscript_expression" else ("pointer_dereference" if n.type == "pointer_expression" else "field_access")
            derefs.append({"line": node_line(n), "kind": kind, "expr": txt[:300], "vars": vars_})
    return derefs


def extract_calls(root, code_bytes: bytes) -> List[Dict[str, Any]]:
    calls = []
    for n in walk(root):
        if n.type == "call_expression":
            name = get_call_name(n, code_bytes)
            args = get_call_args(n, code_bytes)

            fn_node = get_child_by_field(n, "function")
            callee_expr = node_text(fn_node, code_bytes).strip() if fn_node is not None else name

            calls.append({
                "line": node_line(n),
                "name": name,
                "callee_expr": callee_expr,
                "expr": node_text(n, code_bytes).strip()[:400],
                "args": [{"text": a, "vars": vs} for a, vs in args],
            })
    return calls


def contains_call_name(node, code_bytes: bytes, names: Set[str]) -> Optional[str]:
    if node is None:
        return None
    for n in walk(node):
        if n.type == "call_expression":
            call_name = get_call_name(n, code_bytes)
            if call_name in names:
                return call_name
    txt = node_text(node, code_bytes)
    for name in names:
        if re.search(r"\b" + re.escape(name) + r"\s*\(", txt):
            return name
    return None


def dedup_edges(edges: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen, out = set(), []
    for e in edges:
        key = (e["src"], e["dst"], e["kind"], e["line"], e["evidence"])
        if key not in seen:
            seen.add(key)
            out.append(e)
    return out


def extract_full_dfg(root, code_bytes: bytes) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    edges, events = [], []

    for p in find_function_parameters(root, code_bytes):
        events.append({"line": 1, "kind": "function_parameter", "var": p, "expr": p})

    for n in walk(root):
        txt = node_text(n, code_bytes).strip()
        line = node_line(n)

        if n.type == "init_declarator":
            lhs_node = get_child_by_field(n, "declarator")
            rhs_node = get_child_by_field(n, "value")
            lhs_vars = extract_lhs_variables(lhs_node, code_bytes)
            rhs_vars = collect_identifiers(rhs_node, code_bytes) if rhs_node is not None else []
            alloc_name = contains_call_name(rhs_node, code_bytes, ALLOC_FUNCS) if rhs_node is not None else None
            res_name = contains_call_name(rhs_node, code_bytes, RESOURCE_OPEN_FUNCS) if rhs_node is not None else None
            for lhs in lhs_vars:
                if alloc_name:
                    events.append({"line": line, "kind": "allocation", "var": lhs, "call": alloc_name, "expr": txt[:300]})
                    edges.append(make_edge(f"call:{alloc_name}@{line}", lhs, "alloc_init", line, txt))
                if res_name:
                    events.append({"line": line, "kind": "resource_open", "var": lhs, "call": res_name, "expr": txt[:300]})
                    edges.append(make_edge(f"call:{res_name}@{line}", lhs, "resource_open_init", line, txt))
                for rv in rhs_vars:
                    if rv != lhs:
                        edges.append(make_edge(rv, lhs, "init", line, txt))

        elif n.type == "assignment_expression":
            lhs_node = get_child_by_field(n, "left")
            rhs_node = get_child_by_field(n, "right")
            lhs_vars = extract_lhs_variables(lhs_node, code_bytes)
            rhs_vars = collect_identifiers(rhs_node, code_bytes) if rhs_node is not None else []
            alloc_name = contains_call_name(rhs_node, code_bytes, ALLOC_FUNCS) if rhs_node is not None else None
            res_name = contains_call_name(rhs_node, code_bytes, RESOURCE_OPEN_FUNCS) if rhs_node is not None else None
            for lv in lhs_vars:
                if alloc_name:
                    events.append({"line": line, "kind": "allocation", "var": lv, "call": alloc_name, "expr": txt[:300]})
                    edges.append(make_edge(f"call:{alloc_name}@{line}", lv, "alloc_assign", line, txt))
                if res_name:
                    events.append({"line": line, "kind": "resource_open", "var": lv, "call": res_name, "expr": txt[:300]})
                    edges.append(make_edge(f"call:{res_name}@{line}", lv, "resource_open_assign", line, txt))
                for rv in rhs_vars:
                    if rv != lv:
                        edges.append(make_edge(rv, lv, "assign", line, txt))

        elif n.type == "call_expression":
            call_name = get_call_name(n, code_bytes)
            args = get_call_args(n, code_bytes)
            if call_name in DEALLOC_FUNCS or call_name in RESOURCE_CLOSE_FUNCS:
                for _, vs in args:
                    for v in vs:
                        kind = "deallocation" if call_name in DEALLOC_FUNCS else "resource_close"
                        events.append({"line": line, "kind": kind, "var": v, "call": call_name, "expr": txt[:300]})
            if call_name in INPUT_FUNCS:
                events.append({"line": line, "kind": "external_input_call", "var": f"call:{call_name}@{line}", "call": call_name, "expr": txt[:300]})
            for i, (_, vs) in enumerate(args, start=1):
                dst = f"call:{call_name}:arg{i}@{line}"
                for v in vs:
                    edges.append(make_edge(v, dst, "call_arg", line, txt))

        elif n.type in {"if_statement", "while_statement", "for_statement"}:
            cond = get_child_by_field(n, "condition")
            if cond is not None:
                cond_txt = node_text(cond, code_bytes).strip()
                vars_ = collect_identifiers(cond, code_bytes)
            else:
                cond_txt = txt[:200]
                vars_ = collect_full_expr_identifiers(cond_txt)
            dst = f"{n.type.replace('_statement', '')}@{line}"
            for v in vars_:
                edges.append(make_edge(v, dst, "condition", line, cond_txt))

        elif n.type == "return_statement":
            for v in collect_identifiers(n, code_bytes):
                edges.append(make_edge(v, f"return@{line}", "return", line, txt))

        elif n.type == "update_expression":
            for v in collect_identifiers(n, code_bytes):
                edges.append(make_edge(v, v, "update", line, txt))

    return dedup_edges(edges), events


# ============================================================
# Vulnerability-oriented path extraction
# ============================================================

def build_reverse_graph(edges: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    rg = defaultdict(list)
    for e in edges:
        rg[e["dst"]].append(e)
    return rg


def backward_trace(start_vars: List[str], reverse_graph: Dict[str, List[Dict[str, Any]]], max_hops: int = 6) -> List[Dict[str, Any]]:
    result_edges, visited_nodes, visited_edges = [], set(), set()
    q = deque()
    for v in start_vars:
        if v:
            q.append((v, 0))
            visited_nodes.add(v)
    while q:
        cur, depth = q.popleft()
        if depth >= max_hops:
            continue
        for e in reverse_graph.get(cur, []):
            key = (e["src"], e["dst"], e["kind"], e["line"], e["evidence"])
            if key in visited_edges:
                continue
            visited_edges.add(key)
            result_edges.append(e)
            src = e["src"]
            if src not in visited_nodes and not src.startswith("call:"):
                visited_nodes.add(src)
                q.append((src, depth + 1))
    return result_edges


def vars_in_edges(edges: List[Dict[str, Any]]) -> Set[str]:
    s = set()
    for e in edges:
        if not e["src"].startswith("call:"):
            s.add(e["src"])
        if not e["dst"].startswith("call:") and "@" not in e["dst"]:
            s.add(e["dst"])
    return s


def related_checks(checks: List[Dict[str, Any]], related_vars: Set[str], sink_line: int, window: int = 12) -> List[Dict[str, Any]]:
    """
    V3: prefer variable-overlapping checks; allow only very-near security-looking checks.
    """
    out = []
    for c in checks:
        c_line = c.get("line", -1)
        if c_line > sink_line:
            continue
        if sink_line - c_line > window:
            continue
        c_vars = set(c.get("vars", []))
        if related_vars.intersection(c_vars):
            out.append(c)
        elif c.get("is_likely_security_check") and sink_line - c_line <= 3:
            out.append(c)
    return out


def related_events(events: List[Dict[str, Any]], related_vars: Set[str], sink_line: int) -> List[Dict[str, Any]]:
    return [ev for ev in events if ev.get("line", 10**9) <= sink_line and ev.get("var") in related_vars]


def related_arithmetic(ariths: List[Dict[str, Any]], related_vars: Set[str], sink_line: int, window: int = 30) -> List[Dict[str, Any]]:
    return [a for a in ariths if a.get("line", 10**9) <= sink_line and sink_line - a.get("line", 10**9) <= window and related_vars.intersection(set(a.get("vars", [])))]


def related_derefs(derefs: List[Dict[str, Any]], related_vars: Set[str], sink_line: int, window: int = 30) -> List[Dict[str, Any]]:
    return [d for d in derefs if d.get("line", 10**9) <= sink_line and abs(sink_line - d.get("line", 10**9)) <= window and related_vars.intersection(set(d.get("vars", [])))]


def infer_risk_level(vuln_type: str, sink: Dict[str, Any], checks: List[Dict[str, Any]]) -> str:
    if vuln_type == "format_string":
        call_name = sink["name"]
        args = sink.get("args", [])
        idx = likely_format_arg_index(call_name)
        if idx < len(args):
            fmt = args[idx]["text"]
            if is_literal_text(fmt):
                return "low_or_context"
            return "high"
    if checks:
        if any(c.get("is_likely_security_check") for c in checks):
            return "checked_or_lower_risk"
        return "has_condition_context"
    if vuln_type in {"buffer_boundary", "memory_boundary", "command_injection", "path_traversal_or_file_access", "use_after_free", "double_free", "out_of_bounds_access"}:
        return "high"
    return "medium"


def extract_check_arithmetic_paths(checks: List[Dict[str, Any]], ariths: List[Dict[str, Any]], max_paths: int = 8) -> List[Dict[str, Any]]:
    """
    V3 fallback: extract validation/check-only paths for length/limit/size/integer boundary logic.
    """
    paths = []
    for c in checks:
        c_vars = c.get("vars", [])
        if not c.get("is_likely_security_check", False):
            continue
        if not any(is_security_related_name(v) for v in c_vars):
            continue
        related_ariths = []
        for a in ariths:
            if set(c_vars).intersection(set(a.get("vars", []))):
                related_ariths.append(a)
        paths.append({
            "path_type": "integer_boundary_check_or_validation_logic",
            "sink_kind": "condition",
            "sink_name": "boundary_check",
            "sink_line": c.get("line"),
            "sink_expr": c.get("condition", ""),
            "sink_vars": c_vars,
            "flow_edges": [],
            "checks": [c],
            "events": [],
            "arithmetic_ops": related_ariths[:5],
            "dereferences": [],
            "risk_level": "validation_related",
        })
        if len(paths) >= max_paths:
            break
    return paths


def dedup_paths(paths: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen, out = set(), []
    for p in paths:
        key = (p.get("path_type"), p.get("sink_kind"), p.get("sink_name"), p.get("sink_line"), p.get("sink_expr"))
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out



def evidence_path_score(p: Dict[str, Any]) -> float:
    """
    Rank vulnerability-oriented evidence paths.
    Higher score means more useful for model input.
    """
    path_type = str(p.get("path_type", ""))
    risk = str(p.get("risk_level", ""))

    type_score = {
        "use_after_free": 10.0,
        "double_free": 9.8,
        "buffer_boundary": 9.5,
        "memory_boundary": 9.2,
        "command_injection": 9.0,
        "format_string": 8.8,
        "integer_to_memory": 8.2,
        "path_traversal_or_file_access": 7.8,
        "out_of_bounds_access": 7.0,
        "integer_boundary_check_or_validation_logic": 6.5,
        "buffer_boundary_bounded": 6.2,
        "null_pointer_dereference_or_pointer_use": 5.5,
        "memory_leak_or_lifetime_error": 3.0,
        "resource_leak": 2.8,
    }.get(path_type, 1.0)

    risk_score = {
        "high": 3.0,
        "medium": 1.8,
        "validation_related": 1.5,
        "checked_or_lower_risk": 1.2,
        "has_condition_context": 1.0,
        "low_or_context": 0.2,
    }.get(risk, 0.5)

    score = type_score + risk_score

    if p.get("flow_edges"):
        score += 0.7
    if p.get("checks"):
        score += 0.3
    if p.get("arithmetic_ops"):
        score += 0.3
    if p.get("events"):
        # function_parameter events are less important than allocation/free events
        important_events = [
            ev for ev in p.get("events", [])
            if ev.get("kind") not in {"function_parameter"}
        ]
        if important_events:
            score += 0.5
    if p.get("dereferences"):
        score += 0.2

    return score


def finalize_vul_paths(paths: List[Dict[str, Any]], max_paths_per_sample: int = 5) -> List[Dict[str, Any]]:
    """
    Deduplicate, rank and keep top-k paths to reduce text noise.
    """
    paths = dedup_paths(paths)

    # Drop explicitly low-context evidence if there are stronger paths.
    stronger = [p for p in paths if p.get("risk_level") not in {"low_or_context"}]
    if stronger:
        paths = stronger

    paths = sorted(paths, key=evidence_path_score, reverse=True)

    if max_paths_per_sample and max_paths_per_sample > 0:
        paths = paths[:max_paths_per_sample]

    return paths


def extract_vul_paths(edges, calls, checks, events, ariths, derefs, max_hops: int = 6, max_paths_per_sample: int = 5, include_pointer_paths: bool = False, include_lifetime_leak_paths: bool = False) -> List[Dict[str, Any]]:
    reverse_graph = build_reverse_graph(edges)
    paths = []

    # 1) Sensitive calls.
    for call in calls:
        call_name = call["name"]
        args_pairs = [(a["text"], a["vars"]) for a in call.get("args", [])]
        vuln_type = classify_call_sink(call_name, args_pairs, call.get("callee_expr", call_name))
        if vuln_type is None:
            continue
        start_vars = sink_argument_variables(call_name, vuln_type, args_pairs)
        flow_edges = backward_trace(start_vars, reverse_graph, max_hops=max_hops)
        rel_vars = set(start_vars) | vars_in_edges(flow_edges)
        rel_checks = related_checks(checks, rel_vars, call["line"])
        item = {
            "path_type": vuln_type,
            "sink_kind": "call",
            "sink_name": call_name,
            "sink_line": call["line"],
            "sink_expr": call["expr"],
            "sink_vars": start_vars,
            "flow_edges": flow_edges,
            "checks": rel_checks,
            "events": related_events(events, rel_vars, call["line"]),
            "arithmetic_ops": related_arithmetic(ariths, rel_vars, call["line"]),
            "dereferences": related_derefs(derefs, rel_vars, call["line"]),
            "risk_level": infer_risk_level(vuln_type, call, rel_checks),
        }
        if start_vars or rel_checks:
            paths.append(item)

    # 2) Array subscript.
    for d in derefs:
        if d["kind"] != "array_subscript":
            continue

        # V3: skip low-value fixed small constant accesses such as buf[0].
        # Keep variable-index accesses such as buf[i], buf[len-1], arr[offset].
        if is_constant_small_array_access(d.get("expr", "")):
            continue

        if not has_variable_array_index(d.get("expr", "")):
            continue

        start_vars = d.get("vars", [])
        flow_edges = backward_trace(start_vars, reverse_graph, max_hops=max_hops)
        rel_vars = set(start_vars) | vars_in_edges(flow_edges)
        rel_checks = related_checks(checks, rel_vars, d["line"])
        paths.append({
            "path_type": "out_of_bounds_access",
            "sink_kind": "array_subscript",
            "sink_name": "array_subscript",
            "sink_line": d["line"],
            "sink_expr": d["expr"],
            "sink_vars": start_vars,
            "flow_edges": flow_edges,
            "checks": rel_checks,
            "events": related_events(events, rel_vars, d["line"]),
            "arithmetic_ops": related_arithmetic(ariths, rel_vars, d["line"]),
            "dereferences": [d],
            "risk_level": "checked_or_lower_risk" if rel_checks else "medium",
        })

    # 3) Pointer dereference.
    # V3: generic pointer-use evidence is very noisy in C/C++.
    # It is disabled by default and can be enabled with --include_pointer_paths.
    if include_pointer_paths:
        for d in derefs:
            if d["kind"] != "pointer_dereference":
                continue
            start_vars = d.get("vars", [])
            flow_edges = backward_trace(start_vars, reverse_graph, max_hops=max_hops)
            rel_vars = set(start_vars) | vars_in_edges(flow_edges)
            rel_checks = related_checks(checks, rel_vars, d["line"])
            paths.append({
                "path_type": "null_pointer_dereference_or_pointer_use",
                "sink_kind": "pointer_dereference",
                "sink_name": "pointer_dereference",
                "sink_line": d["line"],
                "sink_expr": d["expr"],
                "sink_vars": start_vars,
                "flow_edges": flow_edges,
                "checks": rel_checks,
                "events": related_events(events, rel_vars, d["line"]),
                "arithmetic_ops": related_arithmetic(ariths, rel_vars, d["line"]),
                "dereferences": [d],
                "risk_level": "checked_or_lower_risk" if rel_checks else "medium",
            })

    # 4) UAF and double free.
    frees_by_var, uses_by_var = defaultdict(list), defaultdict(list)
    for ev in events:
        if ev["kind"] == "deallocation":
            frees_by_var[ev["var"]].append(ev)
    for d in derefs:
        for v in d.get("vars", []):
            uses_by_var[v].append(d)

    for var, frees in frees_by_var.items():
        frees = sorted(frees, key=lambda x: x["line"])
        if len(frees) >= 2:
            for i in range(1, len(frees)):
                paths.append({
                    "path_type": "double_free",
                    "sink_kind": "deallocation",
                    "sink_name": frees[i].get("call", "free"),
                    "sink_line": frees[i]["line"],
                    "sink_expr": frees[i]["expr"],
                    "sink_vars": [var],
                    "flow_edges": backward_trace([var], reverse_graph, max_hops=max_hops),
                    "checks": related_checks(checks, {var}, frees[i]["line"]),
                    "events": [frees[i - 1], frees[i]],
                    "arithmetic_ops": [],
                    "dereferences": [],
                    "risk_level": "high",
                })
        first_free = frees[0]
        for use in uses_by_var.get(var, []):
            if use["line"] > first_free["line"]:
                paths.append({
                    "path_type": "use_after_free",
                    "sink_kind": use["kind"],
                    "sink_name": use["kind"],
                    "sink_line": use["line"],
                    "sink_expr": use["expr"],
                    "sink_vars": [var],
                    "flow_edges": backward_trace([var], reverse_graph, max_hops=max_hops),
                    "checks": related_checks(checks, {var}, use["line"]),
                    "events": [first_free],
                    "arithmetic_ops": [],
                    "dereferences": [use],
                    "risk_level": "high",
                })

    # 5) Resource/memory leak rough candidate.
    # V3: disabled by default because function-local leak heuristics are noisy.
    if include_lifetime_leak_paths:
        closed_vars = {ev["var"] for ev in events if ev["kind"] in {"deallocation", "resource_close"}}
        for ev in events:
            if ev["kind"] in {"allocation", "resource_open"}:
                var = ev.get("var")
                if var and var not in closed_vars:
                    leak_type = "memory_leak_or_lifetime_error" if ev["kind"] == "allocation" else "resource_leak"
                    paths.append({
                        "path_type": leak_type,
                        "sink_kind": "function_exit",
                        "sink_name": "missing_release",
                        "sink_line": ev["line"],
                        "sink_expr": ev["expr"],
                        "sink_vars": [var],
                        "flow_edges": backward_trace([var], reverse_graph, max_hops=max_hops),
                        "checks": related_checks(checks, {var}, ev["line"]),
                        "events": [ev],
                        "arithmetic_ops": [],
                        "dereferences": [],
                        "risk_level": "medium",
                    })

    # 6) V3 fallback: boundary check / validation logic if no explicit sink path exists.
    if not paths:
        paths.extend(extract_check_arithmetic_paths(checks=checks, ariths=ariths, max_paths=8))

    return finalize_vul_paths(paths, max_paths_per_sample=max_paths_per_sample)


# ============================================================
# Serialization
# ============================================================

def edges_to_text(edges: List[Dict[str, Any]], max_edges: int = 120) -> str:
    if not edges:
        return "<DFG> none </DFG>"
    parts = [f"{e['src']} -> {e['dst']} [{e['kind']}@L{e['line']}]" for e in edges[:max_edges]]
    if len(edges) > max_edges:
        parts.append(f"... {len(edges) - max_edges} more edges")
    return "<DFG> " + " ; ".join(parts) + " </DFG>"


def vul_paths_to_text(paths: List[Dict[str, Any]], max_paths: int = 20, max_edges_each: int = 20) -> str:
    if not paths:
        return "<VUL_PATH> none </VUL_PATH>"
    chunks = []
    for p in paths[:max_paths]:
        flow = " ; ".join(f"{e['src']} -> {e['dst']}" for e in p.get("flow_edges", [])[:max_edges_each]) or "none"
        checks = " | ".join(c.get("condition", "") for c in p.get("checks", [])[:5]) or "none"
        important_events = [ev for ev in p.get("events", []) if ev.get("kind") != "function_parameter"]
        events = " | ".join(f"{ev.get('kind')}:{ev.get('var')}@L{ev.get('line')}" for ev in important_events[:5]) or "none"
        arith = " | ".join(a.get("expr", "") for a in p.get("arithmetic_ops", [])[:5]) or "none"
        deref = " | ".join(d.get("expr", "") for d in p.get("dereferences", [])[:5]) or "none"
        sink_vars = p.get("sink_vars", [])
        if not isinstance(sink_vars, list):
            sink_vars = []
        chunks.append(
            f"<VUL_PATH> type: {p.get('path_type')} ; sink: {p.get('sink_expr')} ; "
            f"sink_line: L{p.get('sink_line')} ; sink_vars: {', '.join(sink_vars) or 'none'} ; "
            f"flow: {flow} ; checks: {checks} ; events: {events} ; arithmetic: {arith} ; "
            f"dereference: {deref} ; risk: {p.get('risk_level')} </VUL_PATH>"
        )
    if len(paths) > max_paths:
        chunks.append(f"<VUL_PATH> omitted: {len(paths) - max_paths} more paths </VUL_PATH>")
    return " ".join(chunks)



def is_likely_arithmetic_line(line: str) -> bool:
    """
    Regex fallback arithmetic filter.
    Avoid treating pointer declarations (char *p) and field access (p->x)
    as arithmetic operations.
    """
    s = str(line or "")
    s = s.replace("->", " ARROW ")

    # Remove common pointer declaration/use patterns before testing arithmetic.
    s = re.sub(r"\b[A-Za-z_]\w*\s*\*", " PTR ", s)
    s = re.sub(r"\*\s*[A-Za-z_]\w*", " PTR ", s)

    # Function headers with pointer parameters are usually not arithmetic.
    if re.search(r"\)\s*\{", s) and not re.search(r"=\s*[^;]+", s):
        return False

    # True binary arithmetic roughly requires operand-op-operand.
    return bool(re.search(
        r"(\b[A-Za-z_]\w*|\d+)\s*(\+|-|\*|/|%|<<|>>)\s*(\b[A-Za-z_]\w*|\d+)",
        s,
    ))


# ============================================================
# Regex fallback
# ============================================================

def regex_fallback_extract(code: str, max_hops: int = 6, max_paths_per_sample: int = 5, include_pointer_paths: bool = False, include_lifetime_leak_paths: bool = False) -> Dict[str, Any]:
    edges, calls, checks, events, ariths, derefs = [], [], [], [], [], []
    lines = code.splitlines() if "\n" in code else re.split(r";\s*", code)
    assign_pat = re.compile(r"\b([A-Za-z_]\w*)\s*=\s*([^;]+)")
    call_pat = re.compile(r"\b([A-Za-z_]\w*)\s*\((.*?)\)")

    for i, line in enumerate(lines, start=1):
        line_s = line.strip()
        m = assign_pat.search(line_s)
        if m:
            lhs, rhs = m.group(1), m.group(2)
            rhs_vars = collect_full_expr_identifiers(rhs)
            for rv in rhs_vars:
                if rv != lhs:
                    edges.append(make_edge(rv, lhs, "assign_regex", i, line_s))
            for af in ALLOC_FUNCS:
                if re.search(r"\b" + re.escape(af) + r"\s*\(", rhs):
                    events.append({"line": i, "kind": "allocation", "var": lhs, "call": af, "expr": line_s})
            for rf in RESOURCE_OPEN_FUNCS:
                if re.search(r"\b" + re.escape(rf) + r"\s*\(", rhs):
                    events.append({"line": i, "kind": "resource_open", "var": lhs, "call": rf, "expr": line_s})

        if re.search(r"\b(if|while|for)\s*\(", line_s):
            checks.append({"line": i, "kind": "condition_regex", "condition": line_s[:300], "vars": collect_full_expr_identifiers(line_s), "is_likely_security_check": is_likely_security_check(line_s)})

        for cm in call_pat.finditer(line_s):
            # V3 fallback filter: do not treat member/function-pointer calls as
            # direct risky C library calls, e.g. ctx->ops->remove(...).
            start_pos = cm.start(1)
            prefix = line_s[max(0, start_pos - 3):start_pos]
            if prefix.endswith("->") or prefix.endswith(".") or prefix.endswith("::"):
                continue

            name, raw_args = cm.group(1), cm.group(2)
            arg_texts = [a.strip() for a in raw_args.split(",")] if raw_args.strip() else []
            args = [{"text": a, "vars": collect_full_expr_identifiers(a)} for a in arg_texts]
            calls.append({"line": i, "name": name, "callee_expr": name, "expr": cm.group(0), "args": args})
            if name in DEALLOC_FUNCS or name in RESOURCE_CLOSE_FUNCS:
                for a in args:
                    for v in a["vars"]:
                        kind = "deallocation" if name in DEALLOC_FUNCS else "resource_close"
                        events.append({"line": i, "kind": kind, "var": v, "call": name, "expr": line_s})
            for arg_idx, a in enumerate(args, start=1):
                for v in a["vars"]:
                    edges.append(make_edge(v, f"call:{name}:arg{arg_idx}@{i}", "call_arg_regex", i, line_s))

        if is_likely_arithmetic_line(line_s):
            ariths.append({"line": i, "op": "arith", "expr": line_s[:300], "vars": collect_full_expr_identifiers(line_s)})

        for sm in re.finditer(r"\b([A-Za-z_]\w*)\s*\[\s*([^\]]+)\s*\]", line_s):
            expr = sm.group(0)
            derefs.append({"line": i, "kind": "array_subscript", "expr": expr, "vars": collect_full_expr_identifiers(expr)})

    edges = dedup_edges(edges)
    paths = extract_vul_paths(edges=edges, calls=calls, checks=checks, events=events, ariths=ariths, derefs=derefs, max_hops=max_hops, max_paths_per_sample=max_paths_per_sample, include_pointer_paths=include_pointer_paths, include_lifetime_leak_paths=include_lifetime_leak_paths)
    return {"full_dfg": edges, "calls": calls, "checks": checks, "events": events, "arithmetic_ops": ariths, "dereferences": derefs, "vul_paths": paths}


# ============================================================
# Function-level analysis
# ============================================================

def analyze_function(code: str, func_name: str, max_hops: int, max_paths_per_sample: int = 5, include_pointer_paths: bool = False, include_lifetime_leak_paths: bool = False) -> Dict[str, Any]:
    lang = choose_lang(func_name, code)
    tree = parse_code(code, lang)
    if tree is None:
        result = regex_fallback_extract(code, max_hops=max_hops, max_paths_per_sample=max_paths_per_sample, include_pointer_paths=include_pointer_paths, include_lifetime_leak_paths=include_lifetime_leak_paths)
        result["parser"] = "regex_fallback"
        result["language"] = lang
        return result

    code_bytes = code.encode("utf-8", errors="ignore")
    root = tree.root_node
    try:
        edges, events = extract_full_dfg(root, code_bytes)
        checks = extract_checks(root, code_bytes)
        calls = extract_calls(root, code_bytes)
        ariths = extract_arithmetic_ops(root, code_bytes)
        derefs = extract_dereferences(root, code_bytes)
        paths = extract_vul_paths(edges=edges, calls=calls, checks=checks, events=events, ariths=ariths, derefs=derefs, max_hops=max_hops, max_paths_per_sample=max_paths_per_sample, include_pointer_paths=include_pointer_paths, include_lifetime_leak_paths=include_lifetime_leak_paths)
        return {"parser": "tree_sitter", "language": lang, "full_dfg": edges, "calls": calls, "checks": checks, "events": events, "arithmetic_ops": ariths, "dereferences": derefs, "vul_paths": paths}
    except Exception as e:
        result = regex_fallback_extract(code, max_hops=max_hops, max_paths_per_sample=max_paths_per_sample, include_pointer_paths=include_pointer_paths, include_lifetime_leak_paths=include_lifetime_leak_paths)
        result["parser"] = "regex_fallback_after_error"
        result["language"] = lang
        result["tree_sitter_error"] = str(e)
        return result


# ============================================================
# JSONL IO and split processing
# ============================================================

def read_jsonl(path: str):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def safe_func_name(row: Dict[str, Any]) -> str:
    if row.get("func_name"):
        return str(row["func_name"])
    project = str(row.get("project", "unknown_project"))
    idx = str(row.get("idx", "unknown_idx"))
    return f"{project}_{idx}.c"


def process_split(split: str, input_dir: str, output_dir: str, code_key: str, max_hops: int, max_paths_per_sample: int = 5, include_pointer_paths: bool = False, include_lifetime_leak_paths: bool = False):
    in_path = os.path.join(input_dir, f"{split}.jsonl")
    if not os.path.exists(in_path):
        print(f"[Skip] {in_path} not found.")
        return

    full_out_path = os.path.join(output_dir, "full_dfg", f"{split}.full_dfg.jsonl")
    vul_out_path = os.path.join(output_dir, "vul_paths", f"{split}.vul_paths.jsonl")
    os.makedirs(os.path.dirname(full_out_path), exist_ok=True)
    os.makedirs(os.path.dirname(vul_out_path), exist_ok=True)

    n = tree_sitter_n = fallback_n = empty_vul_path_n = 0
    rows = list(read_jsonl(in_path))

    with open(full_out_path, "w", encoding="utf-8") as fout_full, open(vul_out_path, "w", encoding="utf-8") as fout_vul:
        for row in tqdm(rows, desc=f"Processing {split}"):
            code = row.get(code_key, "")
            if not isinstance(code, str):
                code = str(code)
            fname = safe_func_name(row)
            result = analyze_function(code, fname, max_hops=max_hops, max_paths_per_sample=max_paths_per_sample, include_pointer_paths=include_pointer_paths, include_lifetime_leak_paths=include_lifetime_leak_paths)

            if result.get("parser") == "tree_sitter":
                tree_sitter_n += 1
            else:
                fallback_n += 1
            if not result.get("vul_paths", []):
                empty_vul_path_n += 1

            base = dict(row)
            base["func_name"] = fname

            full_row = dict(base)
            full_row["dfg_parser"] = result["parser"]
            full_row["dfg_language"] = result["language"]
            full_row["full_dfg"] = result["full_dfg"]
            full_row["full_dfg_text"] = edges_to_text(result["full_dfg"])
            if "tree_sitter_error" in result:
                full_row["tree_sitter_error"] = result["tree_sitter_error"]
            fout_full.write(json.dumps(full_row, ensure_ascii=False) + "\n")

            vul_row = dict(base)
            vul_row["dfg_parser"] = result["parser"]
            vul_row["dfg_language"] = result["language"]
            vul_row["vul_paths"] = result["vul_paths"]
            vul_row["vul_path_text"] = vul_paths_to_text(result["vul_paths"])
            vul_row["checks"] = result["checks"]
            vul_row["events"] = result["events"]
            vul_row["arithmetic_ops"] = result["arithmetic_ops"]
            vul_row["dereferences"] = result["dereferences"]
            if "tree_sitter_error" in result:
                vul_row["tree_sitter_error"] = result["tree_sitter_error"]
            fout_vul.write(json.dumps(vul_row, ensure_ascii=False) + "\n")
            n += 1

    print(f"[Done] {split}: {n} samples")
    print(f"       tree-sitter parsed : {tree_sitter_n}")
    print(f"       fallback parsed    : {fallback_n}")
    print(f"       empty vul_paths    : {empty_vul_path_n}")
    print(f"       full DFG -> {full_out_path}")
    print(f"       vul paths -> {vul_out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", type=str, required=True, help="Directory containing train.jsonl, valid.jsonl, test.jsonl")
    ap.add_argument("--output_dir", type=str, required=True, help="Output directory")
    ap.add_argument("--code_key", type=str, default="func", help="JSON key for source code")
    ap.add_argument("--max_hops", type=int, default=6, help="Backward tracing hops from sink variables")
    ap.add_argument("--max_paths_per_sample", type=int, default=5, help="Keep only top-k vulnerability-oriented evidence paths per sample")
    ap.add_argument("--include_pointer_paths", action="store_true", help="Enable generic pointer dereference evidence paths; disabled by default because they are noisy")
    ap.add_argument("--include_lifetime_leak_paths", action="store_true", help="Enable rough memory/resource leak candidate paths; disabled by default because they are noisy")
    ap.add_argument("--splits", type=str, default="train,valid,test", help="Comma-separated split names")
    args = ap.parse_args()

    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        process_split(split=split, input_dir=args.input_dir, output_dir=args.output_dir, code_key=args.code_key, max_hops=args.max_hops, max_paths_per_sample=args.max_paths_per_sample, include_pointer_paths=args.include_pointer_paths, include_lifetime_leak_paths=args.include_lifetime_leak_paths)


if __name__ == "__main__":
    main()
