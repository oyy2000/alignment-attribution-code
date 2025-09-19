# Utility to detect samples where the *process/CoT has arithmetic mistakes* 
# but the *final answer is correct*. Works on JSON or JSONL files or in-memory lists. 
#
# What it checks:
# - Scans the model "output" text for explicit arithmetic equalities like
#   "3 + 5 = 9", "12*4= 47", "100,000 / 5 = 19999", etc.
# - Verifies each equality by evaluating the left-hand expression and 
#   comparing to the stated right-hand number.
# - Flags a sample as "process_wrong_but_answer_correct" if:
#       (a) pred == gold  (final answer correct) AND
#       (b) any explicit equation inside "output" is numerically false.
#
# Limitations:
# - If the "wrong reasoning" is due to omission/logic without explicit "="
#   (e.g., forgetting a term) this simple checker won't catch it.
# - For broader coverage, you could add dataset-specific templates to reconstruct
#   implied calculations or compare against the "reason" field, but this script
#   intentionally uses a conservative, explainable criterion.
#
# You can run this against a local JSONL by putting the path in FILE_PATH below.

import json, re, ast, operator, math, os
from typing import Any, Dict, List, Tuple, Optional
import pandas as pd

ALLOWED_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
    ast.Mod: operator.mod,
    # (extend if needed)
}

def _safe_eval_expr(expr: str) -> Optional[float]:
    """Safely evaluate a basic arithmetic expression. Returns None if unsafe/invalid."""
    # Normalize unicode operators
    expr = expr.replace("×", "*").replace("⋅", "*").replace("·", "*").replace("÷", "/")
    expr = expr.replace("—", "-").replace("–", "-")
    # Remove thousands separators
    expr = expr.replace(",", "")
    # Disallow letters
    if re.search(r"[A-Za-z]", expr):
        return None
    # Only allow digits, parentheses, decimal points, and ops
    if re.search(r"[^0-9\.\+\-\*\/\(\)\s%]", expr):
        return None
    try:
        node = ast.parse(expr, mode="eval")
    except Exception:
        return None

    def _eval(n):
        if isinstance(n, ast.Expression):
            return _eval(n.body)
        elif isinstance(n, ast.Num):
            return n.n
        elif isinstance(n, ast.UnaryOp) and type(n.op) in ALLOWED_OPS:
            return ALLOWED_OPS[type(n.op)](_eval(n.operand))
        elif isinstance(n, ast.BinOp) and type(n.op) in ALLOWED_OPS:
            return ALLOWED_OPS[type(n.op)](_eval(n.left), _eval(n.right))
        elif isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        else:
            raise ValueError("Unsafe/unsupported node")
    try:
        return float(_eval(node))
    except Exception:
        return None

# Regex to capture *whole* equalities in natural text. We split on "=" and try to evaluate LHS.
EQUALITY_PATTERN = re.compile(r"(?P<lhs>[\d\(\)\s\.,\+\-\*×xX⋅·/÷%]+)\s*=\s*(?P<rhs>[-+]?\s*[\d,]+(?:\.\d+)?)")

def analyze_output_text(output_text: str) -> Dict[str, Any]:
    """Return details about arithmetic equalities found in the output CoT."""
    findings = []
    # Replace common 'x' (as multiplication) conservatively only when it's between numbers/spaces
    text_norm = re.sub(r"(?<=\d)\s*[xX]\s*(?=\d)", "*", output_text)
    for m in EQUALITY_PATTERN.finditer(text_norm):
        lhs_raw = m.group("lhs").strip()
        rhs_raw = m.group("rhs").replace(",", "").strip()
        try:
            rhs_val = float(rhs_raw)
        except Exception:
            rhs_val = None

        lhs_val = _safe_eval_expr(lhs_raw)
        correct = (lhs_val is not None and rhs_val is not None and abs(lhs_val - rhs_val) < 1e-6)
        findings.append({
            "span": m.group(0),
            "lhs": lhs_raw,
            "rhs": rhs_raw,
            "lhs_val": lhs_val,
            "rhs_val": rhs_val,
            "equation_correct": bool(correct),
        })
    has_wrong_equation = any((f["equation_correct"] is False) for f in findings) if findings else False
    return {"equations": findings, "has_wrong_equation": has_wrong_equation, "num_equations": len(findings)}

def normalize_ans(s: Any) -> str:
    if s is None:
        return ""
    s = str(s).strip()
    # Keep only trailing numeric portion
    # e.g., "The answer is 82." -> "82"
    m = re.search(r"([-+]?\d+(?:,\d{3})*(?:\.\d+)?)", s)
    if m:
        return m.group(1).replace(",", "")
    return s

def classify_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    out = sample.get("output") or ""
    pred = normalize_ans(sample.get("pred") or sample.get("answer") or "")
    gold = normalize_ans(sample.get("gold") or "")
    final_correct = (pred != "" and gold != "" and pred == gold)

    analysis = analyze_output_text(out)
    return {
        "id": sample.get("id", ""),
        "final_pred": pred,
        "gold": gold,
        "final_correct": final_correct,
        "has_wrong_equation": analysis["has_wrong_equation"],
        "num_equations_found": analysis["num_equations"],
        "examples": [f["span"] for f in analysis["equations"][:3]],
        "process_wrong_but_answer_correct": (final_correct and analysis["has_wrong_equation"]),
    }

# --- Demo on user's provided sample and a synthetic "wrong CoT but correct answer" case ---
samples = []
# load samples from a local JSONL file if desired
file_path = "/common/users/sl2148/Public/yang_ouyang/alignment-attribution-code/out/GSM8K/llama2-7b-chat-hf/unstructured/wanda_3_set_difference_utility_weightonly/GSM8K/eval_selected_samples/prompt_direct,cot0shot/step_0.01_sp_2e-07_k_0.01/pq_0.1_0.1_k_0.17_u_0.15/gsm8k_bottom_0.001101_direct,cot0shot_selected_samples_prompt_cot0shot.jsonl"
if os.path.isfile(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

rows = [classify_sample(s) for s in samples]
df = pd.DataFrame(rows)
save_path = "./data/reasoning_check_demo.csv"
df.to_csv(save_path, index=False)

