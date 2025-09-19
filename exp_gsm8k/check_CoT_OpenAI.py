#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ========= Install tip =========
# pip install --upgrade openai tqdm
# =================================

import os, json, time, csv, sys, signal, argparse, re
from typing import List, Dict, Any, Iterable, Tuple, Optional
from openai import OpenAI
from tqdm import tqdm

# -------- Config (can be overridden by CLI) --------
MODEL = "gpt-4o"  # or gpt-4o-2024-08-06 / gpt-4.1-mini

RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "cot_process_judgment",
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "final_pred": {"type": "string"},
                "gold": {"type": "string"},
                "final_correct": {"type": "boolean"},
                "process_has_error": {"type": "boolean"},
                "process_wrong_but_answer_correct": {"type": "boolean"},
                "wrong_steps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "span": {"type": "string"},
                            "explanation": {"type": "string"}
                        },
                        "required": ["span", "error_type", "explanation"]
                    }
                }
            },
            "required": [
                "final_pred", "gold", "final_correct",
                "process_has_error", "process_wrong_but_answer_correct", "wrong_steps"
            ]
        },
        "strict": True
    },
}

SYSTEM_INSTRUCTIONS = """\
You judge math-CoT outputs. Read the provided sample fields:
- question, reason (gold reasoning if present), output (model's CoT+answer), pred, gold.
Decide:
1) final_pred vs gold: extract numeric answers and check equality (ignore punctuation and text).
2) process_has_error: TRUE if the CoT contains any incorrect arithmetic equality,
   inconsistent tallying (e.g., missing a required addend), or logical step that contradicts the numbers.
   Prefer explicit "=" checks when possible; otherwise explain the inconsistency (omission/logic).
3) wrong_steps: list 1-5 concrete snippets (short spans from "output") that are has arithmetic error with 1-2 sentence explanation.
4) process_wrong_but_answer_correct = final_correct == TRUE AND process_has_error == TRUE.

Be precise and conservative. If uncertain, set process_has_error to FALSE and leave wrong_steps empty.
Return only JSON matching the schema.
"""

# -------- OpenAI client --------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ========= Utilities =========

def normalize_num(s: str) -> str:
    if not s:
        return ""
    m = re.search(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?", str(s))
    return m.group(0).replace(",", "") if m else ""

def build_user_payload(sample: Dict[str, Any]) -> Dict[str, Any]:
    body = {
        "id": sample.get("id", ""),
        "question": sample.get("question", ""),
        "reason": sample.get("reason", ""),
        "output": sample.get("output", ""),
        "pred": sample.get("pred") or sample.get("answer") or "",
        "gold": sample.get("gold") or sample.get("answer") or "",
    }
    return {
        "role": "user",
        "content": [{"type": "input_text", "text": json.dumps(body, ensure_ascii=False)}],
    }

def _call_with_optional_schema(model: str, sys_msg: str, user_msg: Dict[str, Any], use_schema: bool = True):
    kwargs = {
        "model": model,
        "input": [{"role": "system", "content": sys_msg}, user_msg],
    }
    if use_schema:
        kwargs["response_format"] = RESPONSE_FORMAT
    return client.responses.create(**kwargs)

def parse_json_strict(txt: str) -> Dict[str, Any]:
    try:
        return json.loads(txt)
    except Exception:
        m = re.search(r"\{.*\}", txt, flags=re.DOTALL)
        if not m:
            raise ValueError(f"Model did not return JSON. Got: {txt[:200]}...")
        return json.loads(m.group(0))

def judge_one(sample: Dict[str, Any], max_retries: int = 6, base_delay: float = 1.0) -> Dict[str, Any]:
    # Local normalization as fallback
    pred_local = normalize_num(sample.get("pred") or sample.get("answer") or "")
    gold_local = normalize_num(sample.get("gold") or sample.get("answer") or "")

    last_err = None
    for attempt in range(max_retries + 1):
        try:
            try:
                rsp = _call_with_optional_schema(MODEL, SYSTEM_INSTRUCTIONS, build_user_payload(sample), use_schema=True)
            except TypeError:
                rsp = _call_with_optional_schema(MODEL, SYSTEM_INSTRUCTIONS, build_user_payload(sample), use_schema=False)

            data = parse_json_strict(rsp.output_text)
            # Post-fix fields with local normalization as a backstop
            data["final_pred"] = normalize_num(data.get("final_pred", "") or pred_local)
            data["gold"] = normalize_num(data.get("gold", "") or gold_local)
            data["final_correct"] = (data["final_pred"] != "" and data["final_pred"] == data["gold"])
            data["process_wrong_but_answer_correct"] = (data["final_correct"] and bool(data.get("process_has_error", False)))
            data["id"] = sample.get("id", "")
            return data
        except Exception as e:
            last_err = e
            # Exponential backoff with jitter
            delay = base_delay * (2 ** attempt)
            time.sleep(min(delay, 20.0))
    # If all retries exhausted, raise
    raise RuntimeError(f"judge_one failed after retries: {last_err}")

# ========= Resume / I/O helpers =========

def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    if not os.path.isfile(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except Exception:
                    # skip corrupted line
                    continue

def load_done_ids(jsonl_path: str) -> set:
    done = set()
    for obj in read_jsonl(jsonl_path):
        _id = obj.get("id")
        if _id:
            done.add(_id)
    return done

def append_jsonl_one(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def rewrite_csv_from_jsonl(jsonl_path: str, csv_path: str) -> None:
    fieldnames = [
        "id", "final_pred", "gold", "final_correct",
        "process_has_error", "process_wrong_but_answer_correct", "wrong_steps_json"
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=fieldnames)
        writer.writeheader()
        for r in read_jsonl(jsonl_path):
            row = {
                "id": r.get("id", ""),
                "final_pred": r.get("final_pred", ""),
                "gold": r.get("gold", ""),
                "final_correct": r.get("final_correct", False),
                "process_has_error": r.get("process_has_error", False),
                "process_wrong_but_answer_correct": r.get("process_wrong_but_answer_correct", False),
                "wrong_steps_json": json.dumps(r.get("wrong_steps", []), ensure_ascii=False),
            }
            writer.writerow(row)

# ========= Batch with resume =========

INTERRUPTED = False
def _sigint_handler(signum, frame):
    global INTERRUPTED
    INTERRUPTED = True
signal.signal(signal.SIGINT, _sigint_handler)

def judge_batch_resume(
    samples: List[Dict[str, Any]],
    out_jsonl: str,
    out_csv: str,
    flush_every: int = 20,
) -> None:
    """
    - 从 out_jsonl 恢复已完成样本（按 id）
    - 对未完成样本：逐条评估并 **立刻追加** 到 JSONL
    - 每 flush_every 条 / 或收到 Ctrl-C：重写 CSV 快照，安全退出
    """
    os.makedirs(os.path.dirname(os.path.abspath(out_jsonl)) or ".", exist_ok=True)

    done_ids = load_done_ids(out_jsonl)
    total = len(samples)
    initial = len(done_ids)

    pbar = tqdm(total=total, initial=initial, unit="sample", desc="Judging samples (resumable)")

    processed_since_flush = 0

    for s in samples:
        if INTERRUPTED:
            break
        sid = s.get("id", "")
        if sid in done_ids:
            # 已完成：仅推进进度
            pbar.update(1)
            continue

        try:
            result = judge_one(s)
        except Exception as e:
            # 将失败样本写入一个错误日志，避免阻塞全集
            err_log = os.path.splitext(out_jsonl)[0] + ".errors.jsonl"
            append_jsonl_one(err_log, {"id": sid, "error": str(e)})
            # 也推进，防止卡死（你也可以选择不推进以便后续重跑）
            pbar.update(1)
            continue

        append_jsonl_one(out_jsonl, result)
        done_ids.add(sid)
        processed_since_flush += 1
        pbar.update(1)

        if processed_since_flush >= flush_every:
            rewrite_csv_from_jsonl(out_jsonl, out_csv)
            processed_since_flush = 0

    # 收尾：不论是否中断，都再刷新一次 CSV
    rewrite_csv_from_jsonl(out_jsonl, out_csv)
    pbar.close()

# ========= CLI =========

def load_input_samples(path: str) -> List[Dict[str, Any]]:
    samples = []
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))
    return samples

def main():

    input_file_path = "/common/users/sl2148/Public/yang_ouyang/alignment-attribution-code/out/GSM8K/llama2-7b-chat-hf/unstructured/wanda_3_set_difference_utility_weightonly/wanda_4_set_difference_cot0shot/eval_selected_samples/prompt_direct,cot0shot/pure_pq_0.01_granular/pq_0.1_0.1_k_0.17_u_0.15/gsm8k_bottom_0.001101_direct,cot0shot_selected_samples_prompt_cot0shot.jsonl"
    # input_file_path = "/common/users/sl2148/Public/yang_ouyang/alignment-attribution-code/data/GSM8K_eval_build/eval_cot0shot.jsonl"
    output_file_prefix = "pq_0.1_0.1_k_0.17_u_0.15_gsm8k_cot_judgments"
    # output_file_prefix = "full_gsm8k_cot_judgments"
    parser = argparse.ArgumentParser(description="CoT judgment with resumable progress.")
    parser.add_argument("--in_jsonl", type=str, required=False,
                        default=input_file_path,
                        help="输入样本 JSONL")
    parser.add_argument("--out_prefix", type=str, default=output_file_prefix,
                        help="输出前缀（会生成 <prefix>.jsonl 与 <prefix>.csv）")
    parser.add_argument("--flush_every", type=int, default=20, help="每 N 条刷新一次 CSV")
    args = parser.parse_args()

    in_path = args.in_jsonl
    out_jsonl = f"{args.out_prefix}.jsonl"
    out_csv = f"{args.out_prefix}.csv"

    samples = load_input_samples(in_path)
    if not samples:
        print(f"[WARN] No samples loaded from {in_path}", file=sys.stderr)
        return

    print(f"[INFO] Loaded {len(samples)} samples.")
    print(f"[INFO] Output -> {out_jsonl} (append, resumable), {out_csv}")

    judge_batch_resume(samples, out_jsonl=out_jsonl, out_csv=out_csv, flush_every=args.flush_every)
    print(f"[DONE] Saved: {out_jsonl} and {out_csv}", file=sys.stderr)

if __name__ == "__main__":
    main()
