#!/usr/bin/env python3
"""
Adds pqku (p, q, k, u) to the figure and the printed summary table.
- Parses pqku from directory names like: pq_0.2_0.2_k_64_u_0.1/
- Annotates points in the plot with k values
- Adds columns P, Q, K, U to the summary table
- Stores pqku metadata alongside results in the JSON output

Assumes the same directory/file layout as the previous refactor.
"""
import json
import os
import re
import glob
from typing import Dict, List, Tuple, Optional
import numpy as np  

import matplotlib.pyplot as plt
# collect_k_values_data_600_cot4shot_pqku_granular
# --- CONFIG ---
# BASE_DIR = ("/common/users/sl2148/Public/yang_ouyang/alignment-attribution-code/out/llama2-7b-chat-hf/unstructured/"
# "wanda_3_set_difference_utility_weightonly/wanda_3_set_difference_utility/eval_selected_samples/prompt_direct,cot0shot/")
# BASE_DIR = ("/common/users/sl2148/Public/yang_ouyang/alignment-attribution-code/out/llama2-7b-chat-hf/unstructured/"
# "wanda_3_set_difference_utility_weightonly/wanda_4_set_difference_cot0shot/eval_selected_samples/prompt_direct,cot0shot/0.01_sp_0.0005_0.05_granular")

# BASE_DIR = ("/common/users/sl2148/Public/yang_ouyang/alignment-attribution-code/out/llama2-7b-chat-hf/unstructured/wanda_3_set_difference_utility_weightonly/wanda_4_set_difference_cot0shot"
# "/eval_selected_samples/prompt_direct,cot0shot/0.01_sp_0.00001_0.05_granular")

BASE_DIR = ("/common/users/sl2148/Public/yang_ouyang/alignment-attribution-code/out/llama2-7b-chat-hf/unstructured/wanda_2_set_difference_utility_weightonly/wanda_3_set_difference_cot0shot/eval_selected_samples/prompt_direct,cot0shot/step_0.01_sp_5e-06_k_0.01/")
BASE_DIR = ("/common/users/sl2148/Public/yang_ouyang/alignment-attribution-code/out/Addition:6/llama2-7b-chat-hf/unstructured/wanda_3_set_difference_utility_weightonly/wanda_3_set_difference_cot0shot/eval_all/prompt_direct,cot0shot/step_0.01_sp_2e-07_k_0.01/")
K_OPTIONS = [0.17] 
U_OPTIONS = [0.15] 
OUTPUT_DIR = "/common/users/sl2148/Public/yang_ouyang/alignment-attribution-code/out/Addition:6/llama2-7b-chat-hf/unstructured/wanda_3_set_difference_utility_weightonly/wanda_3_set_difference_cot0shot/eval_all/prompt_direct,cot0shot/step_0.01_sp_2e-07_k_0.01/"

ORIG_COT4_FILE = (
    "/common/users/sl2148/Public/yang_ouyang/alignment-attribution-code/out/Addition:6/llama2-7b-chat-hf/direct,cot0shot/eval_all/addition_bottom_0.000000_direct,cot0shot_all_prompt_cot0shot.jsonl"
)
ORIG_DIRECT_FILE = (
    "/common/users/sl2148/Public/yang_ouyang/alignment-attribution-code/out/Addition:6/llama2-7b-chat-hf/direct,cot0shot/eval_all/addition_bottom_0.000000_direct,cot0shot_all_prompt_direct.jsonl"
)

CATEGORIES = ["direct_success_cot_success", "direct_fail_cot_success", "direct_success_cot_success_cot_prompt"]
CATEGORY_COLORS = ['blue', 'red', 'green']

# --- IO helpers ---

def load_jsonl(file_path: str) -> List[Dict]:
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def extract_sample_info(data: List[Dict]) -> Dict[str, Dict]:
    sample_info = {}
    for item in data:
        sid = item.get('id', '')
        if sid:
            sample_info[sid] = {
                'correct': item.get('correct', False),
                'question': item.get('question', ''),
                'pred': item.get('pred', ''),
                'gold': item.get('gold', '')
            }
    return sample_info


def match_samples_by_question(data1: List[Dict], data2: List[Dict]) -> Dict[str, str]:
    q_to_id1 = {item.get('question', ''): item.get('id', '') for item in data1}
    matches = {}
    for item in data2:
        q = item.get('question', '')
        if q in q_to_id1:
            matches[q_to_id1[q]] = item.get('id', '')
    return matches

# --- Categorization ---

def categorize_samples(cot_info: Dict[str, Dict], direct_info: Dict[str, Dict], matches: Dict[str, str]):
    d_s_c_s, d_s_c_f, d_f_c_s, d_f_c_f = [], [], [], []
    for cot_id, direct_id in matches.items():
        cot_ok = cot_info.get(cot_id, {}).get('correct', False)
        dir_ok = direct_info.get(direct_id, {}).get('correct', False)
        if dir_ok and cot_ok:
            d_s_c_s.append(cot_id)
        elif dir_ok and not cot_ok:
            d_s_c_f.append(cot_id)
        elif (not dir_ok) and cot_ok:
            d_f_c_s.append(cot_id)
        else:
            d_f_c_f.append(cot_id)
    return d_s_c_s, d_s_c_f, d_f_c_s, d_f_c_f

# --- Analysis ---

def analyze_category_performance_original(category: str, sample_cot_ids, orig_cot, orig_direct, matches):
    if not sample_cot_ids:
        return {
            'category': category,
            'pruned_correct_count': 0,
            'original_correct': 0,
            'original_accuracy': 0.0,
            'pruned_correct': 0,
            'pruned_accuracy': 0.0,
            'accuracy_change': 0.0,
        }
    total = len(sample_cot_ids)
    orig_correct = 0
    if category == 'direct_success_cot_success':
        for cot_id in sample_cot_ids:
            dir_id = matches.get(cot_id)
            if dir_id and orig_direct.get(dir_id, {}).get('correct', False):
                orig_correct += 1
    elif category == 'direct_fail_cot_success':
        for cot_id in sample_cot_ids:
            if orig_cot.get(cot_id, {}).get('correct', False):
                orig_correct += 1
    elif category == 'direct_success_cot_success_cot_prompt':
        for cot_id in sample_cot_ids:
            dir_id = matches.get(cot_id)
            if dir_id and orig_cot.get(dir_id, {}).get('correct', False):
                orig_correct += 1
        
    acc = orig_correct / total if total else 0.0
    return {
        'category': category,
        'pruned_correct_count': orig_correct,
        'original_correct': orig_correct,
        'original_accuracy': acc,
        'pruned_correct': orig_correct,
        'pruned_accuracy': acc,
        'accuracy_change': 0.0,
    }


def analyze_category_performance_pruned(category: str, sample_cot_ids, orig_cot, orig_direct, pruned_cot, pruned_direct, matches):
    if not sample_cot_ids:
        return {
            'category': category,
            'pruned_correct_count': 0,
            'original_correct': 0,
            'original_accuracy': 0.0,
            'pruned_correct': 0,
            'pruned_accuracy': 0.0,
            'accuracy_change': 0.0,
        }
    total = len(sample_cot_ids)
    base = analyze_category_performance_original(category, sample_cot_ids, orig_cot, orig_direct, matches)
    base_acc = base['original_accuracy']
    pruned_correct = 0
    if category == 'direct_success_cot_success':
        for cot_id in sample_cot_ids:
            dir_id = matches.get(cot_id)
            if dir_id and pruned_direct.get(dir_id, {}).get('correct', False):
                pruned_correct += 1
    elif category == 'direct_fail_cot_success':
        for cot_id in sample_cot_ids:
            if pruned_cot.get(cot_id, {}).get('correct', False):
                pruned_correct += 1
    elif category == 'direct_success_cot_success_cot_prompt':
        for cot_id in sample_cot_ids:
            dir_id = matches.get(cot_id)
            if dir_id and pruned_cot.get(dir_id, {}).get('correct', False):
                pruned_correct += 1
    pruned_acc = pruned_correct / total if total else 0.0
    print(f"Category: {category}, Total: {total}, Original Correct: {base['original_correct']}, Pruned Correct: {pruned_correct}")
    return {
        'category': category,
        'pruned_correct_count': pruned_correct,
        'original_correct': int(round(base_acc * total)),
        'original_accuracy': base_acc,
        'pruned_correct': pruned_correct,
        'pruned_accuracy': pruned_acc,
        'accuracy_change': pruned_acc - base_acc,
    }

# --- pqku parsing / discovery ---

PQKU_RE = re.compile(r"pq_([0-9.]+)_([0-9.]+)_k_([0-9.]+)_u_([0-9.]+)")


def parse_pqku_from_dir(dir_path: str) -> Optional[Dict[str, float]]:
    m = PQKU_RE.search(os.path.basename(dir_path.rstrip('/')))
    if not m:
        # sometimes BASE_DIR/pq_.../ ; take the last component
        parts = [p for p in dir_path.split(os.sep) if p]
        for comp in reversed(parts):
            mm = PQKU_RE.search(comp)
            if mm:
                m = mm
                break
    if not m:
        return None
    p, q, k, u = m.groups()
    return {"p": float(p), "q": float(q), "k": float(k), "u": float(u)}


def discover_sparsity_dirs(base_dir: str) -> Dict[float, Dict]:
    """Return {sparsity_ratio: {dir, p,q,k,u}}."""
    mapping: Dict[float, Dict] = {}
    k_dirs = glob.glob(os.path.join(base_dir, FILE_NAME))
    for d in k_dirs:
        files = glob.glob(os.path.join(d, "*.jsonl"))
        if not files:
            continue
        fname = os.path.basename(files[0])
        m = re.search(r"(?:gsm8k|addition)_bottom_([0-9.]+)_", fname, flags=re.IGNORECASE)

        if not m:
            continue
        ratio = float(m.group(1))
        if ratio <= 0:
            continue
        meta = parse_pqku_from_dir(d) or {"p": None, "q": None, "k": None, "u": None}
        meta.update({"dir": d})
        mapping[ratio] = meta
    return dict(sorted(mapping.items()))

# --- Collection ---

def collect_all_results_with_meta():
    # Load originals
    cot4_data = load_jsonl(ORIG_COT4_FILE)
    dir_data = load_jsonl(ORIG_DIRECT_FILE)
    cot4_info = extract_sample_info(cot4_data)
    dir_info = extract_sample_info(dir_data)
    matches = match_samples_by_question(cot4_data, dir_data)

    d_s_c_s, _, d_f_c_s, _ = categorize_samples(cot4_info, dir_info, matches)
    category_samples = {
        "direct_success_cot_success": d_s_c_s,
        "direct_fail_cot_success": d_f_c_s,
        "direct_success_cot_success_cot_prompt": d_s_c_s,
    }

    all_results: Dict[float, Dict] = {}

    # 0.0 original row (no pqku)
    orig_bucket = {}
    for name, samples in category_samples.items():
        orig_bucket[name] = analyze_category_performance_original(name, samples, cot4_info, dir_info, matches)
    all_results[0.0] = {"metrics": orig_bucket, "meta": {"p": None, "q": None, "k": None, "u": None, "dir": None}}

    ratio_to_meta = discover_sparsity_dirs(BASE_DIR)
    # Debug: show discovered sparsity dirs
    print(f"Discovered {len(ratio_to_meta)} sparsity dirs under BASE_DIR={BASE_DIR}")
    for rr, mm in ratio_to_meta.items():
        print(f"  ratio={rr} -> dir={mm.get('dir')} meta(p,q,k,u)={(mm.get('p'), mm.get('q'), mm.get('k'), mm.get('u'))}")

    for ratio, meta in ratio_to_meta.items():
        d = meta["dir"]
        # robust file resolution
        cand_cot = sorted(glob.glob(os.path.join(d, "*prompt_cot0shot.jsonl")))
        cand_dir = sorted(glob.glob(os.path.join(d, "*prompt_direct.jsonl")))
        if not cand_cot or not cand_dir:
            all_results[ratio] = {"metrics": {}, "meta": meta}
            continue
        pruned_cot_info = extract_sample_info(load_jsonl(cand_cot[0]))
        pruned_dir_info = extract_sample_info(load_jsonl(cand_dir[0]))

        bucket = {}
        for name, samples in category_samples.items():
            bucket[name] = analyze_category_performance_pruned(
                name, samples, cot4_info, dir_info, pruned_cot_info, pruned_dir_info, matches
            )
        all_results[ratio] = {"metrics": bucket, "meta": meta}

    return dict(sorted(all_results.items(), key=lambda kv: kv[0]))


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for pq in PQ_OPTIONS:
        fig, axes = plt.subplots(len(U_OPTIONS), 1, figsize=(8, 4 * len(U_OPTIONS)), sharex=False)
        # 在 main() 内 for idx, u in enumerate(U_OPTIONS): 这一层里，替换子图绘制部分
        for idx, u in enumerate(U_OPTIONS):
            global FILE_NAME
            FILE_NAME = f"pq_{pq}_{pq}_k_*_u_{u}/"

            print(f"Collecting results for u={u}...")
            all_results = collect_all_results_with_meta()

            ratios = sorted(all_results.keys())
            ax = axes[idx] if len(U_OPTIONS) > 1 else axes

            # 先分别计算两条曲线的 delta
            deltas_per_cat = []
            for i, cat in enumerate(CATEGORIES):
                delta = []
                for r in ratios:
                    res = all_results[r].get("metrics", {}).get(cat, {})
                    delta.append(res.get('accuracy_change', 0.0))
                deltas_per_cat.append(delta)
                ax.plot(ratios, delta, 'o-', label=cat, color=CATEGORY_COLORS[i])

            # ---- 在图中标注 k（使用两条曲线的平均 y 作为标注位置）----
            # deltas_per_cat[0] 对应 direct_success_cot_success
            # deltas_per_cat[1] 对应 direct_fail_cot_success
            for j, r in enumerate(ratios):
                if r == 0.0:  # 原始点没有 pqku 元信息，跳过
                    continue
                meta = all_results[r].get("meta", {})
                k_val = meta.get("k", None)
                if k_val is None:
                    continue
                # 计算标注 y 位置（两条曲线的平均；若某条不存在就用另一条）
                y_candidates = []
                if len(deltas_per_cat) > 0 and j < len(deltas_per_cat[0]):
                    y_candidates.append(deltas_per_cat[0][j])
                if len(deltas_per_cat) > 1 and j < len(deltas_per_cat[1]):
                    y_candidates.append(deltas_per_cat[1][j])
                if not y_candidates:
                    continue
                y_pos = float(np.mean(y_candidates))
                # 文本格式：k=64（如果是整数就不带小数）
                k_text = f"k={int(k_val) if float(k_val).is_integer() else k_val}"
                ax.annotate(
                    k_text,
                    (r, y_pos),
                    textcoords="offset points",
                    xytext=(6, 6),
                    fontsize=7
                )
            # -----------------------------------------------------------

            ax.set_title(f'u = {u}')
            ax.set_xlabel('Sparsity Ratio')
            ax.set_ylabel('Accuracy Change')
            ax.set_ylim(-1.0, 0.0)  # 统一 y 轴
            ax.grid(True, alpha=0.3)
            ax.legend()

            FILE_NAME = f"pq_{pq}_{pq}_k_*_u_{u}/"

            print(f"Collecting results for u={u}...")
            all_results = collect_all_results_with_meta()

            ratios = sorted(all_results.keys())
            ax = axes[idx] if len(U_OPTIONS) > 1 else axes

            # 先分别计算两条曲线的 delta
            deltas_per_cat = []
            for i, cat in enumerate(CATEGORIES):
                delta = []
                for r in ratios:
                    res = all_results[r].get("metrics", {}).get(cat, {})
                    delta.append(res.get('accuracy_change', 0.0))
                deltas_per_cat.append(delta)
                ax.plot(ratios, delta, 'o-', label=cat, color=CATEGORY_COLORS[i])

            # ---- 标注 k 值 ----
            for j, r in enumerate(ratios):
                if r == 0.0:
                    continue
                meta = all_results[r].get("meta", {})
                k_val = meta.get("k", None)
                if k_val is None:
                    continue
                y_candidates = []
                if len(deltas_per_cat) > 0 and j < len(deltas_per_cat[0]):
                    y_candidates.append(deltas_per_cat[0][j])
                if len(deltas_per_cat) > 1 and j < len(deltas_per_cat[1]):
                    y_candidates.append(deltas_per_cat[1][j])
                if not y_candidates:
                    continue
                y_pos = float(np.mean(y_candidates))
                k_text = f"k={int(k_val) if float(k_val).is_integer() else k_val}"
                ax.annotate(
                    k_text,
                    (r, y_pos),
                    textcoords="offset points",
                    xytext=(6, 6),
                    fontsize=7
                )
            # -------------------

            ax.set_title(f'u = {u}')
            ax.set_xlabel('Sparsity Ratio')
            ax.set_ylabel('Accuracy Change')
            ax.set_ylim(-1.0, 0.0)
            ax.grid(True, alpha=0.3)
            ax.legend()

            # === 单独保存每个 u 的图 ===
            fig_single, ax_single = plt.subplots(figsize=(6, 4))
            for i, cat in enumerate(CATEGORIES):
                ax_single.plot(ratios, deltas_per_cat[i], 'o-', label=cat, color=CATEGORY_COLORS[i])
            # 再画 k 值标注
            for j, r in enumerate(ratios):
                if r == 0.0:
                    continue
                meta = all_results[r].get("meta", {})
                k_val = meta.get("k", None)
                if k_val is None:
                    continue
                y_candidates = []
                if len(deltas_per_cat) > 0 and j < len(deltas_per_cat[0]):
                    y_candidates.append(deltas_per_cat[0][j])
                if len(deltas_per_cat) > 1 and j < len(deltas_per_cat[1]):
                    y_candidates.append(deltas_per_cat[1][j])
                if not y_candidates:
                    continue
                y_pos = float(np.mean(y_candidates))
                k_text = f"k={int(k_val) if float(k_val).is_integer() else k_val}"
                ax_single.annotate(
                    k_text,
                    (r, y_pos),
                    textcoords="offset points",
                    xytext=(6, 6),
                    fontsize=7
                )

            ax_single.set_title(f'u = {u}')
            ax_single.set_xlabel('Sparsity Ratio')
            ax_single.set_ylabel('Accuracy Change')
            ax_single.set_ylim(-1.0, 0.0)
            ax_single.grid(True, alpha=0.3)
            ax_single.legend()

            single_out = os.path.join(OUTPUT_DIR, f'sparsity_ratios_analysis_pq_{pq}_u_{u}.png')
            fig_single.tight_layout()
            fig_single.savefig(single_out, dpi=300, bbox_inches='tight')
            plt.close(fig_single)
            print(f"Single plot saved to: {single_out}")


        out_png = os.path.join(OUTPUT_DIR, f'sparsity_ratios_analysis_pq_{pq}_u_sweep.png')
        plt.tight_layout()
        plt.savefig(out_png, dpi=300, bbox_inches='tight')
        print(f"Subplots saved to: {out_png}")

if __name__ == '__main__':
    main()
