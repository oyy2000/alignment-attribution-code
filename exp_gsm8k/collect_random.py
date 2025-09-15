#!/usr/bin/env python3
"""
Switch from 'k wildcard' to 'pq wildcard':
- Discover dirs with pattern: pq_*_*_k_<k>_u_<u>/
- Annotate points with pq=(p,q) instead of k
- Minimal changes elsewhere
"""
import json
import os
import re
import glob
from typing import Dict, List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
import math


# Max number of sparsity ratio points (including baseline 0.0) to plot per (k,u) panel.
# Set to None or a large number to disable.
MAX_POINTS = 20
FORCE_ZERO_ORIGIN = False  # Use raw accuracy on y-axis (no delta)
NORMALIZE_BASELINE = True   # When True, insert a baseline point at x=0 with y=1 (accuracy normalized by baseline)

# Figure sizing controls (avoid overly wide x dimension)
FIG_MIN_WIDTH = 6
FIG_MAX_WIDTH = 12   # clamp maximum width
FIG_WIDTH_PER_LABEL = 0.2  # previous was 0.7, now narrower
TICK_THIN_THRESHOLD_1 = 25  # if labels exceed this, thin every 2
TICK_THIN_THRESHOLD_2 = 40  # if exceed this, thin every 3

# Annotate each plotted (p,q) point with its sparsity ratio value (derived from filenames)
ANNOTATE_SPARSITY = True
ANNOTATION_FONTSIZE = 6
ANNOTATION_ROTATION = 90  # vertical to save horizontal space
ANNOTATION_Y_OFFSET = 4   # points upward shift
ANNOTATE_CATEGORY_INDEX = 0  # which category series to anchor annotations on

BASE_DIR = ("/common/users/sl2148/Public/yang_ouyang/alignment-attribution-code/out/Addition:6/llama2-7b-chat-hf/unstructured/random_weightonly/eval_all/prompt_direct,cot0shot/")
OUTPUT_DIR = "/common/users/sl2148/Public/yang_ouyang/alignment-attribution-code/out/Addition:6/figures/"
K_OPTIONS = [0.90] 
U_OPTIONS = [0.90] 

CATEGORIES = ["direct_success_cot_success", "direct_fail_cot_success", "direct_success_cot_success_cot_prompt"]
CATEGORY_COLORS = ['blue', 'red', 'green']

ORIG_COT4_FILE = (
    "/common/users/sl2148/Public/yang_ouyang/alignment-attribution-code/out/Addition:6/llama2-7b-chat-hf/direct,cot0shot/eval_all/addition_bottom_0.000000_direct,cot0shot_all_prompt_cot0shot.jsonl"
)
ORIG_DIRECT_FILE = (
    "/common/users/sl2148/Public/yang_ouyang/alignment-attribution-code/out/Addition:6/llama2-7b-chat-hf/direct,cot0shot/eval_all/addition_bottom_0.000000_direct,cot0shot_all_prompt_direct.jsonl"
)

PQ_DIR = ("/common/users/sl2148/Public/yang_ouyang/alignment-attribution-code/out/Addition:6/llama2-7b-chat-hf/unstructured/wanda_3_set_difference_utility_weightonly/wanda_3_set_difference_cot0shot/eval_all/prompt_direct,cot0shot/step_0.01_sp_2e-07_k_0.01/")
PQKU_RE = re.compile(r"pq_([0-9.]+)_([0-9.]+)_k_([0-9.]+)_u_([0-9.]+)")

def parse_pqku_from_dir(dir_path: str) -> Optional[Dict[str, float]]:
    """
    从目录名解析 p,q,k,u。目录末级名需形如 pq_<p>_<q>_k_<k>_u_<u>
    """
    base = os.path.basename(os.path.normpath(dir_path))
    m = PQKU_RE.fullmatch(base)
    if not m:
        # 兼容偶尔多一层路径前缀，用 search 再试一次
        m = PQKU_RE.search(base)
        if not m:
            return None
    p, q, k, u = m.groups()
    return {"p": float(p), "q": float(q), "k": float(k), "u": float(u)}


def discover_sparsity_ratios(base_dir):
    """
    扫描 base_dir 下的 pq_* 目录，读取其中 *.jsonl 文件名里的
    (gsm8k|addition)_bottom_<ratio>_ 片段，建立:
        ratio -> {p,q,k,u, dir, _alts?}
    若同一个 ratio 出现多组 (p,q,k,u)，首次出现作为主项，其它收集到 _alts 列表。
    """
    dirs = glob.glob(os.path.join(base_dir, "pq_*_*_k_*_u_*/"))
    ratio_to_meta = {}

    for d in dirs:
        pqku = parse_pqku_from_dir(d)
        if not pqku:
            continue

        files = glob.glob(os.path.join(d, "*.jsonl"))
        for f in files:
            m = re.search(r"(?:gsm8k|addition)_bottom_([0-9.]+)_",
                          os.path.basename(f), flags=re.IGNORECASE)
            if not m:
                continue
            ratio = float(m.group(1))
            if ratio <= 0:
                continue

            # 记录该 ratio 的 meta
            cur_meta = {**pqku, "dir": d}
            if ratio not in ratio_to_meta:
                ratio_to_meta[ratio] = cur_meta
            else:
                # 若已有且参数不同，作为备选记录
                exist = ratio_to_meta[ratio]
                if any(exist.get(k) != pqku.get(k) for k in ("p", "q", "k", "u")):
                    ratio_to_meta[ratio].setdefault("_alts", []).append(cur_meta)

    # 返回按 ratio 升序的映射
    return dict(sorted(ratio_to_meta.items(), key=lambda kv: kv[0]))


# 建立全局映射：ratio -> {p,q,k,u,dir,...}
ratio_to_meta_map = discover_sparsity_ratios(PQ_DIR)



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

def match_samples_by_id(data1: List[Dict], data2: List[Dict]) -> Dict[str, str]:
    # Directly match by id
    id_set1 = set(item.get('id', '') for item in data1)
    matches = {}
    for item in data2:
        sid = item.get('id', '')
        if sid in id_set1:
            matches[sid] = sid
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

def discover_sparsity_dirs(base_dir: str) -> Dict[float, Dict]:
    """Return {sparsity_ratio: {dir, p,q,k,u}}."""
    mapping: Dict[float, Dict] = {}
    files = glob.glob(os.path.join(base_dir, "*.jsonl"))
    for file in files:
        if not file:
            continue
        # Support both GSM8K and Addition filename prefixes
        fname = os.path.basename(file)
        print("[DEBUG] fname:", fname)
        m = re.search(r"(?:.*)_bottom_([0-9.]+)_", fname, flags=re.IGNORECASE)
        if not m:
            continue
        ratio = float(m.group(1))
        if ratio <= 0:
            continue
        print(f"[DEBUG] Found ratio: {ratio} in file: {file}")
        meta = ratio_to_meta_map.get(ratio, {"p": None, "q": None, "k": None, "u": None})
        mapping[ratio] = meta
    return dict(sorted(mapping.items()))

# --- Collection ---
def collect_all_results_with_meta():
    # Load originals
    cot4_data = load_jsonl(ORIG_COT4_FILE)
    dir_data = load_jsonl(ORIG_DIRECT_FILE)
    cot4_info = extract_sample_info(cot4_data)
    dir_info = extract_sample_info(dir_data)
    matches = match_samples_by_id(cot4_data, dir_data)

    print(f"[DEBUG] Baseline cot4_data: {len(cot4_data)}, dir_data: {len(dir_data)}")
    print(f"[DEBUG] Baseline cot4_info: {len(cot4_info)}, dir_info: {len(dir_info)}")
    print(f"[DEBUG] matches: {len(matches)}")

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
    print(f"Discovered {len(ratio_to_meta)} sparsity dirs under BASE_DIR={BASE_DIR}")
    for rr, mm in ratio_to_meta.items():
        print(f"  ratio={rr} -> dir={mm.get('dir')} meta(p,q,k,u)={(mm.get('p'), mm.get('q'), mm.get('k'), mm.get('u'))}")

    for ratio, meta in ratio_to_meta.items():
        d = meta["dir"]
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


def _format_float_for_tag(x: Optional[float]) -> str:
    if x is None:
        return "-"
    xi = float(x)
    if xi.is_integer():
        return str(int(xi))
    return str(xi)


def plot_accuracy_by_pq_for_each_k(
    K_OPTIONS, U_OPTIONS, categories, colors, OUTPUT_DIR,
    collect_all_results_with_meta, _format_float_for_tag
):
    """Plot pruned accuracy (not delta) with x-axis labeled by (p,q) pairs instead of sparsity ratio.

    For each fixed (k,u), collect all discovered directories (wildcard pq) and:
      - Build a unique list of (p,q) pairs (exclude the baseline 0.0 entry which has p=q=None)
      - Sort by p then q numerically
      - (Optional) limit to MAX_POINTS
      - Plot pruned_accuracy for each category vs enumerated index of (p,q) and label ticks as '(p,q)'
      - Also draw the original (unpruned) accuracy as a horizontal dashed line per category for reference.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for k in K_OPTIONS:
        for u in U_OPTIONS:
            global FILE_NAME
            FILE_NAME = f"pq_*_*_k_{k}_u_{u}/"
            print(f"[Collect-PQ-X] pattern={FILE_NAME}")
            all_results = collect_all_results_with_meta()

            # Extract (p,q) entries (skip ratio 0.0 baseline with None p/q)
            pq_entries = []  # list of dict: { 'p':p, 'q':q, 'ratio':r, 'metrics':metrics }
            for ratio, bundle in all_results.items():
                meta = bundle.get('meta', {})
                p_val, q_val = meta.get('p'), meta.get('q')
                if p_val is None or q_val is None:
                    continue
                pq_entries.append({
                    'p': p_val,
                    'q': q_val,
                    'ratio': ratio,
                    'metrics': bundle.get('metrics', {})
                })

            # Remove potential duplicate (p,q) keeping the FIRST occurrence (order after sort below)
            # First sort by p then q
            pq_entries.sort(key=lambda x: (x['p'], x['q']))
            unique = []
            seen = set()
            for ent in pq_entries:
                key = (ent['p'], ent['q'])
                if key in seen:
                    continue
                seen.add(key)
                unique.append(ent)
            pq_entries = unique

            if MAX_POINTS is not None and len(pq_entries) > MAX_POINTS:
                pq_entries = pq_entries[:MAX_POINTS]

            if not pq_entries:
                print(f"[Skip] k={k} u={u} no (p,q) entries found.")
                continue

            # Build accuracy series per category (raw pruned accuracy)
            raw_acc = {cat: [] for cat in categories}
            for ent in pq_entries:
                for cat in categories:
                    raw_acc[cat].append(ent['metrics'].get(cat, {}).get('pruned_accuracy', 0.0))

            # Baseline original accuracies
            base_metrics = all_results.get(0.0, {}).get('metrics', {})
            baseline_acc = {cat: base_metrics.get(cat, {}).get('original_accuracy', 0.0) for cat in categories}

            if NORMALIZE_BASELINE:
                # Normalize each category by its baseline; insert baseline point (0,1)
                acc_series = {cat: [ (val / baseline_acc[cat]) if baseline_acc[cat] > 0 else 0.0 for val in raw_acc[cat]] for cat in categories}
                for cat in categories:
                    acc_series[cat].insert(0, 1.0)
                x_idx = list(range(len(pq_entries) + 1))
                tick_labels = ["0"] + [f"({ _format_float_for_tag(ent['p']) },{ _format_float_for_tag(ent['q']) })" for ent in pq_entries]
                y_label = 'Accuracy'
                title = f'Accuracy vs (p,q)  k={_format_float_for_tag(k)}, u={_format_float_for_tag(u)}'
                y_min, y_max = 0.0, 1.05
                # If normalization pushes some >1, adjust upper
                max_val = max(max(vals) for vals in acc_series.values()) if acc_series else 1.0
                if max_val > 1.05:
                    y_max = min(1.25, max_val * 1.05)
            else:
                acc_series = raw_acc
                x_idx = list(range(len(pq_entries)))
                tick_labels = [f"({ _format_float_for_tag(ent['p']) },{ _format_float_for_tag(ent['q']) })" for ent in pq_entries]
                y_label = 'Accuracy'
                title = f'Accuracy vs (p,q)  k={_format_float_for_tag(k)}, u={_format_float_for_tag(u)}'
                y_min, y_max = 0.0, 1.05

            # Determine a narrower width with clamping
            desired_width = FIG_WIDTH_PER_LABEL * len(tick_labels)
            fig_width = max(FIG_MIN_WIDTH, min(FIG_MAX_WIDTH, desired_width))
            fig, ax = plt.subplots(figsize=(fig_width, 4))
            for i, cat in enumerate(categories):
                color_i = colors[i % len(colors)] if colors else None
                ax.plot(x_idx, acc_series[cat], 'o-', label=f"{cat}", color=color_i)
                if not NORMALIZE_BASELINE:
                    # baseline horizontal line only for raw accuracy
                    ax.hlines(baseline_acc[cat], xmin=-0.2, xmax=len(x_idx)-0.8, colors=color_i, linestyles='dashed', alpha=0.4)
                else:
                    # reference line at y=1
                    ax.axhline(1.0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)

            # Annotate sparsity ratio above points of a chosen category series (to avoid clutter)
            if ANNOTATE_SPARSITY and pq_entries:
                anchor_cat = categories[ANNOTATE_CATEGORY_INDEX % len(categories)]
                series_vals = acc_series[anchor_cat]
                # series_vals may include inserted baseline at index 0 if NORMALIZE_BASELINE
                start_idx = 1 if NORMALIZE_BASELINE else 0
                for local_i, ent in enumerate(pq_entries, start=start_idx):
                    if local_i >= len(series_vals):
                        break
                    ratio_str = f"{ent['ratio']:.6f}".rstrip('0').rstrip('.')
                    ax.annotate(
                        ratio_str,
                        xy=(local_i, series_vals[local_i]),
                        xytext=(0, ANNOTATION_Y_OFFSET),
                        textcoords='offset points',
                        ha='center', va='bottom',
                        fontsize=ANNOTATION_FONTSIZE,
                        rotation=ANNOTATION_ROTATION,
                        color='dimgray',
                        alpha=0.7
                    )
            # Potentially thin x tick labels to avoid overcrowding
            display_idx = x_idx
            display_labels = tick_labels
            if len(tick_labels) > TICK_THIN_THRESHOLD_2:
                step = 3
            elif len(tick_labels) > TICK_THIN_THRESHOLD_1:
                step = 2
            else:
                step = 1
            if step > 1:
                display_idx = [i for i in x_idx if i % step == 0]
                display_labels = [tick_labels[i] for i in display_idx]
            ax.set_xticks(display_idx)
            ax.set_xticklabels(display_labels, rotation=45, ha='right', fontsize=7 if step>1 else 8)
            # Minor note: all points still plotted; only labels thinned
            ax.set_ylabel(y_label)
            ax.set_xlabel('pq pairs')
            ax.set_title(title)
            ax.set_ylim(y_min, y_max)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
            fig.tight_layout()
            out_path = os.path.join(
                OUTPUT_DIR,
                f'random_accuracy_by_pq_k_{_format_float_for_tag(k)}_u_{_format_float_for_tag(u)}.png'
            )
            fig.savefig(out_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"[Saved] {out_path}")
            print(
                "  points={pts}  (p,q) range p:[{pmin},{pmax}] q:[{qmin},{qmax}]".format(
                    pts=len(pq_entries),
                    pmin=_format_float_for_tag(pq_entries[0]['p']),
                    pmax=_format_float_for_tag(pq_entries[-1]['p']),
                    qmin=_format_float_for_tag(pq_entries[0]['q']),
                    qmax=_format_float_for_tag(pq_entries[-1]['q'])
                )
            )

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # New plot: accuracy vs (p,q)
    plot_accuracy_by_pq_for_each_k(
        K_OPTIONS=K_OPTIONS,
        U_OPTIONS=U_OPTIONS,
        categories=CATEGORIES,
        colors=CATEGORY_COLORS,
        OUTPUT_DIR=OUTPUT_DIR,
        collect_all_results_with_meta=collect_all_results_with_meta,
        _format_float_for_tag=_format_float_for_tag
    )


if __name__ == '__main__':
    main()
