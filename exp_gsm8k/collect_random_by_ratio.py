#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import re
import glob
from typing import Dict, List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt

# ----------------- 可调开关 -----------------
MAX_POINTS = 40
NORMALIZE_BASELINE = True   # 为 True 时在 x=0 插入基线点 y=1
ANNOTATE_SPARSITY = True
ANNOTATION_FONTSIZE = 6
ANNOTATION_ROTATION = 90
ANNOTATION_Y_OFFSET = 4

FIG_MIN_WIDTH = 6
FIG_MAX_WIDTH = 12
FIG_WIDTH_PER_LABEL = 0.2
TICK_THIN_THRESHOLD_1 = 25
TICK_THIN_THRESHOLD_2 = 40

# ----------------- 固定配置（按需修改） -----------------
model = "Qwen2.5-7B-Instruct"
eval_dataset = "GSM8K"
eval_type = "selected_samples"
add_template_flag = False
use_template = True if add_template_flag else False
sparsity_threshold = 0.0000002
set_difference_data = "GSM8K"
sparsity_type = "unstructured"
suffix = "weightonly"
prune_method = "wanda_234_set_difference"
prompt_method = "direct,cot0shot"
number_of_sets = 3

BASIC_DIR = "/common/users/sl2148/Public/yang_ouyang/alignment-attribution-code/"
DISCOVER_DIR = os.path.join(
    BASIC_DIR,
    f"out/eval_{eval_dataset}/{model}/{sparsity_type}/{prune_method}_{suffix}/sets_{number_of_sets}/set_difference_data_{set_difference_data}/eval_{eval_type}/prompt_{prompt_method}/add_template_{use_template}/step_0.01_sp_{sparsity_threshold}/"
)
ORIG_COT4_FILE = os.path.join(
    BASIC_DIR, f"out/eval_{eval_dataset}/{model}/full_model/eval_all/prompt_cot0shot/add_template_{use_template}/gsm8k_bottom_0.000000_alpaca_cleaned_no_safety_seed_0.jsonl"
)
ORIG_DIRECT_FILE = os.path.join(
    BASIC_DIR, f"out/eval_{eval_dataset}/{model}/full_model/eval_all/prompt_direct/add_template_{use_template}/gsm8k_bottom_0.000000_alpaca_cleaned_no_safety_seed_0.jsonl"
)
OUTPUT_DIR = os.path.join(DISCOVER_DIR, "figures")

CATEGORIES = ["direct_success_cot_success", "direct_fail_cot_success", "direct_success_cot_success_cot_prompt"]
CATEGORY_COLORS = [
    (200/255, 178/255, 191/255),
    (127/255, 143/255, 172/255),
    (111/255, 148/255, 135/255)
]

# ----------------- IO helpers -----------------
def load_jsonl(file_path: str) -> List[Dict]:
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def extract_sample_info(data: List[Dict]) -> Dict[str, Dict]:
    json_file = os.path.join(
        BASIC_DIR, "data/GSM8K_eval_build", model, "eval_datasets/selected_samples.jsonl"
    )
    data_new = load_jsonl(json_file)
    data_id = set(item.get('id', '') for item in data_new)
    sample_info = {}
    for item in data:
        sid = item.get('id', '')
        if sid in data_id:
            sample_info[sid] = {
                'correct': item.get('correct', False),
                'question': item.get('question', ''),
                'pred': item.get('pred', ''),
                'gold': item.get('gold', '')
            }
    return sample_info

def match_samples_by_id(data1: List[Dict], data2: List[Dict]) -> Dict[str, str]:
    id_set1 = set(item.get('id', '') for item in data1)
    matches = {}
    for item in data2:
        sid = item.get('id', '')
        if sid in id_set1:
            matches[sid] = sid
    return matches

# ----------------- 分类与评估 -----------------
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
    return {
        'category': category,
        'pruned_correct_count': pruned_correct,
        'original_correct': int(round(base_acc * total)),
        'original_accuracy': base_acc,
        'pruned_correct': pruned_correct,
        'pruned_accuracy': pruned_acc,
        'accuracy_change': pruned_acc - base_acc,
    }

# ----------------- 仅依赖“当前目录”收集稀疏率 -----------------
def discover_ratios_in_current_dir(base_dir: str) -> Dict[float, Dict[str, str]]:
    """
    只在 base_dir 目录本身收集文件，解析文件名中的 ..._bottom_<ratio>_...
    返回 { ratio: {"cot": cot_file, "dir": direct_file} }（缺失则不返回该 ratio）
    """
    files = glob.glob(os.path.join(base_dir, "*.jsonl"))
    buckets: Dict[float, Dict[str, str]] = {}
    for f in files:
        fname = os.path.basename(f)
        m = re.search(r"_bottom_([0-9.]+)_", fname, flags=re.IGNORECASE)
        if not m:
            continue
        ratio = float(m.group(1))
        if ratio < 0:
            continue
        is_cot = "prompt_cot0shot" in fname
        is_dir = "prompt_direct" in fname
        if not (is_cot or is_dir):
            continue
        buckets.setdefault(ratio, {})
        if is_cot:
            buckets[ratio]["cot"] = f
        if is_dir:
            buckets[ratio]["dir"] = f

    # 仅保留“cot 与 direct 文件均存在”的 ratio
    cleaned = {r: pair for r, pair in buckets.items() if "cot" in pair and "dir" in pair}
    # 升序
    return dict(sorted(cleaned.items(), key=lambda kv: kv[0]))

def collect_all_results_from_current_dir():
    # 加载 full model baseline
    cot4_data = load_jsonl(ORIG_COT4_FILE)
    dir_data = load_jsonl(ORIG_DIRECT_FILE)
    cot4_info = extract_sample_info(cot4_data)
    dir_info = extract_sample_info(dir_data)
    matches = match_samples_by_id(cot4_data, dir_data)

    d_s_c_s, _, d_f_c_s, _ = categorize_samples(cot4_info, dir_info, matches)
    category_samples = {
        "direct_success_cot_success": d_s_c_s,
        "direct_fail_cot_success": d_f_c_s,
        "direct_success_cot_success_cot_prompt": d_s_c_s,
    }

    all_results: Dict[float, Dict] = {}

    # 0.0 基线
    orig_bucket = {}
    for name, samples in category_samples.items():
        orig_bucket[name] = analyze_category_performance_original(
            name, samples, cot4_info, dir_info, matches
        )
    all_results[0.0] = {"metrics": orig_bucket}

    # 收集当前目录下的各个 ratio
    ratio_pairs = discover_ratios_in_current_dir(DISCOVER_DIR)
    print(f"[INFO] Found {len(ratio_pairs)} ratios in {DISCOVER_DIR}")

    count = 0
    for ratio, pair in ratio_pairs.items():
        if MAX_POINTS is not None and count >= MAX_POINTS:
            break
        pruned_cot_info = extract_sample_info(load_jsonl(pair["cot"]))
        pruned_dir_info = extract_sample_info(load_jsonl(pair["dir"]))
        bucket = {}
        for name, samples in category_samples.items():
            bucket[name] = analyze_category_performance_pruned(
                name, samples, cot4_info, dir_info, pruned_cot_info, pruned_dir_info, matches
            )
        all_results[ratio] = {"metrics": bucket}
        count += 1

    return dict(sorted(all_results.items(), key=lambda kv: kv[0]))

def _fmt_ratio_tag(x: float) -> str:
    s = f"{x:.6f}"
    s = s.rstrip('0').rstrip('.')
    return s if s else "0"

# ----------------- 绘图：x 轴为稀疏率 -----------------
def plot_accuracy_vs_sparsity(categories, colors):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results = collect_all_results_from_current_dir()
    if not results:
        print("[WARN] No results to plot.")
        return

    # 构造 x 轴（ratio），以及每个类别的准确率序列
    ratios = [r for r in results.keys() if r != 0.0]
    ratios.sort()
    raw_acc = {cat: [] for cat in categories}
    for r in ratios:
        metrics = results[r].get("metrics", {})
        for cat in categories:
            raw_acc[cat].append(metrics.get(cat, {}).get('pruned_accuracy', 0.0))

    base_metrics = results.get(0.0, {}).get("metrics", {})
    baseline_acc = {cat: base_metrics.get(cat, {}).get('original_accuracy', 0.0) for cat in categories}

    if NORMALIZE_BASELINE:
        acc_series = {
            cat: [ (val / baseline_acc[cat]) if baseline_acc[cat] > 0 else 0.0 for val in raw_acc[cat] ]
            for cat in categories
        }
        # 在开头插入基线点 1.0
        for cat in categories:
            acc_series[cat].insert(0, 1.0)
        x_vals = [0.0] + ratios
        y_min, y_max = 0.0, 1.05
        max_val = max(max(vals) for vals in acc_series.values()) if acc_series else 1.0
        if max_val > 1.05:
            y_max = min(1.25, max_val * 1.05)
    else:
        acc_series = raw_acc
        x_vals = ratios
        y_min, y_max = 0.0, 1.05

    tick_labels = [_fmt_ratio_tag(r) for r in x_vals]

    desired_width = FIG_WIDTH_PER_LABEL * len(tick_labels)
    fig_width = max(FIG_MIN_WIDTH, min(FIG_MAX_WIDTH, desired_width))

    fig, ax = plt.subplots(figsize=(fig_width, 4))
    for i, cat in enumerate(categories):
        color_i = colors[i % len(colors)] if colors else None
        ax.plot(range(len(x_vals)), acc_series[cat], 'o-', label=f"{cat}", color=color_i)
        if NORMALIZE_BASELINE:
            ax.axhline(1.0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
        else:
            ax.hlines(baseline_acc[cat], xmin=-0.2, xmax=len(x_vals)-0.8, colors=color_i, linestyles='dashed', alpha=0.4)

    # 标注稀疏率（避免过密时只标部分）
    if ANNOTATE_SPARSITY and len(x_vals) > 0:
        # 以第一个类别作锚点
        anchor_cat = categories[0]
        series_vals = acc_series[anchor_cat]
        start_idx = 0  # 包含 0 点
        for idx in range(start_idx, len(x_vals)):
            ax.annotate(
                tick_labels[idx],
                xy=(idx, series_vals[idx]),
                xytext=(0, ANNOTATION_Y_OFFSET),
                textcoords='offset points',
                ha='center', va='bottom',
                fontsize=ANNOTATION_FONTSIZE,
                rotation=ANNOTATION_ROTATION,
                color='dimgray',
                alpha=0.7
            )

    # 稀疏 x 轴刻度文字
    if len(tick_labels) > TICK_THIN_THRESHOLD_2:
        step = 3
    elif len(tick_labels) > TICK_THIN_THRESHOLD_1:
        step = 2
    else:
        step = 1
    disp_idx = [i for i in range(len(x_vals)) if i % step == 0]
    disp_labels = [tick_labels[i] for i in disp_idx]

    ax.set_xticks(disp_idx)
    ax.set_xticklabels(disp_labels, rotation=45, ha='right', fontsize=7 if step > 1 else 8)
    ax.set_ylabel('Accuracy' if NORMALIZE_BASELINE else 'Accuracy')
    ax.set_xlabel('sparsity ratio')
    ax.set_title('Accuracy vs sparsity ratio (current folder only)')
    ax.set_ylim(y_min, y_max)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, 'accuracy_vs_sparsity_current_dir.png')
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[Saved] {out_path}")
    print(f"  points={len(x_vals)}  ratios=[{tick_labels[0]}..{tick_labels[-1]}]")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plot_accuracy_vs_sparsity(CATEGORIES, CATEGORY_COLORS)

if __name__ == '__main__':
    main()
