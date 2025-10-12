#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot accuracy vs sparsity ratio (no p/q/k/u needed).

This script:
- Scans DISCOVER_DIR for subdirs like: sparsity_ratio_{ratio}/
- In each subdir, looks for the pruned results:
    * gsm8k_bottom_{ratio}_direct,cot0shot_selected_samples_prompt_cot0shot.json
      (CoT prompt file)
    * *_prompt_direct.json  (Direct prompt file; optional but recommended)
- Compares against ORIGINAL full-model baselines (jsonl) to compute per-category
  accuracy and (optionally) normalized accuracy.
- Produces a figure under DISCOVER_DIR/figures/.

Notes:
- No pqku parsing. Purely keyed by sparsity_ratio.
- Supports .json or .jsonl for pruned files (auto-detected by loader).
"""

import json
import os
import re
import glob
from typing import Dict, List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt

model = "llama2-7b-chat-hf" # "Qwen2.5-7B-Instruct"  # "llama2-7b-chat-hf"
eval_dataset = "GSM8K"
eval_type = "selected_samples"
use_template = True  # 与原始基线文件路径一致
set_difference_data = "GSM8K"
sparsity_type = "unstructured"
suffix = "weightonly"
prune_method = "wanda_ratio_diff" #"wanda_ratio_diff"  
prompt_method = "direct,cot0shot"
number_of_sets = 2

# 项目根目录（你给的 BASIC_DIR）
BASIC_DIR = "/common/users/sl2148/Public/yang_ouyang/alignment-attribution-code/"

if prune_method == "random":
    DISCOVER_DIR = os.path.join(
    BASIC_DIR,
    f"out/eval_{eval_dataset}/{model}/{sparsity_type}/{prune_method}_{suffix}/"
    f"sets_{number_of_sets}/"
    f"eval_{eval_type}/prompt_{prompt_method}/add_template_{use_template}/"
)
else: # 只需处理这个目录（里面有多个 sparsity_ratio_xxx 子目录）
    DISCOVER_DIR = os.path.join(
        BASIC_DIR,
        f"out/eval_{eval_dataset}/{model}/{sparsity_type}/{prune_method}_{suffix}/"
        f"sets_{number_of_sets}/set_difference_data_{set_difference_data}/"
        f"eval_{eval_type}/prompt_{prompt_method}/add_template_{use_template}/"
    )

# 基线：full_model（不依赖 pruned 子目录）
ORIG_COT4_FILE = os.path.join(
    BASIC_DIR,
    f"out/eval_{eval_dataset}/{model}/full_model/eval_all/"
    f"prompt_cot0shot/add_template_{use_template}/"
    "gsm8k_bottom_0.000000_alpaca_cleaned_no_safety_seed_0.jsonl",
)
ORIG_DIRECT_FILE = os.path.join(
    BASIC_DIR,
    f"out/eval_{eval_dataset}/{model}/full_model/eval_all/"
    f"prompt_direct/add_template_{use_template}/"
    "gsm8k_bottom_0.000000_alpaca_cleaned_no_safety_seed_0.jsonl",
)

# 仅选用 selected_samples 的映射（与原脚本一致；如路径不同请自行调整）
SELECTED_SAMPLES_FILE = os.path.join(
    BASIC_DIR,
    f"data/{eval_dataset}_eval_build/{model}/eval_datasets/selected_samples.jsonl"
)

# 输出图目录
OUTPUT_DIR = os.path.join(DISCOVER_DIR, "figures")
MAX_POINTS = 10
# 可视化与控制
NORMALIZE_BASELINE = True
ANNOTATE_SPARSITY = True          # 在点上方标注 ratio
ANNOTATION_FONTSIZE = 7
ANNOTATION_ROTATION = 90
ANNOTATION_Y_OFFSET = 4
FIG_MIN_WIDTH = 6
FIG_MAX_WIDTH = 14
FIG_WIDTH_PER_LABEL = 0.28

# 类别与颜色（与原脚本一致）
CATEGORIES = [
    "direct_success_cot_success",
    "direct_fail_cot_success",
    "direct_success_cot_success_cot_prompt",
]
CATEGORY_COLORS = [
    (200/255, 178/255, 191/255),  # red-ish
    (127/255, 143/255, 172/255),  # blue-ish
    (111/255, 148/255, 135/255),  # green-ish
]
# ============================================================


# ---------------- IO helpers ----------------
def _load_json_or_jsonl(file_path: str) -> List[Dict]:
    data = []
    if not os.path.isfile(file_path):
        return data
    if file_path.endswith(".jsonl"):
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
    elif file_path.endswith(".json"):
        with open(file_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
            # 兼容两种形态：文件本身是一行一个对象的列表；或是字典里装着列表字段
            if isinstance(obj, list):
                data = obj
            elif isinstance(obj, dict):
                # 尝试在常见键下找记录
                for k in ["records", "data", "items", "list", "rows"]:
                    if k in obj and isinstance(obj[k], list):
                        data = obj[k]
                        break
                if not data:
                    # 最后兜底：把整个 dict 当成一条
                    data = [obj]
    else:
        # 后缀未知，按 jsonl 读法尝试
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data.append(json.loads(line))
                    except Exception:
                        pass
    return data


def load_jsonl(file_path: str) -> List[Dict]:
    return _load_json_or_jsonl(file_path)


def extract_sample_info(data: List[Dict]) -> Dict[str, Dict]:
    """仅保留 selected_samples.jsonl 中存在的 id；并抽取所需字段。"""
    selected = load_jsonl(SELECTED_SAMPLES_FILE)
    selected_ids = set(item.get("id", "") for item in selected)
    sample_info = {}
    for item in data:
        sid = item.get("id", "")
        if sid in selected_ids:
            sample_info[sid] = {
                "correct": item.get("correct", False),
                "question": item.get("question", ""),
                "pred": item.get("pred", ""),
                "gold": item.get("gold", ""),
            }
    return sample_info


def match_samples_by_id(data1: List[Dict], data2: List[Dict]) -> Dict[str, str]:
    print(f"[Info] Matching samples by 'id' field between two datasets")
    """直接用 id 做一一匹配。"""
    id_set1 = set(item.get("id", "") for item in data1)
    matches = {}
    for item in data2:
        sid = item.get("id", "")
        if sid in id_set1:
            matches[sid] = sid
    return matches


# --------------- 分类与分析 -----------------
def categorize_samples(cot_info: Dict[str, Dict], direct_info: Dict[str, Dict], matches: Dict[str, str]):
    d_s_c_s, d_s_c_f, d_f_c_s, d_f_c_f = [], [], [], []
    for cot_id, direct_id in matches.items():
        cot_ok = cot_info.get(cot_id, {}).get("correct", False)
        dir_ok = direct_info.get(direct_id, {}).get("correct", False)
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
    """计算原始(full model)在该类别上的 accuracy。"""
    if not sample_cot_ids:
        return {
            "category": category,
            "original_correct": 0,
            "original_accuracy": 0.0,
            "pruned_correct": 0,
            "pruned_accuracy": 0.0,
            "accuracy_change": 0.0,
        }
    total = len(sample_cot_ids)
    orig_correct = 0
    if category == "direct_success_cot_success":
        for cot_id in sample_cot_ids:
            dir_id = matches.get(cot_id)
            if dir_id and orig_direct.get(dir_id, {}).get("correct", False):
                orig_correct += 1
    elif category == "direct_fail_cot_success":
        for cot_id in sample_cot_ids:
            if orig_cot.get(cot_id, {}).get("correct", False):
                orig_correct += 1
    elif category == "direct_success_cot_success_cot_prompt":
        for cot_id in sample_cot_ids:
            dir_id = matches.get(cot_id)
            if dir_id and orig_cot.get(dir_id, {}).get("correct", False):
                orig_correct += 1
    acc = orig_correct / total if total else 0.0
    return {
        "category": category,
        "original_correct": orig_correct,
        "original_accuracy": acc,
        "pruned_correct": orig_correct,
        "pruned_accuracy": acc,
        "accuracy_change": 0.0,
    }


def analyze_category_performance_pruned(category: str, sample_cot_ids, orig_cot, orig_direct,
                                        pruned_cot, pruned_direct, matches):
    """计算裁剪后在该类别上的 accuracy，并和原始比较。"""
    if not sample_cot_ids:
        return {
            "category": category,
            "original_correct": 0,
            "original_accuracy": 0.0,
            "pruned_correct": 0,
            "pruned_accuracy": 0.0,
            "accuracy_change": 0.0,
        }
    total = len(sample_cot_ids)
    base = analyze_category_performance_original(category, sample_cot_ids, orig_cot, orig_direct, matches)
    base_acc = base["original_accuracy"]
    pruned_correct = 0
    if category == "direct_success_cot_success":
        for cot_id in sample_cot_ids:
            dir_id = matches.get(cot_id)
            if dir_id and pruned_direct.get(dir_id, {}).get("correct", False):
                pruned_correct += 1
    elif category == "direct_fail_cot_success":
        for cot_id in sample_cot_ids:
            if pruned_cot.get(cot_id, {}).get("correct", False):
                pruned_correct += 1
    elif category == "direct_success_cot_success_cot_prompt":
        for cot_id in sample_cot_ids:
            dir_id = matches.get(cot_id)
            if dir_id and pruned_cot.get(dir_id, {}).get("correct", False):
                pruned_correct += 1
    pruned_acc = pruned_correct / total if total else 0.0
    print(f"[Category] {category} | total={total} | base_correct={base['original_correct']} | pruned_correct={pruned_correct}")
    return {
        "category": category,
        "original_correct": int(round(base_acc * total)),
        "original_accuracy": base_acc,
        "pruned_correct": pruned_correct,
        "pruned_accuracy": pruned_acc,
        "accuracy_change": pruned_acc - base_acc,
    }


# --------------- 发现 sparsity 结果 -----------------
def _parse_ratio_from_dirname(name: str) -> Optional[float]:
    """
    从子目录名解析 ratio：期望形如 'sparsity_ratio_0.1'。
    """
    m = re.fullmatch(r"sparsity_ratio_([0-9.]+)", name)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def discover_by_sparsity(discover_dir: str) -> Dict[float, Dict]:
    """
    扫描 DISCOVER_DIR 下的 'sparsity_ratio_*' 子目录，找到其中的 pruned 结果文件。
    返回: { ratio: { 'dir': subdir, 'cot_file': <path or None>, 'direct_file': <path or None> } }
    支持 .json / .jsonl
    """
    mapping: Dict[float, Dict] = {}
    if not os.path.isdir(discover_dir):
        return mapping

    for entry in sorted(os.listdir(discover_dir)):
        sub = os.path.join(discover_dir, entry)
        if not os.path.isdir(sub):
            continue
        ratio = _parse_ratio_from_dirname(entry)
        if ratio is None:
            continue
     

        # 目标文件名模式（你的示例）
        # gsm8k_bottom_{ratio}_direct,cot0shot_selected_samples_prompt_cot0shot.json
        # 也兼容 .jsonl
        cot_pattern_json = os.path.join(sub, f"gsm8k_bottom_{ratio}_direct,cot0shot_selected_samples_prompt_cot0shot.json")
        cot_pattern_jsonl = cot_pattern_json + "l"  # .jsonl
        # direct 文件：同目录下任意 '*_prompt_direct.json(.l)' 都可以
        direct_candidates = sorted(
            glob.glob(os.path.join(sub, "*prompt_direct.json"))
            + glob.glob(os.path.join(sub, "*prompt_direct.jsonl"))
        )

        cot_file = None
        if os.path.isfile(cot_pattern_json):
            cot_file = cot_pattern_json
        elif os.path.isfile(cot_pattern_jsonl):
            cot_file = cot_pattern_jsonl
        else:
            # 兜底：找任何包含 'prompt_cot0shot' 的文件
            cand_cot = sorted(
                glob.glob(os.path.join(sub, "*prompt_cot0shot.json"))
                + glob.glob(os.path.join(sub, "*prompt_cot0shot.jsonl"))
            )
            if cand_cot:
                cot_file = cand_cot[0]

        direct_file = direct_candidates[0] if direct_candidates else None

        # 只要拿到任一文件就记录（cot 或 direct 至少有一个）
        if cot_file or direct_file:
            mapping[ratio] = {
                "dir": sub,
                "cot_file": cot_file,
                "direct_file": direct_file,
            }
            print(f"[Found] ratio={ratio:.6f} | cot={bool(cot_file)} | direct={bool(direct_file)} | dir={entry}")

    return dict(sorted(mapping.items(), key=lambda kv: kv[0]))


# --------------- 汇总与作图 -----------------
def collect_all_results():
    # 加载 baseline
    cot4_data = load_jsonl(ORIG_COT4_FILE)
    dir_data = load_jsonl(ORIG_DIRECT_FILE)
    cot4_info = extract_sample_info(cot4_data)
    dir_info = extract_sample_info(dir_data)
    print(f"[Baseline] Loaded {len(cot4_data)} CoT and {len(dir_data)} Direct samples")
    matches = match_samples_by_id(cot4_data, dir_data)

    print(f"[Baseline] cot4_data={len(cot4_data)} | dir_data={len(dir_data)}")
    print(f"[Baseline] cot4_info={len(cot4_info)} | dir_info={len(dir_info)} | matches={len(matches)}")

    d_s_c_s, _, d_f_c_s, _ = categorize_samples(cot4_info, dir_info, matches)
    category_samples = {
        "direct_success_cot_success": d_s_c_s,
        "direct_fail_cot_success": d_f_c_s,
        "direct_success_cot_success_cot_prompt": d_s_c_s,
    }

    # 0.0 行（原始，不依赖 ratio）
    all_results: Dict[float, Dict] = {}
    orig_bucket = {}
    for name, samples in category_samples.items():
        orig_bucket[name] = analyze_category_performance_original(name, samples, cot4_info, dir_info, matches)
    all_results[0.0] = {"metrics": orig_bucket, "meta": {"dir": None}}

    # 各个 sparsity_ratio 的裁剪结果
    ratio_map = discover_by_sparsity(DISCOVER_DIR)
    print(f"[Discover] {len(ratio_map)} sparsity buckets under DISCOVER_DIR={DISCOVER_DIR}")

    for ratio, meta in ratio_map.items():
        cot_path = meta.get("cot_file")
        dir_path = meta.get("direct_file")
        if not cot_path and not dir_path:
            # 没有可用文件，跳过
            all_results[ratio] = {"metrics": {}, "meta": {"dir": meta.get("dir")}}
            continue

        pruned_cot_info = extract_sample_info(load_jsonl(cot_path)) if cot_path else {}
        pruned_dir_info = extract_sample_info(load_jsonl(dir_path)) if dir_path else {}

        bucket = {}
        for name, samples in category_samples.items():
            bucket[name] = analyze_category_performance_pruned(
                name, samples, cot4_info, dir_info, pruned_cot_info, pruned_dir_info, matches
            )
        all_results[ratio] = {"metrics": bucket, "meta": {"dir": meta.get("dir")}}

    return dict(sorted(all_results.items(), key=lambda kv: kv[0]))


def _format_ratio_tag(x: float) -> str:
    s = f"{x:.6f}"
    s = s.rstrip("0").rstrip(".")
    return s if s else "0"


def plot_accuracy_by_sparsity():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_results = collect_all_results()

    # x 轴：按 ratio 升序
    ratios = sorted([r for r in all_results.keys() if r > 0.0])
    if MAX_POINTS is not None and len(ratios) > MAX_POINTS:
        ratios = ratios[:MAX_POINTS]
    if not ratios:
        print("[Warn] No sparsity_ratio_* results found. Only baseline exists.")
        return

    # 取 baseline
    base_metrics = all_results.get(0.0, {}).get("metrics", {})
    baseline_acc = {cat: base_metrics.get(cat, {}).get("original_accuracy", 0.0) for cat in CATEGORIES}

    # 每个类别一条曲线（raw 或 normalized）
    series = {cat: [] for cat in CATEGORIES}
    for r in ratios:
        m = all_results[r].get("metrics", {})
        for cat in CATEGORIES:
            series[cat].append(m.get(cat, {}).get("pruned_accuracy", 0.0))

    if NORMALIZE_BASELINE:
        # 归一化，并在开头插入 baseline=1.0
        for cat in CATEGORIES:
            b = baseline_acc.get(cat, 0.0)
            series[cat] = [ (v / b) if b > 0 else 0.0 for v in series[cat] ]
            series[cat].insert(0, 1.0)
        x_idx = list(range(len(ratios) + 1))
        tick_labels = ["0"] + [_format_ratio_tag(r) for r in ratios]
        y_label = "Accuracy (normalized to baseline)"
    else:
        x_idx = list(range(len(ratios)))
        tick_labels = [_format_ratio_tag(r) for r in ratios]
        y_label = "Accuracy"

    # 图尺寸
    desired_width = FIG_WIDTH_PER_LABEL * len(tick_labels)
    fig_width = max(FIG_MIN_WIDTH, min(FIG_MAX_WIDTH, desired_width))
    fig, ax = plt.subplots(figsize=(fig_width, 4))

    # 画线
    for i, cat in enumerate(CATEGORIES):
        color_i = CATEGORY_COLORS[i % len(CATEGORY_COLORS)]
        ax.plot(x_idx, series[cat], "o-", label=cat, color=color_i)
        if NORMALIZE_BASELINE:
            ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
        else:
            ax.hlines(baseline_acc[cat], xmin=-0.2, xmax=len(x_idx)-0.8,
                      colors=color_i, linestyles="dashed", alpha=0.4)

    # 在点上方标注 sparsity ratio
    if ANNOTATE_SPARSITY:
        anchor_cat = CATEGORIES[0]
        vals = series[anchor_cat]
        start_idx = 1 if NORMALIZE_BASELINE else 0
        labels_for_points = (["0"] + [_format_ratio_tag(r) for r in ratios]) if NORMALIZE_BASELINE else [_format_ratio_tag(r) for r in ratios]
        for i_pt in range(start_idx, len(vals)):
            ax.annotate(
                labels_for_points[i_pt],
                xy=(i_pt, vals[i_pt]),
                xytext=(0, ANNOTATION_Y_OFFSET),
                textcoords="offset points",
                ha="center", va="bottom",
                fontsize=ANNOTATION_FONTSIZE,
                rotation=ANNOTATION_ROTATION,
                color="dimgray", alpha=0.75
            )

    ax.set_xticks(list(range(len(tick_labels))))
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(y_label)
    ax.set_xlabel("Sparsity ratio")
    ax.set_title(f"Accuracy vs Sparsity Ratio  ({model}, {eval_dataset}, sets={number_of_sets})")
    ax.set_ylim(0.0, max(1.05, max(max(v) for v in series.values()) * 1.05) if series else 1.05)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, f"accuracy_by_sparsity_{model}_sets_{number_of_sets}.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {out_path}")


def main():
    print(f"[INFO] DISCOVER_DIR = {DISCOVER_DIR}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plot_accuracy_by_sparsity()


if __name__ == "__main__":
    main()
