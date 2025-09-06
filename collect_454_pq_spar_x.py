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
# --- CONFIG ---
BASE_DIR = ("/common/users/sl2148/Public/yang_ouyang/alignment-attribution-code/out/llama2-7b-chat-hf/"
            "unstructured/wanda_3_set_difference_utility_weightonly/wanda_4_set_difference_cot0shot/"
            "eval_selected_samples/prompt_direct,cot0shot/pure_pq_0.01_granular")

K_OPTIONS = [0.17] 
U_OPTIONS = [0.15] 
OUTPUT_DIR = "/common/users/sl2148/Public/yang_ouyang/alignment-attribution-code/scripts/figures/pq_wildcard_u_sweep_3"

# Max number of sparsity ratio points (including baseline 0.0) to plot per (k,u) panel.
# Set to None or a large number to disable.
MAX_POINTS = 20
FORCE_ZERO_ORIGIN = True  # 仍然确保包含 baseline ratio=0.0，但不再把 y 设为 0
X_LOG_SCALE = True        # 将 x 轴改为对数尺度（sparsity ratio）

# OUTPUT_DIR = "/common/users/sl2148/Public/yang_ouyang/alignment-attribution-code/scripts/figures/pq_wildcard_u_sweep_sp_2e-06"

ORIG_COT4_FILE = (
    "/common/users/sl2148/Public/yang_ouyang/alignment-attribution-code/data/GSM8K_eval_build/eval_cot0shot.jsonl"
)
ORIG_DIRECT_FILE = (
    "/common/users/sl2148/Public/yang_ouyang/alignment-attribution-code/data/GSM8K_eval_build/eval_direct.jsonl"
)

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
        m = re.search(r"gsm8k_bottom_([0-9.]+)_", os.path.basename(files[0]))
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

# --- Plot & Table ---
def print_summary_table_with_pqku(all_results: Dict[float, Dict]):
    ratios = sorted(all_results.keys())
    cats = ["direct_success_cot_success", "direct_fail_cot_success"]
    print("\n" + "="*110)
    print("SUMMARY TABLE (with pqku)")
    print("="*110)
    header = f"{'Sparsity':<10} {'P':<6} {'Q':<6} {'K':<6} {'U':<6} {'Category':<25} {'Original':<10} {'Pruned':<10} {'Change':<10} {'Pruned Correct':<15}"
    print(header)
    print("-"*110)
    for r in ratios:
        meta = all_results[r].get("meta", {})
        P = meta.get('p', '-') if r>0 else '-'
        Q = meta.get('q', '-') if r>0 else '-'
        K = meta.get('k', '-') if r>0 else '-'
        U = meta.get('u', '-') if r>0 else '-'
        for c in cats:
            res = all_results[r].get("metrics", {}).get(c, {})
            print(
                f"{r:<10} {P!s:<6} {Q!s:<6} {K!s:<6} {U!s:<6} "
                f"{c:<25} {res.get('original_accuracy',0.0):<10.3f} {res.get('pruned_accuracy',0.0):<10.3f} "
                f"{res.get('accuracy_change',0.0):<10.3f} {res.get('pruned_correct_count',0):<15}"
            )

def _format_float_for_tag(x: Optional[float]) -> str:
    if x is None:
        return "-"
    xi = float(x)
    if xi.is_integer():
        return str(int(xi))
    return str(xi)


def plot_grouped_by_u_for_each_k(
    K_OPTIONS, U_OPTIONS, categories, colors, OUTPUT_DIR,
    collect_all_results_with_meta, _format_float_for_tag
):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 每个 k 生成一张图，子图为不同 u
    N_COLS = min(3, max(1, len(U_OPTIONS)))   # 每行最多 3 个子图
    for k in K_OPTIONS:
        nrows = math.ceil(len(U_OPTIONS) / N_COLS)
        fig, axes = plt.subplots(nrows=nrows, ncols=N_COLS, figsize=(6*N_COLS, 4*nrows), squeeze=False)
        fig.suptitle(f'Grouped by u — k = {_format_float_for_tag(k)}', fontsize=14)

        for idx_u, u in enumerate(U_OPTIONS):
            ax = axes[idx_u // N_COLS][idx_u % N_COLS]

            # 组装通配路径并收集数据
            global FILE_NAME
            FILE_NAME = f"pq_*_*_k_{k}_u_{u}/"
            print(f"[Collect] Using pattern: {FILE_NAME}")
            all_results = collect_all_results_with_meta()

            ratios = sorted(all_results.keys())
            # 确保包含 0.0 baseline
            if FORCE_ZERO_ORIGIN and (0.0 not in ratios):
                ratios = [0.0] + ratios
            # Optional: limit to first MAX_POINTS ratios (keeps baseline 0.0 plus earliest discovered)
            if MAX_POINTS is not None and len(ratios) > MAX_POINTS:
                ratios = ratios[:MAX_POINTS]
            # 计算每个类别的 accuracy 曲线（baseline 用 original_accuracy，其余用 pruned_accuracy）
            values_per_cat = []
            for cat in categories:
                series = []
                for r in ratios:
                    res = all_results[r].get("metrics", {}).get(cat, {})
                    if r == 0.0:
                        series.append(res.get('original_accuracy', 0.0))
                    else:
                        series.append(res.get('pruned_accuracy', res.get('original_accuracy', 0.0)))
                values_per_cat.append(series)

            # 确保 baseline 0.0 存在（不修改其 y 值）
            if FORCE_ZERO_ORIGIN and ratios and ratios[0] != 0.0:
                ratios = [0.0] + ratios
                # 需要同步插入 baseline accuracy（使用第一个 ratio 的 original_accuracy 作为参考）
                for idx_cat, cat in enumerate(categories):
                    baseline_metrics = all_results[0.0].get('metrics', {}).get(cat, {})
                    base_acc = baseline_metrics.get('original_accuracy', 0.0)
                    values_per_cat[idx_cat].insert(0, base_acc)

            if len(ratios) == 0:
                # 没有任何点：画空面板
                ax.text(0.5, 0.5, "No data", ha='center', va='center', fontsize=12)
                ax.set_title(f'u = {_format_float_for_tag(u)} (n=0)')
                ax.set_xlabel('Sparsity Ratio')
                ax.set_ylabel('Accuracy Change')
                ax.set_xlim(0, 1)   # 给个占位坐标
                ax.set_ylim(-1.0, 0.0)
                ax.grid(True, alpha=0.3)
                continue

            # 准备 x 轴（对数尺度需要 >0），如果需要保留 0.0 基线，用一个很小的正值表示并自定义刻度标签
            plot_x = list(ratios)
            baseline_repl = None
            use_log_scale = X_LOG_SCALE  # copy to local to avoid modifying global constant
            if use_log_scale:
                # 找到第一个正的 ratio
                positive_ratios = [r for r in ratios if r > 0]
                if positive_ratios:
                    first_pos = positive_ratios[0]
                    # 将 baseline 0.0 映射到 first_pos / 10，避免与后面点重合
                    if ratios and ratios[0] == 0.0:
                        baseline_repl = first_pos / 10.0
                        if baseline_repl <= 0:
                            baseline_repl = first_pos * 0.1 + 1e-12
                        plot_x[0] = baseline_repl
                else:
                    # 极端情况：除了 0 没有点，直接线性绘制（关闭本地对数）
                    use_log_scale = False

            # 画曲线（点数多少都画），使用 plot_x
            for i, cat in enumerate(categories):
                color_i = colors[i % len(colors)] if len(colors) > 0 else None
                ax.plot(plot_x, values_per_cat[i], 'o-', label=cat, color=color_i)

            # 标注 pq（以多条曲线均值作为 y 位置）
            for j, r in enumerate(ratios):
                if r == 0.0:
                    continue
                meta = all_results[r].get("meta", {})
                p_val, q_val = meta.get("p", None), meta.get("q", None)
                if p_val is None or q_val is None:
                    continue
                y_candidates = []
                for series in values_per_cat:
                    if j < len(series):
                        y_candidates.append(series[j])
                if not y_candidates:
                    continue
                y_pos = float(np.mean(y_candidates))
                p_text = _format_float_for_tag(p_val)
                q_text = _format_float_for_tag(q_val)
                # ax.annotate(
                #     f"pq=({p_text},{q_text})",
                #     (r, y_pos),
                #     textcoords="offset points",
                #     xytext=(6, 6),
                #     fontsize=7
                # )

            ax.set_title(f'u = {_format_float_for_tag(u)} (n={len(ratios)})')
            ax.set_xlabel('Sparsity Ratio (log)' if use_log_scale else 'Sparsity Ratio')
            ax.set_ylabel('Accuracy')
            # 动态 y 轴：基于当前面板的数值范围，留出 5% 边距
            flat_vals = [v for series in values_per_cat for v in series]
            if flat_vals:
                vmin = max(0.0, min(flat_vals) - 0.05)
                vmax = min(1.0, max(flat_vals) + 0.05)
                if vmax - vmin < 0.1:  # 避免太扁平
                    mid = 0.5 * (vmin + vmax)
                    vmin = max(0.0, mid - 0.05)
                    vmax = min(1.0, mid + 0.05)
                ax.set_ylim(vmin, vmax)
            else:
                ax.set_ylim(0.0, 1.0)
            if use_log_scale:
                ax.set_xscale('log')
                # 自定义刻度：包含 baseline 标签 0.0
                ticks = []
                tick_labels = []
                if baseline_repl is not None:
                    ticks.append(baseline_repl)
                    tick_labels.append('0.0')
                # 选取一些代表性刻度（使用当前 plot_x 中的点）
                for val in plot_x:
                    if val == baseline_repl:
                        continue
                    ticks.append(val)
                    tick_labels.append(f"{val:.2g}")
                ax.set_xticks(ticks)
                ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=8)
            else:
                ax.set_xticks(ratios)
                ax.set_xticklabels([f"{r:.3g}" for r in ratios], rotation=45, ha='right', fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9)

        # 把多余的空子图关掉（当 U_OPTIONS 不能整除 N_COLS 时）
        total_axes = nrows * N_COLS
        for spare in range(len(U_OPTIONS), total_axes):
            axes[spare // N_COLS][spare % N_COLS].axis('off')

        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        out_path = os.path.join(
            OUTPUT_DIR,
            f'sparsity_ratios_analysis_grouped_u_k_{_format_float_for_tag(k)}.png'
        )
        fig.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"[Saved] {out_path}")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    categories = ["direct_success_cot_success", "direct_fail_cot_success"]
    colors = ['blue', 'red']

    plot_grouped_by_u_for_each_k(
        K_OPTIONS=K_OPTIONS,
        U_OPTIONS=U_OPTIONS,
        categories=categories,
        colors=colors,
        OUTPUT_DIR=OUTPUT_DIR,
        collect_all_results_with_meta=collect_all_results_with_meta,
        _format_float_for_tag=_format_float_for_tag
    )

    # for k in K_OPTIONS:
    #     for u in U_OPTIONS:
    #         # 关键改动：按 pq 通配；固定 k 与 u
    #         global FILE_NAME
    #         FILE_NAME = f"pq_*_*_k_{k}_us_{u}_granular/"

    #         print(f"[Collect] Using pattern: {FILE_NAME}")
    #         all_results = collect_all_results_with_meta()
    #         # print_summary_table_with_pqku(all_results)

    #         ratios = sorted(all_results.keys())
    #         # 计算两条曲线
    #         deltas_per_cat = []
    #         for i, cat in enumerate(categories):
    #             delta = []
    #             for r in ratios:
    #                 res = all_results[r].get("metrics", {}).get(cat, {})
    #                 delta.append(res.get('accuracy_change', 0.0))
    #             deltas_per_cat.append(delta)

    #         if len(ratios) < 2:
    #             print(f"[Skip] k={k}, u={u} 只有 {len(ratios)} 个点，不画图")
    #             continue
    #         # === 单独保存每个 (k,u) 的图 ===
    #         fig_single, ax_single = plt.subplots(figsize=(6, 4))
    #         for i, cat in enumerate(categories):
    #             ax_single.plot(ratios, deltas_per_cat[i], 'o-', label=cat, color=colors[i])

    #         # 标注 pq
    #         for j, r in enumerate(ratios):
    #             if r == 0.0:
    #                 continue
    #             meta = all_results[r].get("meta", {})
    #             p_val, q_val = meta.get("p", None), meta.get("q", None)
    #             if p_val is None or q_val is None:
    #                 continue
    #             # y 位置：两条曲线均值
    #             y_candidates = []
    #             if len(deltas_per_cat) > 0 and j < len(deltas_per_cat[0]):
    #                 y_candidates.append(deltas_per_cat[0][j])
    #             if len(deltas_per_cat) > 1 and j < len(deltas_per_cat[1]):
    #                 y_candidates.append(deltas_per_cat[1][j])
    #             if not y_candidates:
    #                 continue
    #             y_pos = float(np.mean(y_candidates))
    #             p_text = _format_float_for_tag(p_val)
    #             q_text = _format_float_for_tag(q_val)
    #             ax_single.annotate(
    #                 f"pq=({p_text},{q_text})",
    #                 (r, y_pos),
    #                 textcoords="offset points",
    #                 xytext=(6, 6),
    #                 fontsize=7
    #             )

    #         ax_single.set_title(f'k = {_format_float_for_tag(k)}, u = {_format_float_for_tag(u)}')
    #         ax_single.set_xlabel('Sparsity Ratio')
    #         ax_single.set_ylabel('Accuracy Change')
    #         ax_single.set_ylim(-1.0, 0.0)  # 与原先保持一致
    #         ax_single.grid(True, alpha=0.3)
    #         ax_single.legend()

    #         single_out = os.path.join(
    #             OUTPUT_DIR,
    #             f'sparsity_ratios_analysis_pq_wildcard_k_{_format_float_for_tag(k)}_u_{_format_float_for_tag(u)}.png'
    #         )
    #         fig_single.tight_layout()
    #         fig_single.savefig(single_out, dpi=300, bbox_inches='tight')
    #         plt.close(fig_single)
    #         print(f"[Saved] {single_out}")




if __name__ == '__main__':
    main()
