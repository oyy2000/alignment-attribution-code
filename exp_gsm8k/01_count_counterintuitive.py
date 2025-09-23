import json
import os
from typing import Dict, List, Tuple

# ==== 配置部分 ====
# ORIG_COT0_FILE = (
#     "/common/users/sl2148/Public/yang_ouyang/alignment-attribution-code/out/GSM8K/llama2-7b-chat-hf/unstructured/wanda_3_set_difference_utility_weightonly/wanda_4_set_difference_cot0shot/eval_selected_samples/prompt_direct,cot0shot/pure_pq_0.01_granular/pq_0.1_0.1_k_0.17_u_0.15/gsm8k_bottom_0.001101_direct,cot0shot_selected_samples_prompt_cot0shot.jsonl"
# )
# ORIG_DIRECT_FILE = (
#     "/common/users/sl2148/Public/yang_ouyang/alignment-attribution-code/out/GSM8K/llama2-7b-chat-hf/unstructured/wanda_3_set_difference_utility_weightonly/wanda_4_set_difference_cot0shot/eval_selected_samples/prompt_direct,cot0shot/pure_pq_0.01_granular/pq_0.1_0.1_k_0.17_u_0.15/gsm8k_bottom_0.001101_direct,cot0shot_selected_samples_prompt_direct.jsonl"
# )
ORIG_COT0_FILE = (
"/common/users/sl2148/Public/yang_ouyang/alignment-attribution-code/data/GSM8K_eval_build/eval_cot0shot.jsonl"
)
ORIG_DIRECT_FILE = (
  "/common/users/sl2148/Public/yang_ouyang/alignment-attribution-code/data/GSM8K_eval_build/eval_direct.jsonl"
)

JUDGE_FILE = "/common/users/sl2148/Public/yang_ouyang/alignment-attribution-code/exp_gsm8k/full_gsm8k_cot_judgments.jsonl"
AFTER_PRUNE_COT0_FILE = "/common/users/sl2148/Public/yang_ouyang/alignment-attribution-code/out/GSM8K/llama2-7b-chat-hf/unstructured/wanda_3_set_difference_utility_weightonly/wanda_4_set_difference_cot0shot/eval_selected_samples/prompt_direct,cot0shot/pure_pq_0.01_granular/pq_0.1_0.1_k_0.17_u_0.15/gsm8k_bottom_0.001101_direct,cot0shot_selected_samples_prompt_cot0shot.jsonl"
AFTER_PRUNE_DIRECT_FILE = "/common/users/sl2148/Public/yang_ouyang/alignment-attribution-code/out/GSM8K/llama2-7b-chat-hf/unstructured/wanda_3_set_difference_utility_weightonly/wanda_4_set_difference_cot0shot/eval_selected_samples/prompt_direct,cot0shot/pure_pq_0.01_granular/pq_0.1_0.1_k_0.17_u_0.15/gsm8k_bottom_0.001101_direct,cot0shot_selected_samples_prompt_direct.jsonl"
# =================

def load_jsonl_by_id(path: str) -> Dict[str, dict]:
    """读取 JSONL，返回 {id: obj}"""
    data = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            sid = str(obj.get("id"))
            if sid:
                data[sid] = obj
    return data


def match_ids(dict1: Dict[str, dict], dict2: Dict[str, dict]) -> List[str]:
    """以 id 直接取交集"""
    return list(set(dict1.keys()) & set(dict2.keys()))


def categorize_samples(
    cot_info: Dict[str, dict],
    direct_info: Dict[str, dict],
    common_ids: List[str],
) -> Tuple[List[str], List[str]]:
    """
    返回两个类别的 id 列表：
      d_s_c_s: direct 成功 & cot 成功
      d_f_c_s: direct 失败 & cot 成功
    """
    d_s_c_s, d_f_c_s = [], []
    for sid in common_ids:
        cot_ok = bool(cot_info.get(sid, {}).get("correct", False))
        dir_ok = bool(direct_info.get(sid, {}).get("correct", False))
        if cot_ok and dir_ok:
            d_s_c_s.append(sid)
        elif cot_ok and (not dir_ok):
            d_f_c_s.append(sid)
    return d_s_c_s, d_f_c_s


def load_judge_true_ids(path: str) -> set:
    """
    从评分类文件里筛出 process_wrong_but_answer_correct == True 的样本 id 集合
    """
    judge_true = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("process_wrong_but_answer_correct") is True:
                sid = str(obj.get("id"))
                if sid:
                    judge_true.add(sid)
    return judge_true


def summarize_categories(categories: Dict[str, List[str]], judge_true_ids: set) -> Dict[str, dict]:
    """
    对每个类别统计与 judge_true_ids 的交集数量（不抽样）
    """
    summary = {}
    for name, id_list in categories.items():
        hit = [sid for sid in id_list if sid in judge_true_ids]
        summary[name] = {
            "total_in_category": len(id_list),
            "process_wrong_but_answer_correct_true": len(hit),
        }
    return summary


def main():
    # 1) 读入原始数据
    direct_by_id = load_jsonl_by_id(ORIG_DIRECT_FILE)
    cot0_by_id = load_jsonl_by_id(ORIG_COT0_FILE)

    # 2) 匹配共同 id
    common_ids = match_ids(cot0_by_id, direct_by_id)
    print(f"[Info] Matched samples by id: {len(common_ids)}")

    # 3) 分类（只保留你需要的两类）
    d_s_c_s, d_f_c_s = categorize_samples(cot0_by_id, direct_by_id, common_ids)

    category_samples = {
        "direct_success_cot_success": d_s_c_s,
        "direct_fail_cot_success": d_f_c_s,
    }

    # 4) 读入评分类文件（只取 process_wrong_but_answer_correct == True）
    judge_true_ids = load_judge_true_ids(JUDGE_FILE)
    print(f"[Info] Judge TRUE ids (process_wrong_but_answer_correct): {len(judge_true_ids)}")

    # 5) 仅统计（不抽样）
    summary = summarize_categories(category_samples, judge_true_ids)

    # 6) 打印摘要
    print("\n===== Summary =====")
    for name, stats in summary.items():
        print(
            f"{name:>28}: total={stats['total_in_category']:4d} | "
            f"process_wrong_but_answer_correct_true={stats['process_wrong_but_answer_correct_true']:4d}"
        )

    # 7) 输出每个类别命中 judge 条件的完整 id 列表
    out_all = os.path.splitext(JUDGE_FILE)[0] + "_per_category_hits.json"
    hits_dict = {
        name: [sid for sid in ids if sid in judge_true_ids]
        for name, ids in category_samples.items()
    }
    with open(out_all, "w", encoding="utf-8") as f:
        json.dump(hits_dict, f, ensure_ascii=False, indent=2)
    print(f"[Info] Saved per-category hit ids to: {out_all}")


if __name__ == "__main__":
    main()
