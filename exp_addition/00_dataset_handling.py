import json
import os
import random

# ==== 配置部分 ====
ORIG_COT0_FILE = "/common/users/sl2148/Public/yang_ouyang/alignment-attribution-code/out/Addition:6/llama2-7b-chat-hf/direct,cot0shot/eval_all/addition_bottom_0.000000_direct,cot0shot_all_prompt_cot0shot.jsonl"
ORIG_DIRECT_FILE = "/common/users/sl2148/Public/yang_ouyang/alignment-attribution-code/out/Addition:6/llama2-7b-chat-hf/direct,cot0shot/eval_all/addition_bottom_0.000000_direct,cot0shot_all_prompt_direct.jsonl"

# 已经选择过的集合 (这里直接用 list/集合，自己替换成实际 id 列表即可)
selected_direct_correct_cot_correct = set()
selected_cot0shot_correct_direct_wrong = set()
dataset = "Addition:6"
OUT_DIR = f"../data/{dataset}_eval_build"
OUT_DIRECT = os.path.join(OUT_DIR, f"calibration_{dataset}_direct.jsonl")
OUT_COT0   = os.path.join(OUT_DIR, f"calibration_{dataset}_cot0shot.jsonl")
OUT_IDS    = os.path.join(OUT_DIR, f"calibration_{dataset}_ids.txt")

N = 120
SEED = 2025
# =================

random.seed(SEED)
os.makedirs(OUT_DIR, exist_ok=True)


def load_jsonl_by_id(path):
    data = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            data[str(obj["id"])] = obj
    return data


# 1. 读入原始数据
direct_by_id = load_jsonl_by_id(ORIG_DIRECT_FILE)
cot0_by_id = load_jsonl_by_id(ORIG_COT0_FILE)


# Build helper sets
direct_correct_ids_set = {qid for qid, rec in direct_by_id.items() if rec.get("correct")}
cot0shot_correct_ids_set = {qid for qid, rec in cot0_by_id.items() if rec.get("correct")}

selected_direct_correct_cot_correct = direct_correct_ids_set & cot0shot_correct_ids_set
selected_cot0shot_correct_direct_wrong = cot0shot_correct_ids_set - direct_correct_ids_set

# 2. 构造候选池
all_ids = set(direct_by_id.keys()) | set(cot0_by_id.keys())
exclude_ids = selected_direct_correct_cot_correct | selected_cot0shot_correct_direct_wrong
candidates = list(all_ids - exclude_ids)

print(f"All ids: {len(all_ids)}, Exclude: {len(exclude_ids)}, Candidates: {len(candidates)}")

# 3. 随机抽取
n_pick = min(N, len(candidates))
sampled_ids = sorted(random.sample(candidates, n_pick))

# 4. 写 calibration_ids.txt
with open(OUT_IDS, "w", encoding="utf-8") as f:
    for qid in sampled_ids:
        f.write(f"{qid}\n")

# 5. 写 direct.jsonl 和 cot0shot.jsonl
n_direct, n_cot = 0, 0
with open(OUT_DIRECT, "w", encoding="utf-8") as f_out:
    for qid in sampled_ids:
        if qid in direct_by_id:
            rec = dict(direct_by_id[qid])
            rec["sample_type"] = "calibration"
            f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_direct += 1

with open(OUT_COT0, "w", encoding="utf-8") as f_out:
    for qid in sampled_ids:
        if qid in cot0_by_id:
            rec = dict(cot0_by_id[qid])
            rec["sample_type"] = "calibration"
            f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_cot += 1

print(f"[Done] Picked {len(sampled_ids)} ids -> direct {n_direct}, cot0shot {n_cot}")
print(f"Saved to {OUT_DIRECT}, {OUT_COT0}, {OUT_IDS}")
