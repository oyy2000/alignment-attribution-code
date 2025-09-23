import os
import json
import time
import threading
import subprocess
from multiprocessing import Queue
import argparse

TIME_OUT = 18000  # 5 hours

# ===== 原配置中保留可改的常量（会在 main() 中用 args 覆盖） =====
suffix = "weightonly"
nsamples = 600
log_lock = threading.Lock()

FREE_RATIO_THRESHOLD = 0.90
MONITOR_TIMEOUT_SEC = 180
MONITOR_POLL_SEC = 5


def parse_args():
    parser = argparse.ArgumentParser(description="Eval runner with GPU monitoring")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen2.5-7B-Instruct", #Qwen2.5-7B-Instruct, deepseek-ai/DeepSeek-R1-Distill-Llama-8B
        help="模型名称/路径，例如 'mistral-7B-Instruct' 或 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B'"
    )
 
    parser.add_argument(
        "--free_ratio_threshold",
        type=float,
        default=0.90,
        help="仅当 free/total >= 该阈值时认定 GPU 可用"
    )
    parser.add_argument(
        "--monitor_timeout_sec",
        type=int,
        default=180,
        help="等待 GPU 达到阈值的最长秒数"
    )
    parser.add_argument(
        "--monitor_poll_sec",
        type=int,
        default=5,
        help="轮询 GPU 状态的间隔秒数"
    )
    return parser.parse_args()

# 注意：原代码里是列表中放了一个逗号串；这里改为真正的列表
def parse_prompt_methods(s: str):
    return [x.strip() for x in s.split(",") if x.strip()]

def sanitize_for_path(s: str):
    # 处理包含 "/" 的 huggingface 名称，避免当成子目录
    return s.replace("/", "_")

model = "Qwen2.5-7B-Instruct"#"Qwen2.5-7B-Instruct" # deepseek-ai/DeepSeek-R1-Distill-Llama-8B, Qwen2.5-7B-Instruct, mistral-7B-Instruct, llama2-7b-chat-hf
model_for_path = sanitize_for_path(model)
eval_dataset = "GSM8K"
eval_type = "selected_samples"
add_template_flag = False
nsamples = 120
prompt_methods_list = ["direct,cot0shot"] # 可以改成字符串，自动 split
sparsity_type = "unstructured"
suffix = "weightonly"
prune_method = "wanda_234_set_difference"
number_of_sets_options = [3, 4]  # [1, 2, 3, 4]
prune_data="GSM8K"
prune_prompt_options = ["cot0shot", "cot0shot_goldreason", "alpaca_cleaned_no_safety", "direct"] #, "cot0shot", "cot0shot_goldreason", "cot4shot"]  # "GSM8K_cot0shot_goldreason_truncated" "alpaca_cleaned_no_safety" #"GSM8K_cot4shot_120"

p=0.01
q=0.01 
k=0.01
u=0.01
pq_options = [round(0.01 * i, 2) for i in range(21, 41)]   # 0.01 → 0.10
k_options = [0.17]
u_options = [0.15]
sparsity_threshold=0.0000002
set_difference_data="GSM8K"

def build_command(
    prompt_method="direct",
    p=0.01,
    q=0.01,
    k=0.01,
    u=0.01,
    number_of_sets=3,
):
    use_template = True if add_template_flag else False
    save_dir = (
        f"/common/users/sl2148/Public/yang_ouyang/alignment-attribution-code/"
        f"out/eval_{eval_dataset}/{model}/{sparsity_type}/{prune_method}_{suffix}/sets_{number_of_sets}/"
        f"set_difference_data_{set_difference_data}/eval_{eval_type}/"
        f"prompt_{prompt_method}/add_template_{use_template}/step_0.01_sp_{sparsity_threshold}/pq_{p}_{q}_k_{k}_u_{u}/"
    )
    out_dir = f"/common/users/sl2148/Public/yang_ouyang/alignment-attribution-code/out"
    command = (
        f"python ../main.py "
        f"--model {model} "
        f"--save {save_dir} "
        f"--nsamples {nsamples} "
        f"--out_dir {out_dir} "
        f"--eval_gsm8k "
        f"--dataset {eval_dataset} "
        f"--eval_type {eval_type} "
        f"--sparsity_type {sparsity_type} "
        f"--prune_method {prune_method} "
        f"--number_of_sets {number_of_sets} "
        f"--prune_data {prune_data} "
        f"--sparsity_ratio 0.5 "
        f"--batch_size 64 "
        # f"--dump_wanda_score " 
        f"--set_difference_data {set_difference_data} "
        f"--prompt_method {prompt_method} "
        f"--save_sparsity "  # Save sparsity ratio
        f"--sparsity_threshold {sparsity_threshold} "
        f"--cache_dir {out_dir}/../llm_weights "
        f"--p {p} "
        f"--q {q} "
        f"--k {k} "
        f"--u {u} "
    )
    return command



def initialize_log(log_file,commands):
    if not os.path.exists(log_file):
        with open(log_file, 'w') as f:
            json.dump([{"command": cmd, "result": "pending"} for cmd in commands], f, indent=2)
    else:
        with open(log_file, 'r') as f:
            logs = json.load(f)
        with open(log_file, 'w') as f:
            json.dump(logs, f, indent=2)


def update_log(log_file,command, result):
    with log_lock:
        with open(log_file, 'r') as f:
            logs = json.load(f)
        for entry in logs:
            if entry["command"] == command:
                entry["result"] = result
                break
        atomic_write_json(log_file, logs)

import tempfile, os, json

def atomic_write_json(path, data):
    dir_ = os.path.dirname(path)
    with tempfile.NamedTemporaryFile('w', dir=dir_, delete=False) as tf:
        json.dump(data, tf, indent=2)
        temp_name = tf.name
    os.replace(temp_name, path)   # rename 原子替换

def _query_nvidia_smi(field):
    proc = subprocess.run(
        ['nvidia-smi', f'--query-gpu={field}', '--format=csv,nounits,noheader'],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    out = proc.stdout.decode().strip()
    if not out:
        return []
    return [int(x) for x in out.splitlines()]

def get_gpu_free_memory_list():
    used = _query_nvidia_smi('memory.used')
    total = _query_nvidia_smi('memory.total')
    if not used or not total:
        return []
    return [t - u for t, u in zip(total, used)]

def get_gpu_total_memory_list():
    return _query_nvidia_smi('memory.total')

def get_gpu_free_total_pairs():
    free = get_gpu_free_memory_list()
    total = get_gpu_total_memory_list()
    return list(zip(free, total))  # [(free, total), ...]

def get_eligible_gpus_by_ratio(ratio=FREE_RATIO_THRESHOLD):
    pairs = get_gpu_free_total_pairs()
    eligible = []
    for i, (free, total) in enumerate(pairs):
        if total > 0 and (free / total) >= ratio:
            eligible.append(i)
    return eligible

def monitor_gpu_by_ratio(gpu_id, ratio=FREE_RATIO_THRESHOLD, timeout=MONITOR_TIMEOUT_SEC, poll=MONITOR_POLL_SEC):
    start = time.time()
    while time.time() - start < timeout:
        pairs = get_gpu_free_total_pairs()
        if gpu_id < len(pairs):
            free, total = pairs[gpu_id]
            if total > 0 and (free / total) >= ratio:
                return True
        time.sleep(poll)
    return False

# ================== Runner ==================
def run_command(log_file, command, gpu_id):
    full_cmd = f"CUDA_VISIBLE_DEVICES={gpu_id} {command}"
    print(f"[GPU {gpu_id}] Running: {full_cmd}")
    update_log(log_file, command, "running")
    process = subprocess.Popen(full_cmd, shell=True)
    code = process.wait()
    if code == 0:
        update_log(log_file, command, "success")
    else:
        update_log(log_file, command, f"failed (code {code})")
    time.sleep(10)


def load_pending_or_failed_tasks(log_file):
    with open(log_file, 'r') as f:
        logs = json.load(f)
    tasks = []
    for i, log in enumerate(logs):
        r = log["result"]
        if (r == "pending") or (isinstance(r, str) and r.startswith("failed")):
            tasks.append((i, log["command"]))
    return tasks

def worker(task_queue, gpu_id, ratio, timeout_sec, poll_sec, log_file):
    while True:
        task = task_queue.get()
        if task is None:
            break
        _, command = task
        if monitor_gpu_by_ratio(gpu_id, ratio=ratio, timeout=timeout_sec, poll=poll_sec):
            run_command(log_file, command, gpu_id)
        else:
            print(f"[GPU {gpu_id}] Not enough free ratio (< {ratio:.0%}) for: {command}")
            task_queue.put(task)
            time.sleep(20)


def main():
    global FREE_RATIO_THRESHOLD, MONITOR_TIMEOUT_SEC, MONITOR_POLL_SEC

    args = parse_args()
    # 覆盖全局阈值与超时
    FREE_RATIO_THRESHOLD = args.free_ratio_threshold
    MONITOR_TIMEOUT_SEC = args.monitor_timeout_sec
    MONITOR_POLL_SEC = args.monitor_poll_sec

    # 基于参数生成日志文件名
    log_file = f"command_log_generation_{eval_dataset}_{model_for_path}_set_{number_of_sets_options}_{prune_method}_{prompt_methods_list}_{pq_options}.json"
    
    
    # Generate all commands
    commands = []
    for prompt_method in prompt_methods_list:
        for p in pq_options:
            q = p
            for k in k_options:
                for u in u_options:
                    for number_of_sets in number_of_sets_options:

                        commands.append(    
                            build_command(
                                prompt_method=prompt_method,
                                p=p,
                                q=q,
                                k=k,
                                u=u,
                                number_of_sets=number_of_sets,
                            )
                        )

    initialize_log(log_file, commands)
    pending = load_pending_or_failed_tasks(log_file)
    if not pending:
        print("No tasks to run.")
        return

    # 关键变化：不再用 gpu_count = len(get_gpu_free_memory())
    # 而是按“free/total >= 90%”筛选可用 GPU 作为并发池
    eligible_gpus = get_eligible_gpus_by_ratio(FREE_RATIO_THRESHOLD)

    # 若此刻没有任何卡满足 90% 空闲，则退化为“全卡监听”，
    # 这样当某张卡稍后释放到 90% 时也能自动开跑
    if not eligible_gpus:
        print(f"[Info] No GPUs meet free/total >= {FREE_RATIO_THRESHOLD:.0%} right now; falling back to all GPUs and waiting...")
        eligible_gpus = list(range(len(get_gpu_free_memory_list())))

    task_queue = Queue()
    for task in pending:
        task_queue.put(task)
    # 放入结束标记，与线程数一致
    for _ in range(len(eligible_gpus)):
        task_queue.put(None)

    threads = []
    for gpu_id in eligible_gpus:
        t = threading.Thread(
            target=worker,
            args=(task_queue, gpu_id, FREE_RATIO_THRESHOLD, MONITOR_TIMEOUT_SEC, MONITOR_POLL_SEC, log_file)
        )
        t.start()
        threads.append(t)

    for t in threads:
        t.join()


if __name__ == "__main__":
    main()
