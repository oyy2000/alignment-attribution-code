import os
import json
import time
import threading
import subprocess
from multiprocessing import Queue
import tempfile

# ================== Configurable Parameters ==================
model = "llama2-7b-chat-hf"
sparsity_type = "unstructured"
suffix = "weightonly"
nsamples = 600
eval_type = "selected_samples"
prune_method_options = ["wanda_3_set_difference_utility"]  # ["wanda_3_set_difference_utility"]
prompt_methods = ["direct,cot0shot"]  # ["cot2shot,cot4shot,cot8shot,cot16shot"]
pq_options = [0.01, 0.02]  # (p, q) for 3-set pruning
k_options = [0.01, 0.02, 0.03, 0.04, 0.05]
u_options = [0.01, 0.02, 0.03, 0.04, 0.05]

# 阈值：仅当 free/total >= 0.90 时，这张卡被当作“可用”
FREE_RATIO_THRESHOLD = 0.90
# 监控参数
MONITOR_TIMEOUT_SEC = 180
MONITOR_POLL_SEC = 5

log_file = f"command_log_eval_gsm8k_wanda_4_set_450_alpaca_cleaned_no_safety_pquk_grid_search_{pq_options}_0.01_0.0005.json"
def build_command(prune_method, prompt_method, p, q, k, u):
    save_dir = f"out/{model}/{sparsity_type}/{prune_method}_{suffix}/wanda_4_set_difference_cot0shot/eval_{eval_type}/prompt_{prompt_method}/0.01_sp_0.0005_granular/pq_{p}_{q}_k_{k}_u_{u}/"
    command = (
        f"python main.py "
        f"--model {model} "
        f"--prune_method {prune_method} "
        f"--sparsity_type {sparsity_type} "
        f"--sparsity_ratio 0.5 "
        f"--save {save_dir} "
        f"--nsamples {nsamples} "
        f"--eval_gsm8k "
        f"--eval_type {eval_type} "
        f"--prompt_method {prompt_method} "
        f"--save_sparsity "  # Save sparsity ratio
        f"--p {p} "
        f"--q {q} "
        f"--k {k} "
        f"--u {u} "
    )
  
    return command

def initialize_log(commands):
    if not os.path.exists(log_file):
        with open(log_file, 'w') as f:
            json.dump([{"command": cmd, "result": "pending"} for cmd in commands], f, indent=2)
    else:
        with open(log_file, 'r') as f:
            logs = json.load(f)
        with open(log_file, 'w') as f:
            json.dump(logs, f, indent=2)

# 原子写日志
def atomic_write_json(path, data):
    dir_ = os.path.dirname(path) or "."
    with tempfile.NamedTemporaryFile('w', dir=dir_, delete=False) as tf:
        json.dump(data, tf, indent=2)
        temp_name = tf.name
    os.replace(temp_name, path)

# 线程安全更新日志
log_lock = threading.Lock()
def update_log(command, result):
    with log_lock:
        with open(log_file, 'r') as f:
            logs = json.load(f)
        for entry in logs:
            if entry["command"] == command:
                entry["result"] = result
                break
        atomic_write_json(log_file, logs)

# ================== GPU Memory Utils ==================
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
def run_command(command, gpu_id):
    full_cmd = f"CUDA_VISIBLE_DEVICES={gpu_id} {command}"
    print(f"[GPU {gpu_id}] Running: {full_cmd}")
    update_log(command, "running")
    process = subprocess.Popen(full_cmd, shell=True)
    code = process.wait()
    if code == 0:
        update_log(command, "success")
    else:
        update_log(command, f"failed (code {code})")
    time.sleep(10)

def load_pending_or_failed_tasks():
    with open(log_file, 'r') as f:
        logs = json.load(f)
    tasks = []
    for i, log in enumerate(logs):
        r = log["result"]
        if (r == "pending") or (isinstance(r, str) and r.startswith("failed")):
            tasks.append((i, log["command"]))
    return tasks

def worker(task_queue, gpu_id, ratio=FREE_RATIO_THRESHOLD):
    while True:
        task = task_queue.get()
        if task is None:
            break
        _, command = task
        if monitor_gpu_by_ratio(gpu_id, ratio=ratio):
            run_command(command, gpu_id)
        else:
            print(f"[GPU {gpu_id}] Not enough free ratio (< {ratio:.0%}) for: {command}")
            task_queue.put(task)  # 放回队列，稍后再试
            time.sleep(20)

def main():
    # 生成全部命令
    commands = []
    for prune_method in prune_method_options:
        for prompt_method in prompt_methods:
            for p in pq_options:
                q = p
                for k in k_options:
                    for u in u_options:
                        commands.append(build_command(prune_method, prompt_method, p, q, k, u))

    initialize_log(commands)
    pending = load_pending_or_failed_tasks()
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
        t = threading.Thread(target=worker, args=(task_queue, gpu_id, FREE_RATIO_THRESHOLD))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

if __name__ == "__main__":
    main()
