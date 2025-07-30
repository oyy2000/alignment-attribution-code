import os
import json
import time
import threading
import subprocess
from multiprocessing import Queue
from threading import Lock

# Configurable Parameters
model = "llama2-7b-chat-hf"
sparsity_type = "unstructured"
suffix = "weightonly"
nsamples = 120
log_file = f"command_log_eval_gsm8k_wanda_3_set.json"

prune_data_options = ["GSM8K_direct_120", "GSM8K_cot0shot_120", "GSM8K_cot0shot_goldreason"]
sparsity_ratios = [0.5] #[0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
prune_method_options = ["wanda_3_set_difference"]
prompt_methods = ["direct", "cot0shot", "cot0shot_goldreason"]
pq_options = [0.5]  # (p, q) for 3-set pruning
k_options = [0.5, 0.45, 0.4, 0.35, 0.3]  # k for 3-set pruning
def build_command(prune_data, sparsity_ratio, prune_method, prompt_method, p, q, k):
    save_dir = f"out/{model}/{sparsity_type}/{prune_method}_{suffix}/{prune_data}/sparsity_{sparsity_ratio}"
    command = (
        f"python main.py "
        f"--model {model} "
        f"--prune_method {prune_method} "
        f"--prune_data {prune_data} "
        f"--sparsity_ratio {sparsity_ratio} "
        f"--sparsity_type {sparsity_type} "
        f"--save {save_dir} "
        f"--nsamples {nsamples} "
        f"--eval_gsm8k "
        f"--eval_type held_out "
        f"--prompt_method {prompt_method} "
        f"--save_sparsity "  # Save sparsity ratio
        f"--p {p} "
        f"--q {q} "
        f"--k {k} "
    )
  
    return command

MIN_FREE_MEM = 20000            # 预计每卡至少空多少 MB

def find_gpu_pair(min_free_mem=MIN_FREE_MEM):
    """返回一对满足显存要求的 GPU id；若找不到则返回 None"""
    free = get_gpu_free_memory()
    candidates = [i for i, mem in enumerate(free) if mem >= min_free_mem]
    if len(candidates) < 2:
        return None
    # 这里简单地取前两块；也可以做更多策略（例如最空闲组合）
    return tuple(candidates[:2])

def initialize_log(commands):
    if not os.path.exists(log_file):
        with open(log_file, 'w') as f:
            json.dump([{"command": cmd, "result": "pending"} for cmd in commands], f, indent=2)
    else:
        with open(log_file, 'r') as f:
            logs = json.load(f)
        with open(log_file, 'w') as f:
            json.dump(logs, f, indent=2)

def update_log(command, result):
    lock = threading.Lock()
    with lock:
        with open(log_file, 'r') as f:
            logs = json.load(f)
        for entry in logs:
            if entry["command"] == command:
                entry["result"] = result
                break
        with open(log_file, 'w') as f:
            json.dump(logs, f, indent=2)

def get_gpu_free_memory():
    used = subprocess.run(
        ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'],
        stdout=subprocess.PIPE
    )
    total = subprocess.run(
        ['nvidia-smi', '--query-gpu=memory.total', '--format=csv,nounits,noheader'],
        stdout=subprocess.PIPE
    )
    used = [int(x) for x in used.stdout.decode().strip().split('\n')]
    total = [int(x) for x in total.stdout.decode().strip().split('\n')]
    return [t - u for t, u in zip(total, used)]

def monitor_gpu(gpu_id, min_free_mem, timeout=180):
    start = time.time()
    while time.time() - start < timeout:
        if get_gpu_free_memory()[gpu_id] >= min_free_mem:
            return True
        time.sleep(5)
    return False

def run_command(command, gpu_pair):
    gpu_a, gpu_b = gpu_pair
    full_cmd = f"CUDA_VISIBLE_DEVICES={gpu_a},{gpu_b} {command}"
    print(f"[GPU {gpu_a},{gpu_b}] Running: {full_cmd}")
    process = subprocess.Popen(full_cmd, shell=True)
    update_log(command, "running")
    code = process.wait()
    if code == 0:
        update_log(command, "success")
    else:
        update_log(command, f"failed (code {code})")
    time.sleep(10)

def load_pending_or_failed_tasks():
    with open(log_file, 'r') as f:
        logs = json.load(f)
    return [(i, log["command"]) for i, log in enumerate(logs) if log["result"] == "pending" or log["result"].startswith("failed")]

from collections import deque
available_pairs = deque([(0,1), (2,3), (4,5), (6,7)])  # 全局可用池
cond = threading.Condition()          # 代替 Lock，支持 wait/notify
def worker(task_queue, cond):
    while True:
        task = task_queue.get()
        if task is None:
            break

        # 1. 取 GPU 对（阻塞等待）
        with cond:
            while not available_pairs:
                cond.wait()
            gpu_pair = available_pairs.popleft()

        # 2. 跑任务
        _, cmd = task
        try:
            run_command(cmd, gpu_pair)
        finally:
            # 3. 归还 GPU 对并通知其他线程
            with cond:
                available_pairs.append(gpu_pair)
                cond.notify()


def main():
    # Generate all commands
    commands = []
    for prune_data in prune_data_options:
        for sparsity in sparsity_ratios:
            for prune_method in prune_method_options:
                for prompt_method in prompt_methods:
                    for p in pq_options:
                        q = p
                        for k in k_options:
                            commands.append(build_command(prune_data, sparsity, prune_method, prompt_method, p, q, k))

    initialize_log(commands)
    pending = load_pending_or_failed_tasks()
    if not pending:
        print("No tasks to run.")
        return

    gpu_count = len(get_gpu_free_memory())
    task_queue = Queue()

    for task in pending:
        task_queue.put(task)
    for _ in range(gpu_count):
        task_queue.put(None)

    lock = Lock()
    threads = []
    # 线程数  =  GPU对数  =  gpu_count // 2
    for _ in range(len(get_gpu_free_memory()) // 2):
        t = threading.Thread(target=worker, args=(task_queue, cond), daemon=True)
        t.start()
        threads.append(t)

    for t in threads:
        t.join()    

if __name__ == "__main__":
    main()
