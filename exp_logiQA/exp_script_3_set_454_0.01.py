import os
import json
import time
import threading
import subprocess
import numpy as np
from multiprocessing import Queue

# Configurable Parameters
model = "llama2-7b-chat-hf"
sparsity_type = "unstructured"
suffix = "weightonly"
prune_method_options = ["wanda_3_set_difference_utility"] #["wanda_3_set_difference_utility"]
prompt_methods = ["direct,cot0shot"] #["cot2shot,cot4shot,cot8shot,cot16shot"]
# pq_options = [0.01]  # (p, q) for 3-set pruning
pq_options = [round(0.01 * i, 2) for i in range(1, 20)]   # 
k_options = [0.17]
u_options = [0.15]
sparsity_threshold = 0.0000002
dataset = "LOGIQA"
nsamples = 600
eval_type = "all"

log_file = f"command_log_eval_gsm8k_wanda_3_set_454_alpaca_cleaned_no_safety_pquk_grid_search_[round(0.01 * i, 2) for i in range(1, 20)]_step_0.01_sp_{sparsity_threshold}.json"
def build_command(prune_method, prompt_method, p, q, k, u):
    save_dir = f"/common/users/sl2148/Public/yang_ouyang/alignment-attribution-code/out/{dataset}/{model}/{sparsity_type}/{prune_method}_{suffix}/wanda_3_set_difference_cot0shot/eval_{eval_type}/prompt_{prompt_method}/step_0.01_sp_{sparsity_threshold}_k_0.01/pq_{p}_{q}_k_{k}_u_{u}/"
    command = (
        f"python ../main.py "
        f"--sparsity_threshold {sparsity_threshold} "
        f"--model {model} "
        f"--prune_method {prune_method} "
        f"--sparsity_type {sparsity_type} "
        f"--sparsity_ratio 0.5 "
        f"--save {save_dir} "
        f"--nsamples {nsamples} "
        f"--eval_datasets "
        f"--eval_type {eval_type} "
        f"--prompt_method {prompt_method} "
        f"--dataset {dataset} "
        f"--save_sparsity "  # Save sparsity ratio
        f"--cache_dir /common/users/sl2148/Public/yang_ouyang/alignment-attribution-code/llm_weights "
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

# 放在脚本最前面
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

import tempfile, os, json

def atomic_write_json(path, data):
    dir_ = os.path.dirname(path)
    with tempfile.NamedTemporaryFile('w', dir=dir_, delete=False) as tf:
        json.dump(data, tf, indent=2)
        temp_name = tf.name
    os.replace(temp_name, path)   # rename 原子替换

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

def run_command(command, gpu_id):
    full_cmd = f"CUDA_VISIBLE_DEVICES={gpu_id} {command}"
    print(f"[GPU {gpu_id}] Running: {full_cmd}")
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

def worker(task_queue, gpu_id):
    while True:
        task = task_queue.get()
        if task is None:
            break
        _, command = task
        if monitor_gpu(gpu_id, min_free_mem=20000):  # Adjust if needed
            run_command(command, gpu_id)
        else:
            print(f"[GPU {gpu_id}] Not enough memory for: {command}")
            task_queue.put(task)
            time.sleep(20)
            
def main():
    # Generate all commands
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

    gpu_count = len(get_gpu_free_memory())
    task_queue = Queue()

    for task in pending:
        task_queue.put(task)
    for _ in range(gpu_count):
        task_queue.put(None)

    threads = []
    for gpu_id in range(gpu_count):
        t = threading.Thread(target=worker, args=(task_queue, gpu_id))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

if __name__ == "__main__":
    main()
