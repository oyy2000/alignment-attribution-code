import os
import json
import time
import threading
import subprocess
from multiprocessing import Queue

# Configurable Parameters
model = "llama2-7b-chat-hf"
method = "wanda"
sparsity_type = "unstructured"
suffix = "weightonly"
nsamples = 120
log_file = f"command_log_save_eval_all_0_prune.json"

prune_data_options = ["GSM8K_direct_120", "GSM8K_cot0shot_120", "GSM8K_cot0shot_goldreason"]
sparsity_ratios = [0]
neg_options = [False, True]

def build_command(prune_data, sparsity_ratio, is_neg):
    prompt_method = "cot0shot" if "cot0shot" in prune_data else "direct"
    save_dir = f"out/{model}/{sparsity_type}/{method}_{suffix}/{prune_data}/sparsity_{sparsity_ratio}"
    if is_neg:
        command = (
        f"python main.py "
        f"--model {model} "
        f"--prune_method {method} "
        f"--prune_data {prune_data} "
        f"--sparsity_ratio {sparsity_ratio} "
        f"--sparsity_type {sparsity_type} "
        f"--save {save_dir} "
        f"--nsamples {nsamples} "
        f"--neg_prune "
        f"--save_mask "
        f"--eval_gsm8k "
        f"--eval_type fixed "
        f"--prompt_method {prompt_method} "
    )
    else:
        command = (
        f"python main.py "
        f"--model {model} "
        f"--prune_method {method} "
        f"--prune_data {prune_data} "
        f"--sparsity_ratio {sparsity_ratio} "
        f"--sparsity_type {sparsity_type} "
        f"--save {save_dir} "
        f"--nsamples {nsamples} "
        f"--eval_gsm8k "
        f"--eval_type fixed "
        f"--prompt_method {prompt_method} "
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
    for prune_data in prune_data_options:
        for sparsity in sparsity_ratios:
            for is_neg in neg_options:
                commands.append(build_command(prune_data, sparsity, is_neg))

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
