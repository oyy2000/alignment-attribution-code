import os
import json
import time
import threading
import subprocess
from multiprocessing import Queue

TIME_OUT = 18000  # 5 hours
# Configurable Parameters
# model = "llama2-7b-chat-hf" # deepseek-ai/DeepSeek-R1-Distill-Llama-8B
model = "llama2-7b-chat-hf" #"mistral-7B-Instruct" # Qwen2.5-7B-Instruct
# Qwen/Qwen2.5-7B-Instruct
suffix = "weightonly"
nsamples = 600
dataset = "GSM8K"

prompt_methods = ["cot0shot,direct,cot0shot_goldreason,cot4shot"]
prompt_methods_options = ["cot0shot", "direct", "cot0shot_goldreason", "cot4shot"]  
log_file = f"command_log_generation_{dataset}_{model}_{prompt_methods}_eval_gsm8k.json"
eval_type = "all"

def build_command(prompt_method):
    save_dir = f"/common/users/sl2148/Public/yang_ouyang/alignment-attribution-code/out/{dataset}/{model}/{prompt_methods}/{prompt_method}/eval_{eval_type}"
    command = (
        f"python ../main.py "
        f"--model {model} "
        f"--save {save_dir} "
        f"--nsamples {nsamples} "
        f"--eval_gsm8k " # eval_gsm8k, eval_addition, eval_datasets
        f"--dataset {dataset} "
        f"--eval_type {eval_type} "
        f"--prompt_method {prompt_method} "
        f"--batch_size 64 " # do not add_template
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

def monitor_gpu(gpu_id, min_free_mem, timeout=TIME_OUT):
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
    for prompt_method in prompt_methods_options:
        commands.append(build_command(prompt_method))

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
