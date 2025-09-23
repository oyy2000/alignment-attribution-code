# Code adapted from https://github.com/IST-DASLab/sparsegpt/blob/master/datautils.py

import numpy as np
import random
import torch
from datasets import load_dataset
import json
import os
import random
import time
from tqdm import tqdm
import argparse
import numpy as np
import re
from collections import defaultdict

# Set seed for reproducibility
def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)


# Wrapper for tokenized input IDs
class TokenizerWrapper:
    def __init__(self, input_ids):
        self.input_ids = input_ids


def _truncate_at_marker(text: str, markers=None) -> str:
    """
    在文本中查找最后一次出现的结论标志（如 So / Therefore / therefore），
    并删除该标志及其后内容。若未找到，返回原文。
    """
    if not text:
        return text
    if markers is None:
        # 只按你的要求保留 So / Therefore / therefore；大小写不敏感
        markers = [r"so", r"therefore"]

    # 构造形如 r"\b(?:so|therefore)\b[:,\s\-–—]*" 的模式，匹配词边界后可跟若干标点/空白
    pat = re.compile(r"\b(?:" + "|".join(markers) + r")\b[:,\s\-–—]*", flags=re.IGNORECASE)

    last_match = None
    for m in pat.finditer(text):
        last_match = m
    if last_match is None:
        return text  # 未找到标志：不改动

    # 截取到标志起始位置（不包含标志）
    truncated = text[: last_match.start()].rstrip()
    return truncated

def get_GSM8K_new(args, nsamples, seed, seqlen, tokenizer, disentangle=False, prompt="direct"):
    
    data_file = f'../data/GSM8K_eval_build/{args.model}/calibration_datasets/calibration_{prompt}_shared_120.jsonl'
    print("loading calibration data from ", data_file)
    if data_file.endswith('.jsonl'):
        with open(data_file, 'r') as fin:
            items = [json.loads(line) for line in fin]
    else:
        with open(data_file, 'r') as fin:
            items = json.load(fin)  

    # 设置随机种子
    random.seed(seed)
    sampled_items = random.sample(items, nsamples)

    trainloader = []

    for item in sampled_items:
        input_text = item['input']
        output_text = item['output']
        # else:
            # raise ValueError(f"Unsupported prompt type: {prompt}")

        # tokenizer encode
        input_enc = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=seqlen)
        output_enc = tokenizer(output_text, return_tensors="pt", truncation=True, max_length=seqlen)

        # 拼接并构造target
        inp = torch.cat((input_enc.input_ids, output_enc.input_ids[:, 1:]), dim=1)
        tar = inp.clone()

        input_len = input_enc.input_ids.shape[1]
        tar[:, :input_len] = -100  # mask掉输入部分，只监督输出
        
          # ★ 新增：attention_mask 与 position_ids
        am  = torch.ones_like(inp, dtype=torch.long)  # (1, L)
        pid = torch.arange(inp.shape[1], dtype=torch.long).unsqueeze(0)  # (1, L)

        # ★ 兜底：如果拼接长度超出 seqlen，再统一截断，保持四者形状一致
        if inp.size(1) > seqlen:
            inp  = inp[:, :seqlen]
            tar  = tar[:, :seqlen]
            am   = am[:, :seqlen]
            pid  = pid[:, :seqlen]

        trainloader.append((inp, tar, am, pid))
        # trainloader.append((inp, tar))

    return trainloader, None

def get_GSM8K(nsamples, seed, seqlen, tokenizer, disentangle=False, prompt="GSM8K_direct", truncate_answer_of_cot=False):
    if prompt == "GSM8K_direct":
        # data_file = f'../data/GSM8K/output/output.GSM8K.direct.math_teacher.llama2-7b-chat.json'
        # data_file = f'/common/users/sl2148/Public/yang_ouyang/alignment-attribution-code/data/GSM8K/output/output.GSM8K.direct.math_teacher.llama2-7b-chat.json'
        data_file = f"../data/GSM8K_eval_build/calibration_direct_120_with_conversation_template.jsonl"
    elif prompt == "GSM8K_cot0shot_120":
        # data_file = f'../data/GSM8K/output/output.GSM8K.cot0shot.goldreason.llama2-7b-chat.json'
        # data_file = f'/common/users/sl2148/Public/yang_ouyang/alignment-attribution-code/data/GSM8K/output/calibration_set_cot_120.json'
        data_file = f"../data/GSM8K_eval_build/calibration_cot0shot_120_with_conversation_template.jsonl"
    elif prompt == "GSM8K_direct_120":
        # data_file = f'../data/GSM8K/output/output.GSM8K.direct.math_teacher.llama2-7b-chat.json'
        # data_file = f'/common/users/sl2148/Public/yang_ouyang/alignment-attribution-code/data/GSM8K/output/calibration_set_direct_120.json'
        data_file = f"../data/GSM8K_eval_build/calibration_direct_120_with_conversation_template.jsonl"
    elif prompt == "GSM8K_cot0shot_goldreason":
        # data_file = f'../data/GSM8K/output/output.GSM8K.cot0shot.goldreason.llama2-7b-chat.json'
        # data_file = f'/common/users/sl2148/Public/yang_ouyang/alignment-attribution-code/data/GSM8K/output/output.GSM8K.cot0shot.goldreason.llama2-7b-chat.json'
        data_file = f"../data/GSM8K_eval_build/calibration_cot0shot_goldreason_120_with_conversation_template.jsonl"
    elif prompt == "GSM8K_cot4shot_120":
        data_file = f'../data/GSM8K_eval_build/calibration_cot4shot_120_with_conversation_template.jsonl'

    if data_file.endswith('.jsonl'):
        with open(data_file, 'r') as fin:
            items = [json.loads(line) for line in fin]
    else:
        with open(data_file, 'r') as fin:
            items = json.load(fin)  

    # 设置随机种子
    random.seed(seed)
    sampled_items = random.sample(items, nsamples)

    trainloader = []

    for item in sampled_items:
        # 选择prompt类型（如 direct / cot0shot.math teacher_input）
        # if "direct" in prompt:
        #     input_text = item['direct.math teacher_input']
        #     output_text = item['direct.math teacher_output']
        # elif prompt in ["GSM8K_cot0shot", "GSM8K_cot0shot_120" ]:
        #     input_text = item['cot0shot.math teacher_input']
        #     output_text = item['cot0shot.math teacher_output']
        # elif prompt == "GSM8K_cot0shot_goldreason":
        #     input_text = item['cot0shot.goldreason_input']
        #     output_text = item['cot0shot.goldreason_output']
        # elif prompt == "GSM8K_cot4shot_120":
        input_text = item['input']
        output_text = item['output']
        # else:
            # raise ValueError(f"Unsupported prompt type: {prompt}")

        # tokenizer encode
        input_enc = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=seqlen)
        output_enc = tokenizer(output_text, return_tensors="pt", truncation=True, max_length=seqlen)
        if truncate_answer_of_cot:
            # So 或者 Therefore or therefore 这样的标志最后出现的位置代表要给出答案了，去掉这样标志后的内容
            output_text = _truncate_at_marker(output_text)

        # 拼接并构造target
        inp = torch.cat((input_enc.input_ids, output_enc.input_ids[:, 1:]), dim=1)
        tar = inp.clone()

        input_len = input_enc.input_ids.shape[1]
        tar[:, :input_len] = -100  # mask掉输入部分，只监督输出
        
          # ★ 新增：attention_mask 与 position_ids
        am  = torch.ones_like(inp, dtype=torch.long)  # (1, L)
        pid = torch.arange(inp.shape[1], dtype=torch.long).unsqueeze(0)  # (1, L)

        # ★ 兜底：如果拼接长度超出 seqlen，再统一截断，保持四者形状一致
        if inp.size(1) > seqlen:
            inp  = inp[:, :seqlen]
            tar  = tar[:, :seqlen]
            am   = am[:, :seqlen]
            pid  = pid[:, :seqlen]

        trainloader.append((inp, tar, am, pid))
        # trainloader.append((inp, tar))

    return trainloader, None


def get_addition(nsamples, seed, seqlen, tokenizer, disentangle=False, prompt="", truncate_answer_of_cot=False):
    if prompt == "Addition:6_cot0shot":
        data_file = f"../data/Addition:6_eval_build/calibration_Addition:6_cot0shot.jsonl"
    elif prompt == "Addition:6_direct":
        data_file = f"../data/Addition:6_eval_build/calibration_Addition:6_direct.jsonl"

    if data_file.endswith('.jsonl'):
        with open(data_file, 'r') as fin:
            items = [json.loads(line) for line in fin]
    else:
        with open(data_file, 'r') as fin:
            items = json.load(fin)  

    # 设置随机种子
    random.seed(seed)
    sampled_items = random.sample(items, nsamples)

    trainloader = []

    for item in sampled_items:
        input_text = item['input']
        output_text = item['output']

        # tokenizer encode
        input_enc = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=seqlen)
        output_enc = tokenizer(output_text, return_tensors="pt", truncation=True, max_length=seqlen)
        if truncate_answer_of_cot:
            # So 或者 Therefore or therefore 这样的标志最后出现的位置代表要给出答案了，去掉这样标志后的内容
            output_text = _truncate_at_marker(output_text)

        # 拼接并构造target
        inp = torch.cat((input_enc.input_ids, output_enc.input_ids[:, 1:]), dim=1)
        tar = inp.clone()

        input_len = input_enc.input_ids.shape[1]
        tar[:, :input_len] = -100  # mask掉输入部分，只监督输出
        
        trainloader.append((inp, tar))

    return trainloader, None


# Load and process aligned dataset
def get_align(nsamples, seed, seqlen, tokenizer, disentangle=False, mode="base"):
    # Load train and test datasets
    if mode == "short":
        data_files = {"train": "../data/SFT_aligned_llama2-7b-chat-hf_train_short.csv"}
    else:
        data_files = {"train": "../data/SFT_aligned_llama2-7b-chat-hf_train.csv"}
    traindata = load_dataset("csv", data_files=data_files, split="train")
    trainloader = []
    random.seed(seed)
    if disentangle:
        traindata_sampled = traindata.shuffle(seed=seed).select(range(nsamples))
        for i in range(nsamples):
            trainenc_prompt = tokenizer(
                traindata_sampled["prompt"][i], return_tensors="pt"
            )
            trainenc_response = tokenizer(
                traindata_sampled["response"][i], return_tensors="pt"
            )
            inp = torch.cat(
                (trainenc_prompt.input_ids, trainenc_response.input_ids[:, 1:]), dim=1
            )
            tar = inp.clone()
            trainenc_prompt_len = trainenc_prompt.input_ids.shape[1]
            tar[:, :trainenc_prompt_len] = -100
            trainloader.append((inp, tar))
    else:
        # Encode datasets
        trainenc = tokenizer(" ".join(traindata["text"]), return_tensors="pt")

        # Generate samples from training set
        for _ in range(nsamples):
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
    return trainloader, None


# Load and process wikitext2 dataset
def get_wikitext2(nsamples, seed, seqlen, tokenizer):
    # Load train and test datasets
    # testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    # testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")
    traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
     # Create training dataloader
    random.seed(seed)
    trainloader = []
    trainenc = tokenizer("\n\n".join(traindata["text"]), return_tensors="pt")
    
    # Generate samples from training set
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    
    testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")
    return trainloader, testenc


def get_alpaca(nsamples, seed, seqlen, tokenizer, disentangle=False, dataset="alpaca"):
    if dataset == "alpaca":
        data_files = {"train": "../data/alpaca_train.csv"}
    elif dataset == "alpaca_cleaned":
        data_files = {"train": "../data/alpaca_cleaned_train.csv"}
    elif dataset == "alpaca_cleaned_no_safety":
        data_files = {"train": "../data/alpaca_cleaned_no_safety_train.csv"}
    else: 
        raise ValueError("Dataset not supported")
    traindata = load_dataset("csv", data_files=data_files, split="train")
    random.seed(seed)
    # Encode datasets
    trainloader = []
    if disentangle:
        traindata_sampled = traindata.shuffle(seed=seed).select(range(nsamples))
        for i in range(nsamples):
            trainenc_prompt = tokenizer(
                traindata_sampled["prompt"][i], return_tensors="pt"
            )
            trainenc_response = tokenizer(
                traindata_sampled["response"][i], return_tensors="pt"
            )
            inp = torch.cat(
                (trainenc_prompt.input_ids, trainenc_response.input_ids[:, 1:]), dim=1
            )  # to remove the first token of the response ('1')
            tar = inp.clone()
            trainenc_prompt_len = trainenc_prompt.input_ids.shape[1]
            tar[:, :trainenc_prompt_len] = -100
            trainloader.append((inp, tar))
    else:
        trainenc = tokenizer(" ".join(traindata["text"]), return_tensors="pt")
        # Generate samples from training set
        for _ in range(nsamples):
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
    return trainloader, None


# Function to select the appropriate loader based on dataset name
def get_loaders(
    args, name, nsamples=128, seed=0, seqlen=2048, tokenizer=None, disentangle=False, prompt="direct"
):
    if name == "GSM8K":
        return get_GSM8K_new(args, nsamples, seed, seqlen, tokenizer, disentangle, prompt=prompt)
    if name == "Addition:6_cot0shot":
        return get_addition(nsamples, seed, seqlen, tokenizer, disentangle, prompt="Addition:6_cot0shot")
    if name == "Addition:6_direct":
        return get_addition(nsamples, seed, seqlen, tokenizer, disentangle, prompt="Addition:6_direct")
    if name == "GSM8K_cot0shot":
        return get_GSM8K(nsamples, seed, seqlen, tokenizer, disentangle, prompt="GSM8K_cot0shot")
    if name == "GSM8K_cot4shot_120":
        return get_GSM8K(nsamples, seed, seqlen, tokenizer, disentangle, prompt="GSM8K_cot4shot_120")
    if name == "GSM8K_direct":
        return get_GSM8K(nsamples, seed, seqlen, tokenizer, disentangle, prompt="GSM8K_direct")
    if name == "GSM8K_direct_120":
        return get_GSM8K(nsamples, seed, seqlen, tokenizer, disentangle, prompt="GSM8K_direct_120")
    if name == "GSM8K_cot0shot_120":
        return get_GSM8K(nsamples, seed, seqlen, tokenizer, disentangle, prompt="GSM8K_cot0shot_120")
    if name == "GSM8K_cot0shot_goldreason":
        return get_GSM8K(nsamples, seed, seqlen, tokenizer, disentangle, prompt="GSM8K_cot0shot_goldreason")
    if name == "GSM8K_cot4shot_120_truncated":
        return get_GSM8K(nsamples, seed, seqlen, tokenizer, disentangle, prompt="GSM8K_cot4shot_120", truncate_answer_of_cot=True)
    if name == "GSM8K_cot0shot_120_truncated":
        return get_GSM8K(nsamples, seed, seqlen, tokenizer, disentangle, prompt="GSM8K_cot0shot_120", truncate_answer_of_cot=True)
    if name == "GSM8K_cot0shot_goldreason_truncated":
        return get_GSM8K(nsamples, seed, seqlen, tokenizer, disentangle, prompt="GSM8K_cot0shot_goldreason", truncate_answer_of_cot=True)
    if name == "wikitext":
        return get_wikitext2(nsamples, seed, seqlen, tokenizer)
    if name in ["alpaca", "alpaca_cleaned", "alpaca_cleaned_no_safety"]:
        return get_alpaca(nsamples, seed, seqlen, tokenizer, disentangle, dataset=name)
    if name == "align":
        return get_align(nsamples, seed, seqlen, tokenizer, disentangle=disentangle)
    if name == "align_short":
        return get_align(
            nsamples, seed, seqlen, tokenizer, disentangle=disentangle, mode="short"
        )
