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


def extract_logic(answer):
    pattern1 = r"correct \w+ is:?\s*([A-D])"
    pattern2 = r"correct option is: (true|false|unknown)"
    pattern3 = r"([A-C])\)\s*(True|False|Unknown)"
    pattern4 = r"([A-D])\) "
    pattern5 = r"^[A-D]\.?$"

    match = re.search(pattern1, answer)
    option = None
    # extract pattern
    if match:
        option = match.group(1)
    
    if not option:
        match = re.search(pattern2, answer, re.IGNORECASE)
        if match:
            word_to_option = {"true": "A", "false": "B", "unknown": "C"}
            option = word_to_option.get(match.group(1).lower())

    if not option:
        match = re.search(pattern3, answer, re.IGNORECASE)
        if match:
            option = match.group(1)
    if not option and len(answer)<16:
        if 'true' in answer.lower():
            option = 'A'
        elif 'false' in answer.lower():
            option = 'B'
        elif 'unknown' in answer.lower():
            option = 'C'
    if not option:
        match = re.match(pattern4, answer)
        if match:
            option = match.group(1)
    if not option:
        match = re.match(pattern5, answer)
        if match:
            option = match.group(0) 
    if not option:
        option = None
        # wrong_data.append(d)
    return option

def add_bias_sentence(prompt, bias_sentence):
    pattern = r"(#|##) Reasoning"

    matches = list(re.finditer(pattern, prompt))

    if not matches:
        return prompt+bias_sentence+'\n'

    # 获取最后一个匹配项的位置
    last_match = matches[-1].start()
    return prompt[:last_match] + bias_sentence+ '\n' + prompt[last_match:]

def make_n_shot(dataset,template,nshot):
    demonstration_file = f'./data/{dataset}/train.json'
    demonstration_data = json.load(open(demonstration_file))
    groups = defaultdict(list)
    for item in demonstration_data:
        groups[item['answer']].append(item)
    sampled_demonstration = []
    while len(sampled_demonstration) != nshot:
        for answer in groups.keys(): 
            selected_item = random.choice(groups[answer])
            sampled_demonstration.append(selected_item)
            if len(sampled_demonstration) == nshot:
                break
    random.shuffle(sampled_demonstration)
    rendered_text = template.render(demonstrations=sampled_demonstration)
    return rendered_text
    

def load_prompt(dataset, prompt, do_role='match teacher', do_bias='nobias'):
    # load prompt file
    dataset = dataset.split(':')[0]
    if '_' in dataset:
        dataset = dataset.split('_')[0]
    if dataset == 'ProofWriter' and prompt != 'direct':
        numbers = re.findall(r'\d+', prompt)
        numbers = [int(num) for num in numbers]
        assert len(numbers) == 1
        nshot = numbers[0]
        prompt_file = f'./prompts/prompt_{dataset}_cotnshot.jinja'
        with open(prompt_file, 'r') as fin:
            template_content = fin.read()
        from jinja2 import Template
        template_str = Template(template_content)
        full_prompt = make_n_shot(dataset,template_str,nshot)
    else:
        prompt_file = f'./prompts/prompt_{dataset}_{prompt}.txt'
        with open(prompt_file, 'r') as fin:
            lines = [line.strip() for line in fin.readlines()]
        full_prompt = '\n'.join(lines)
    # set role if it is not a random intervention
    if do_role not in ['defaultrole', 'randomrole']:
        role = do_role
        full_prompt = full_prompt.replace('{{role}}', role)
    # add bias prompt for random intervention
    is_math = dataset in ['Addition', 'Product', 'GSM8K']
    if do_bias == 'strongbias':
        bias_sentence = 'I think the correct answer is: {{biasanswer}}' \
                        if is_math else 'I think the correct option is: {{biasoption}}'
        full_prompt = add_bias_sentence(full_prompt, bias_sentence)
    elif do_bias == 'weakbais':
        bias_sentence = 'I think the correct answer is: {{biasanswer}}, but I’m curious to hear what you think.' \
                        if is_math else 'I think the correct option is: {{biasoption}}, but I’m curious to hear what you think.'
        full_prompt = add_bias_sentence(full_prompt, bias_sentence)
    elif do_bias == 'nobias':
        pass
    else:
        raise NotImplementedError

    return full_prompt

def format_prompt(full_prompt, item):
    fields = re.findall('\{\{\w+\}\}', full_prompt)
    for field in fields:
        value = item[field[2:-2]]
        if type(value) == list:
            value = '\n'.join(value)
        full_prompt = full_prompt.replace(field, value)
    assert full_prompt.find('{{') < 0 and full_prompt.find('}}') < 0
    return full_prompt

def load_dataset_GSM8k(dataset, nsamples):
    if dataset == 'GSM8K':
        data_file = f'./data/{dataset}/test.jsonl'
        with open(data_file, 'r') as fin:
            items = [json.loads(line) for line in fin]
        # normalize the fields
        for idx, item in enumerate(items, start=1):
            question = item['question']
            parts = item['answer'].split('####')
            item.clear()
            item['id'] = f'GSM8K_Q{idx}'
            item['question'] = question
            item['reason'] = parts[0].strip()
            item['answer'] = str(int(parts[1].strip().replace(',', '')))  # expect integer only
        random.shuffle(items)
        return items[:nsamples] if nsamples > 0 else items[:500]  # default 500 samples
    else:  # default loading
        if dataset.find(':') > 0:
            dataset, arg = dataset.split(':')
            data_file = f'./data/{dataset}/dev{arg}.json'
        else:
            data_file = f'./data/{dataset}/dev.json'
        with open(data_file, 'r') as fin:
            items = json.load(fin)
        random.shuffle(items)
        return items[:nsamples] if nsamples > 0 else items

def extract_answer(output, item, dataset):
    try:
        dataset = dataset.split(':')[0]
        if dataset in ['Addition', 'Product', 'GSM8K']:
            gold = item['answer']
            output = output.split('\n')
            output = [line for line in output if len(re.findall('\d+', line)) > 0][-1]
            answer = output.replace(',', '')  # remove middle ',' from numbers like '1,234'
            answer = re.findall('\d+', answer)
            answer = gold if gold in answer else answer[-1]
            answer = answer.strip()
            return str(int(answer))  # expect integer only
        elif dataset.startswith('ProofWriter'):
            answer = extract_logic(output)
            return str(answer)
        elif dataset.startswith('LOGIQA'):
            answer = extract_logic(output)
            return str(answer)
        elif dataset.startswith('FOLIO'):
            answer = extract_logic(output)
            return str(answer)
    except Exception as ex:
        # LLMs may constantly generate wrong output, let's skip the retry and give it a None result.
        print('extract_answer:', ex)
        return str(None)

    raise NotImplemented

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

import json
import torch
import random

def get_GSM8K(nsamples, seed, seqlen, tokenizer, disentangle=False, prompt="GSM8K_direct", truncate_answer_of_cot=False):
    if prompt == "GSM8K_direct":
        # data_file = f'../data/GSM8K/output/output.GSM8K.direct.math_teacher.llama2-7b-chat.json'
        # data_file = f'/common/users/sl2148/Public/yang_ouyang/alignment-attribution-code/data/GSM8K/output/output.GSM8K.direct.math_teacher.llama2-7b-chat.json'
        data_file = f"/common/users/sl2148/Public/yang_ouyang/alignment-attribution-code/data/GSM8K_eval_build/calibration_direct_120_with_conversation_template.jsonl"
    elif prompt == "GSM8K_cot0shot_120":
        # data_file = f'../data/GSM8K/output/output.GSM8K.cot0shot.goldreason.llama2-7b-chat.json'
        # data_file = f'/common/users/sl2148/Public/yang_ouyang/alignment-attribution-code/data/GSM8K/output/calibration_set_cot_120.json'
        data_file = f"/common/users/sl2148/Public/yang_ouyang/alignment-attribution-code/data/GSM8K_eval_build/calibration_cot0shot_120_with_conversation_template.jsonl"
    elif prompt == "GSM8K_direct_120":
        # data_file = f'../data/GSM8K/output/output.GSM8K.direct.math_teacher.llama2-7b-chat.json'
        # data_file = f'/common/users/sl2148/Public/yang_ouyang/alignment-attribution-code/data/GSM8K/output/calibration_set_direct_120.json'
        data_file = f"/common/users/sl2148/Public/yang_ouyang/alignment-attribution-code/data/GSM8K_eval_build/calibration_direct_120_with_conversation_template.jsonl"
    elif prompt == "GSM8K_cot0shot_goldreason":
        # data_file = f'../data/GSM8K/output/output.GSM8K.cot0shot.goldreason.llama2-7b-chat.json'
        # data_file = f'/common/users/sl2148/Public/yang_ouyang/alignment-attribution-code/data/GSM8K/output/output.GSM8K.cot0shot.goldreason.llama2-7b-chat.json'
        data_file = f"/common/users/sl2148/Public/yang_ouyang/alignment-attribution-code/data/GSM8K_eval_build/calibration_cot0shot_goldreason_120_with_conversation_template.jsonl"
    elif prompt == "GSM8K_cot4shot_120":
        data_file = f'/common/users/sl2148/Public/yang_ouyang/alignment-attribution-code/data/GSM8K_eval_build/calibration_cot4shot_120_with_conversation_template.jsonl'

    if data_file.endswith('.jsonl'):
        with open(data_file, 'r') as fin:
            items = [json.loads(line) for line in fin]
    else:
        with open(data_file, 'r') as fin:
            items = json.load(fin)  

    # 规范化字段
    for idx, item in enumerate(items, start=1):
        # id 标准化
        item['id'] = f'GSM8K_Q{idx}'
        # 答案清洗：取出整数部分
        if 'answer' in item:
            item['answer'] = str(int(item['answer'].strip().replace(',', '')))

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
        
        
        trainloader.append((inp, tar))

    return trainloader, None


def get_addition(nsamples, seed, seqlen, tokenizer, disentangle=False):
    data_files = {"train": "./data/addition_direct_train.csv"}
    # input = "1 + 1 = 2\n2 + 2 = 4\n3 + 3 = 6\n4 + 4 = 8\n5 + 5 = 10"


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
        data_files = {"train": "./data/alpaca_train.csv"}
    elif dataset == "alpaca_cleaned":
        data_files = {"train": "./data/alpaca_cleaned_train.csv"}
    elif dataset == "alpaca_cleaned_no_safety":
        data_files = {"train": "./data/alpaca_cleaned_no_safety_train.csv"}
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
    name, nsamples=128, seed=0, seqlen=2048, tokenizer=None, disentangle=False, prompt="direct"
):
    if name == "addition_direct":
        return get_addition(nsamples, seed, seqlen, tokenizer, disentangle, dataset="addition_direct")
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
