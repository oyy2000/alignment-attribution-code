#!/usr/bin/env python3
"""
Debug script to check data loading and matching logic
"""

import json
import os

# File paths
ORIGINAL_COT4SHOT_FILE = "/common/users/sl2148/Public/yang_ouyang/alignment-attribution-code/out/llama2-7b-chat-hf/cot4shot/eval_selected_samples/gsm8k_bottom_0.000000_cot4shot_selected_samples_prompt_cot4shot.jsonl"
ORIGINAL_DIRECT_FILE = "/common/users/sl2148/Public/yang_ouyang/alignment-attribution-code/out/llama2-7b-chat-hf/direct/eval_selected_samples/gsm8k_bottom_0.000000_direct_selected_samples_prompt_direct.jsonl"

def load_jsonl(file_path: str):
    """Load data from a JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def extract_sample_info(data):
    """Extract sample information from data."""
    sample_info = {}
    for item in data:
        sample_id = item.get('id', '')
        if sample_id:
            sample_info[sample_id] = {
                'correct': item.get('correct', False),
                'question': item.get('question', ''),
                'pred': item.get('pred', ''),
                'gold': item.get('gold', '')
            }
    return sample_info

def match_samples_by_question(data1, data2):
    """Match samples by question content."""
    matches = {}
    questions1 = {item.get('question', ''): item.get('id', '') for item in data1}
    
    for item in data2:
        question = item.get('question', '')
        if question in questions1:
            matches[questions1[question]] = item.get('id', '')
    
    return matches

def categorize_samples(cot_goldreason_info, direct_info, matches):
    """Categorize samples into four categories."""
    direct_success_cot_success = []
    direct_success_cot_fail = []
    direct_fail_cot_success = []
    direct_fail_cot_fail = []
    
    for cot_id, direct_id in matches.items():
        cot_correct = cot_goldreason_info.get(cot_id, {}).get('correct', False)
        direct_correct = direct_info.get(direct_id, {}).get('correct', False)
        
        if direct_correct and cot_correct:
            direct_success_cot_success.append(cot_id)
        elif direct_correct and not cot_correct:
            direct_success_cot_fail.append(cot_id)
        elif not direct_correct and cot_correct:
            direct_fail_cot_success.append(cot_id)
        else:
            direct_fail_cot_fail.append(cot_id)
    
    return direct_success_cot_success, direct_success_cot_fail, direct_fail_cot_success, direct_fail_cot_fail

def main():
    print("Loading data...")
    
    # Load data
    cot4shot_data = load_jsonl(ORIGINAL_COT4SHOT_FILE)
    direct_data = load_jsonl(ORIGINAL_DIRECT_FILE)
    
    print(f"COT4Shot data: {len(cot4shot_data)} samples")
    print(f"Direct data: {len(direct_data)} samples")
    
    # Extract sample information
    cot4shot_info = extract_sample_info(cot4shot_data)
    direct_info = extract_sample_info(direct_data)
    
    print(f"COT4Shot info: {len(cot4shot_info)} samples")
    print(f"Direct info: {len(direct_info)} samples")
    
    # Match samples by question content
    matches = match_samples_by_question(cot4shot_data, direct_data)
    print(f"Matches: {len(matches)} samples")
    
    # Show some examples
    print("\nFirst few matches:")
    for i, (cot_id, direct_id) in enumerate(list(matches.items())[:5]):
        print(f"  {cot_id} <-> {direct_id}")
        print(f"    COT4Shot correct: {cot4shot_info[cot_id]['correct']}")
        print(f"    Direct correct: {direct_info[direct_id]['correct']}")
    
    # Categorize samples
    direct_success_cot_success, direct_success_cot_fail, direct_fail_cot_success, direct_fail_cot_fail = categorize_samples(
        cot4shot_info, direct_info, matches
    )
    
    print(f"\nCategories:")
    print(f"  direct_success_cot_success: {len(direct_success_cot_success)}")
    print(f"  direct_success_cot_fail: {len(direct_success_cot_fail)}")
    print(f"  direct_fail_cot_success: {len(direct_fail_cot_success)}")
    print(f"  direct_fail_cot_fail: {len(direct_fail_cot_fail)}")
    
    # Check some samples in each category
    print(f"\nSample IDs in direct_success_cot_success: {direct_success_cot_success[:5]}")
    print(f"Sample IDs in direct_fail_cot_success: {direct_fail_cot_success[:5]}")
    
    # Test the analysis logic
    print(f"\nTesting analysis logic...")
    
    # For direct_success_cot_success category
    original_correct = 0
    for sample_id in direct_success_cot_success:
        if sample_id in cot4shot_info and sample_id in direct_info:
            if direct_info[sample_id].get('correct', False):
                original_correct += 1
    
    print(f"direct_success_cot_success - original_correct: {original_correct}/{len(direct_success_cot_success)}")
    
    # For direct_fail_cot_success category
    original_correct = 0
    for sample_id in direct_fail_cot_success:
        if sample_id in direct_info and sample_id in cot4shot_info:
            if cot4shot_info[sample_id].get('correct', False):
                original_correct += 1
    
    print(f"direct_fail_cot_success - original_correct: {original_correct}/{len(direct_fail_cot_success)}")

if __name__ == "__main__":
    main()
