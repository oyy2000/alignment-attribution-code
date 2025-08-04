#!/usr/bin/env python3
"""
Script to analyze the impact of pruning on different types of samples.
Compares performance between original and pruned models, categorizing samples
into four categories based on direct query and Golden CoT performance.
Uses only held_out files as specified by the user.
"""

import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple, Set

def load_jsonl(file_path: str) -> List[Dict]:
    """Load data from a JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def extract_sample_info(data: List[Dict]) -> Dict[str, Dict]:
    """Extract sample information from data."""
    sample_info = {}
    for item in data:
        sample_id = item.get('id', '')
        sample_info[sample_id] = {
            'correct': item.get('correct', False),
            'pred': item.get('pred', ''),
            'gold': item.get('gold', ''),
            'question': item.get('question', ''),
            'prompt_tag': item.get('prompt_tag', '')
        }
    return sample_info

def match_samples_by_question(original_data: List[Dict], direct_data: List[Dict]) -> Dict[str, str]:
    """Match samples between original and direct data by question content."""
    # Create a mapping from question to sample ID for direct data
    direct_question_to_id = {}
    for item in direct_data:
        question = item.get('question', '').strip()
        if question:
            direct_question_to_id[question] = item.get('id', '')
    
    # Match original samples to direct samples
    matches = {}
    for item in original_data:
        question = item.get('question', '').strip()
        if question in direct_question_to_id:
            original_id = item.get('id', '')
            direct_id = direct_question_to_id[question]
            matches[original_id] = direct_id
    
    return matches

def categorize_samples(original_info: Dict[str, Dict], direct_info: Dict[str, Dict], 
                      matches: Dict[str, str]) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    Categorize samples into four categories:
    1. Direct Query Succeeds & Golden CoT Succeeds
    2. Direct Query Succeeds & Golden CoT Fails
    3. Direct Query Fails & Golden CoT Succeeds
    4. Direct Query Fails & Golden CoT Fails
    """
    direct_success_cot_success = []
    direct_success_cot_fail = []
    direct_fail_cot_success = []
    direct_fail_cot_fail = []
    
    for original_id, direct_id in matches.items():
        if direct_id in direct_info:
            direct_correct = direct_info[direct_id]['correct']
            cot_correct = original_info[original_id]['correct']
            
            if direct_correct and cot_correct:
                # Category 1: Direct Query Succeeds & Golden CoT Succeeds
                direct_success_cot_success.append(original_id)
            elif direct_correct and not cot_correct:
                # Category 2: Direct Query Succeeds & Golden CoT Fails
                direct_success_cot_fail.append(original_id)
            elif not direct_correct and cot_correct:
                # Category 3: Direct Query Fails & Golden CoT Succeeds
                direct_fail_cot_success.append(original_id)
            else:
                # Category 4: Direct Query Fails & Golden CoT Fails
                direct_fail_cot_fail.append(original_id)
    
    return direct_success_cot_success, direct_success_cot_fail, direct_fail_cot_success, direct_fail_cot_fail

def analyze_category_performance(category_name: str, category_samples: List[str], original_data: Dict[str, Dict], 
                                direct_data: Dict[str, Dict], pruned_cot_goldreason_prompt_data: Dict[str, Dict], pruned_direct_prompt_data: Dict[str, Dict]) -> Dict:
    """Analyze performance for a specific category of samples."""
    if not category_samples:
        return {
            'category': category_name,
            'count': 0,
            'original_correct': 0,
            'original_accuracy': 0.0,
            'pruned_correct': 0,
            'pruned_accuracy': 0.0,
            'accuracy_change': 0.0
        }
    
    original_correct = 0
    pruned_correct = 0
    
    if category_name == "direct_success_cot_success":
        for sample_id in category_samples:
            if sample_id in original_data and sample_id in pruned_cot_goldreason_prompt_data:
                # Check if original model got it correct
                if original_data[sample_id].get('correct', False):
                    original_correct += 1
                if pruned_cot_goldreason_prompt_data[sample_id].get('correct', False):
                    pruned_correct += 1
    elif category_name == "direct_fail_cot_success":
        for sample_id in category_samples:
            if sample_id in original_data and sample_id in pruned_direct_prompt_data:
                if original_data[sample_id].get('correct', False):
                    original_correct += 1
                if pruned_direct_prompt_data[sample_id].get('correct', False):
                    pruned_correct += 1
    
    return {
        'category': category_name,
        'count': len(category_samples), 
        'original_correct': original_correct,
        'original_accuracy': original_correct / len(category_samples) if category_samples else 0.0,
        'pruned_correct': pruned_correct,
        'pruned_accuracy': pruned_correct / len(category_samples) if category_samples else 0.0,
        'accuracy_change': (pruned_correct / len(category_samples) if category_samples else 0.0) - (original_correct / len(category_samples) if category_samples else 0.0)
    }

def main():
    # File paths - using held_out files as specified
    cot_goldreason_file = "out/llama2-7b-chat-hf/unstructured/wanda_weightonly/GSM8K_cot0shot_120/sparsity_0/gsm8k_bottom_0.000000_cot0shot_goldreason_held_out.jsonl"
    direct_file = "out/llama2-7b-chat-hf/unstructured/wanda_weightonly/GSM8K_cot0shot_120/sparsity_0/gsm8k_bottom_0.000000_direct_held_out.jsonl"
    pruned_cot_goldreason_prompt_file = "out/llama2-7b-chat-hf/unstructured/wanda_3_set_difference_weightonly_previous4/GSM8K_cot0shot_120/prompt_direct,cot0shot,cot0shot_goldreason/k_0.4/gsm8k_bottom_0.097993_direct,cot0shot,cot0shot_goldreason_held_out_prompt_cot0shot_goldreason.jsonl"
    pruned_direct_prompt_file = "out/llama2-7b-chat-hf/unstructured/wanda_3_set_difference_weightonly_previous4/GSM8K_cot0shot_120/prompt_direct,cot0shot,cot0shot_goldreason/k_0.4/gsm8k_bottom_0.097993_direct,cot0shot,cot0shot_goldreason_held_out_prompt_direct.jsonl"
    
    print("Loading held_out data files...")
    
    # Load data
    cot_goldreason_data = load_jsonl(cot_goldreason_file)
    direct_data = load_jsonl(direct_file)
    pruned_cot_goldreason_prompt_data = load_jsonl(pruned_cot_goldreason_prompt_file)
    pruned_direct_prompt_data = load_jsonl(pruned_direct_prompt_file)
    
    print(f"Loaded {len(cot_goldreason_data)} original held_out samples")
    print(f"Loaded {len(direct_data)} direct held_out samples")
    print(f"Loaded {len(pruned_cot_goldreason_prompt_data)} pruned held_out samples")
    print(f"Loaded {len(pruned_direct_prompt_data)} pruned held_out samples")
    # Extract sample information
    cot_goldreason_info = extract_sample_info(cot_goldreason_data)
    direct_info = extract_sample_info(direct_data)
    pruned_cot_goldreason_prompt_info = extract_sample_info(pruned_cot_goldreason_prompt_data)
    pruned_direct_prompt_info = extract_sample_info(pruned_direct_prompt_data)
    
    # Match samples by question content
    print("Matching samples by question content...")
    matches = match_samples_by_question(cot_goldreason_data, direct_data)
    print(f"Found {len(matches)} matching samples")
    
    # Categorize samples into four categories
    direct_success_cot_success, direct_success_cot_fail, direct_fail_cot_success, direct_fail_cot_fail = categorize_samples(
        cot_goldreason_info, direct_info, matches
    )
    
    print(f"\nSample Categorization:")
    print(f"  Direct Query Succeeds & Golden CoT Succeeds: {len(direct_success_cot_success)} samples")
    print(f"  Direct Query Succeeds & Golden CoT Fails: {len(direct_success_cot_fail)} samples")
    print(f"  Direct Query Fails & Golden CoT Succeeds: {len(direct_fail_cot_success)} samples")
    print(f"  Direct Query Fails & Golden CoT Fails: {len(direct_fail_cot_fail)} samples")
    
    # Analyze performance for each category
    print("\n" + "="*80)
    print("PERFORMANCE ANALYSIS (HELD_OUT SAMPLES)")
    print("="*80)
    
  
    
    category_samples = {
        "direct_success_cot_success": direct_success_cot_success,
        "direct_success_cot_fail": direct_success_cot_fail,
        "direct_fail_cot_success": direct_fail_cot_success,
        "direct_fail_cot_fail": direct_fail_cot_fail
    }
    
    category_analyses = []
    for name, samples in category_samples.items():
        analysis = analyze_category_performance(name, samples, cot_goldreason_info, direct_info, pruned_cot_goldreason_prompt_info, pruned_direct_prompt_info)
        category_analyses.append(analysis)
    
    
    for i, (name, analysis) in enumerate(zip(category_samples.keys(), category_analyses)):
        print(f"{i+1}. {name}:")
        print(f"  Sample count: {len(category_samples[name])}")
        if len(category_samples[name]) > 0:
            print(f"  Original accuracy: {analysis['original_accuracy']:.3f} ({analysis['original_correct']}/{len(category_samples[name])})")
            print(f"  Pruned accuracy: {analysis['pruned_accuracy']:.3f} ({analysis['pruned_correct']}/{len(category_samples[name])})")
            print(f"  Accuracy change: {analysis['accuracy_change']:+.3f}")
        else:
            print("  No samples in this category")
        print()
    
  
    
    # Save detailed results
    results = {
        'categories': {}
    }
    
    for name, analysis in zip(category_samples.keys(), category_analyses):
        results['categories'][name] = {
            'sample_count': len(category_samples[name]),
            'sample_ids': category_samples[name],
            'original_accuracy': analysis['original_accuracy'],
            'original_correct': analysis['original_correct'],
            'pruned_accuracy': analysis['pruned_accuracy'],
            'pruned_correct': analysis['pruned_correct'],
            'accuracy_change': analysis['accuracy_change']
        }
    
    with open('pruning_impact_analysis_held_out.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("Detailed results saved to: pruning_impact_analysis_held_out.json")

if __name__ == "__main__":
    main() 