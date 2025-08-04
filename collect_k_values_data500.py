#!/usr/bin/env python3
"""
Script to collect data for different k values and generate plots
for direct_success_cot_success and direct_fail_cot_success categories.
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple

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
        if sample_id:
            sample_info[sample_id] = {
                'correct': item.get('correct', False),
                'question': item.get('question', ''),
                'pred': item.get('pred', ''),
                'gold': item.get('gold', '')
            }
    return sample_info

def match_samples_by_question(data1: List[Dict], data2: List[Dict]) -> Dict[str, str]:
    """Match samples by question content."""
    matches = {}
    questions1 = {item.get('question', ''): item.get('id', '') for item in data1}
    
    for item in data2:
        question = item.get('question', '')
        if question in questions1:
            matches[questions1[question]] = item.get('id', '')
    
    return matches

def categorize_samples(cot_goldreason_info: Dict[str, Dict], direct_info: Dict[str, Dict], 
                      matches: Dict[str, str]) -> Tuple[List[str], List[str], List[str], List[str]]:
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

def analyze_category_performance(category_name: str, category_samples: List[str], 
                                original_cot_goldreason_data: Dict[str, Dict], 
                                original_direct_data: Dict[str, Dict], 
                                pruned_cot_goldreason_prompt_data: Dict[str, Dict], 
                                pruned_direct_prompt_data: Dict[str, Dict]) -> Dict:
    """Analyze performance for a specific category of samples."""
    if not category_samples:
        return {
            'category': category_name,
            'pruned_correct_count': 0,
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
            if sample_id in original_cot_goldreason_data and sample_id in pruned_cot_goldreason_prompt_data:
                if original_direct_data[sample_id].get('correct', False):
                    original_correct += 1
                if pruned_direct_prompt_data[sample_id].get('correct', False):
                    pruned_correct += 1
    elif category_name == "direct_fail_cot_success":
        for sample_id in category_samples:
            if sample_id in original_direct_data and sample_id in pruned_direct_prompt_data:
                if original_cot_goldreason_data[sample_id].get('correct', False):
                    original_correct += 1
                if pruned_cot_goldreason_prompt_data[sample_id].get('correct', False):
                    pruned_correct += 1
    
    return {
        'category': category_name,
        'pruned_correct_count': pruned_correct,
        'original_correct': original_correct,
        'original_accuracy': original_correct / len(category_samples) if category_samples else 0.0,
        'pruned_correct': pruned_correct,
        'pruned_accuracy': pruned_correct / len(category_samples) if category_samples else 0.0,
        'accuracy_change': (pruned_correct / len(category_samples) if category_samples else 0.0) - (original_correct / len(category_samples) if category_samples else 0.0)
    }

def collect_data_for_sparsity(sparsity_ratio: float) -> Dict:
    """Collect data for a specific sparsity ratio."""
    # File paths
    cot_goldreason_file = "out/llama2-7b-chat-hf/unstructured/wanda_weightonly/GSM8K_direct_120/sparsity_0/nsamples_500/gsm8k_bottom_0.000000_cot0shot_goldreason_held_out_prompt_cot0shot_goldreason.jsonl"
    direct_file = "out/llama2-7b-chat-hf/unstructured/wanda_weightonly/GSM8K_direct_120/sparsity_0/nsamples_500/gsm8k_bottom_0.000000_direct_held_out_prompt_direct.jsonl"
    
    # Find the directory that contains files with the target sparsity ratio
    base_dir = "out/llama2-7b-chat-hf/unstructured/wanda_3_set_difference_weightonly/GSM8K_direct_120/prompt_direct,cot0shot,cot0shot_goldreason/"
    # List all k directories
    import glob
    k_dirs = glob.glob(f"{base_dir}k_*/")
    
    target_sparsity_dir = None
    target_sparsity_value = None
    
    # Find the directory that contains files with the target sparsity ratio
    for k_dir in k_dirs:
        jsonl_files = glob.glob(f"{k_dir}*.jsonl")
        if jsonl_files:
            # Extract sparsity value from the first file
            import re
            filename = os.path.basename(jsonl_files[0])
            sparsity_match = re.search(r'gsm8k_bottom_([0-9.]+)_', filename)
            if sparsity_match:
                sparsity_value = float(sparsity_match.group(1))
                # Check if this sparsity value matches our target (with some tolerance)
                if abs(sparsity_value - sparsity_ratio) < 0.001:
                    target_sparsity_dir = k_dir
                    target_sparsity_value = sparsity_value
                    break
    
    if target_sparsity_dir is None:
        raise FileNotFoundError(f"No directory found with sparsity ratio {sparsity_ratio}")
    
    # Construct file paths with the found sparsity value
    pruned_cot_goldreason_prompt_file = f"{target_sparsity_dir}gsm8k_bottom_{target_sparsity_value}_direct,cot0shot,cot0shot_goldreason_held_out_prompt_cot0shot_goldreason.jsonl"
    pruned_direct_prompt_file = f"{target_sparsity_dir}gsm8k_bottom_{target_sparsity_value}_direct,cot0shot,cot0shot_goldreason_held_out_prompt_direct.jsonl"
    
    print(f"  Found sparsity ratio {sparsity_ratio} in directory: {os.path.basename(target_sparsity_dir.rstrip('/'))}")
    
    # Load data
    cot_goldreason_data = load_jsonl(cot_goldreason_file)
    direct_data = load_jsonl(direct_file)
    pruned_cot_goldreason_prompt_data = load_jsonl(pruned_cot_goldreason_prompt_file)
    pruned_direct_prompt_data = load_jsonl(pruned_direct_prompt_file)
    
    # Extract sample information
    cot_goldreason_info = extract_sample_info(cot_goldreason_data)
    direct_info = extract_sample_info(direct_data)
    pruned_cot_goldreason_prompt_info = extract_sample_info(pruned_cot_goldreason_prompt_data)
    pruned_direct_prompt_info = extract_sample_info(pruned_direct_prompt_data)
    
    # Match samples by question content
    matches = match_samples_by_question(cot_goldreason_data, direct_data)
    
    # Categorize samples
    direct_success_cot_success, direct_success_cot_fail, direct_fail_cot_success, direct_fail_cot_fail = categorize_samples(
        cot_goldreason_info, direct_info, matches
    )
    
    # Analyze categories
    category_samples = {
        "direct_success_cot_success": direct_success_cot_success,
        "direct_fail_cot_success": direct_fail_cot_success
    }
    
    results = {}
    for name, samples in category_samples.items():
        analysis = analyze_category_performance(name, samples, cot_goldreason_info, direct_info, 
                                              pruned_cot_goldreason_prompt_info, pruned_direct_prompt_info)
        results[name] = analysis
    
    return results

def plot_results(all_results: Dict[float, Dict]):
    """Generate plots for the results."""
    sparsity_ratios = sorted(all_results.keys())
    
    # Prepare data for plotting
    categories = ["direct_success_cot_success", "direct_fail_cot_success"]
    colors = ['blue', 'red']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    for i, category in enumerate(categories):
        original_accuracies = []
        pruned_accuracies = []
        accuracy_changes = []
        
        for sparsity_ratio in sparsity_ratios:
            if category in all_results[sparsity_ratio]:
                result = all_results[sparsity_ratio][category]
                original_accuracies.append(result['original_accuracy'])
                pruned_accuracies.append(result['pruned_accuracy'])
                accuracy_changes.append(result['accuracy_change'])
            else:
                original_accuracies.append(0)
                pruned_accuracies.append(0)
                accuracy_changes.append(0)
        
        # Plot 1: Original vs Pruned Accuracy
        ax1.plot(sparsity_ratios, original_accuracies, 'o-', label=f'{category} (Original)', color=colors[i], alpha=0.7)
        ax1.plot(sparsity_ratios, pruned_accuracies, 's-', label=f'{category} (Pruned)', color=colors[i], alpha=0.7, linestyle='--')
        
        # Plot 2: Accuracy Change
        ax2.plot(sparsity_ratios, accuracy_changes, 'o-', label=category, color=colors[i], linewidth=2, markersize=8)
    
    # Customize plots
    ax1.set_xlabel('Sparsity Ratio')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Original vs Pruned Accuracy by Sparsity Ratio')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.1, 1.1)
    
    ax2.set_xlabel('Sparsity Ratio')
    ax2.set_ylabel('Accuracy Change (Pruned - Original)')
    ax2.set_title('Accuracy Change by Sparsity Ratio')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('sparsity_ratios_analysis_500.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary table
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print(f"{'Sparsity':<12} {'Category':<25} {'Original':<10} {'Pruned':<10} {'Change':<10} {'Pruned Correct Count':<8}")
    print("-"*80)
    
    for sparsity_ratio in sparsity_ratios:
        for category in categories:
            if category in all_results[sparsity_ratio]:
                result = all_results[sparsity_ratio][category]
                print(f"{sparsity_ratio:<12} {category:<25} {result['original_accuracy']:<10.3f} {result['pruned_accuracy']:<10.3f} {result['accuracy_change']:<10.3f} {result['pruned_correct_count']:<8}")
    
    # Save detailed results
    with open('sparsity_ratios_detailed_results_500.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nDetailed results saved to: sparsity_ratios_detailed_results_500.json")
    print(f"Plot saved to: sparsity_ratios_analysis_500.png")

def main():
    """Main function to collect data for all sparsity ratios and generate plots."""
    # These are the actual sparsity ratios found in the data files
    sparsity_ratios = [0.021925, 0.054944, 0.097993, 0.145208, 0.194038]
    all_results = {}
    
    print("Collecting data for different sparsity ratios...")
    print("="*50)
    
    for sparsity_ratio in sparsity_ratios:
        print(f"Processing sparsity ratio = {sparsity_ratio}...")
        try:
            results = collect_data_for_sparsity(sparsity_ratio)
            all_results[sparsity_ratio] = results
            print(f"  ✓ Successfully collected data for sparsity ratio = {sparsity_ratio}")
            
            # Print summary for this sparsity ratio
            for category in ["direct_success_cot_success", "direct_fail_cot_success"]:
                if category in results:
                    result = results[category]
                    print(f"    {category}: {result['pruned_correct_count']} samples, "
                          f"Original: {result['original_accuracy']:.3f}, "
                          f"Pruned: {result['pruned_accuracy']:.3f}, "
                          f"Change: {result['accuracy_change']:+.3f}")
        except Exception as e:
            print(f"  ✗ Error processing sparsity ratio = {sparsity_ratio}: {e}")
            all_results[sparsity_ratio] = {}
    
    print("\nGenerating plots...")
    plot_results(all_results)

if __name__ == "__main__":
    main() 