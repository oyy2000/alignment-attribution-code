#!/usr/bin/env python3
"""
Script to collect data for different pq_ku values and generate plots
for direct_success_cot_success and direct_fail_cot_success categories.
Extracts k and u values from folder names and sparsity from JSONL files.
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np
import re
import glob
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
        'accuracy_change': (pruned_correct / original_correct) - 1 if original_correct > 0 else 0.0
    }

def parse_folder_name(folder_name: str) -> Tuple[float, float, float, float]:
    """Parse folder name to extract pq, k, and u values."""
    # Example: pq_0.1_0.1_k_0.1_u_0.1
    pattern = r'pq_([0-9.]+)_([0-9.]+)_k_([0-9.]+)_u_([0-9.]+)'
    match = re.match(pattern, folder_name)
    if match:
        pq1, pq2, k, u = map(float, match.groups())
        return pq1, pq2, k, u
    else:
        raise ValueError(f"Could not parse folder name: {folder_name}")

def extract_sparsity_from_folder(folder_path: str) -> float:
    """Extract sparsity value from JSONL files in the folder."""
    jsonl_files = glob.glob(f"{folder_path}/*.jsonl")
    if not jsonl_files:
        raise FileNotFoundError(f"No JSONL files found in {folder_path}")
    
    # Look for the cot0shot_goldreason file to extract sparsity
    target_file = None
    for file in jsonl_files:
        if 'cot0shot_goldreason' in file:
            target_file = file
            break
    
    if not target_file:
        # If not found, use the first JSONL file
        target_file = jsonl_files[0]
    
    # Extract sparsity from filename
    filename = os.path.basename(target_file)
    sparsity_match = re.search(r'gsm8k_bottom_([0-9.]+)_', filename)
    
    if sparsity_match:
        return float(sparsity_match.group(1))
    else:
        raise ValueError(f"Could not extract sparsity from filename: {filename}")

def collect_data_for_pq_ku_folder(folder_path: str) -> Dict:
    """Collect data for a specific pq_ku folder."""
    folder_name = os.path.basename(folder_path.rstrip('/'))
    
    # Parse folder name to get pq, k, u values
    pq1, pq2, k, u = parse_folder_name(folder_name)
    
    # Extract sparsity from the folder
    sparsity = extract_sparsity_from_folder(folder_path)
    
    # File paths for original data (baseline)
    cot_goldreason_file = "out/llama2-7b-chat-hf/unstructured/wanda_weightonly/GSM8K_direct_120/sparsity_0/nsamples_500/gsm8k_bottom_0.000000_cot0shot_goldreason_held_out_prompt_cot0shot_goldreason.jsonl"
    direct_file = "out/llama2-7b-chat-hf/unstructured/wanda_weightonly/GSM8K_direct_120/sparsity_0/nsamples_500/gsm8k_bottom_0.000000_direct_held_out_prompt_direct.jsonl"
    
    # File paths for pruned data
    pruned_cot_goldreason_prompt_file = f"{folder_path}/gsm8k_bottom_{sparsity}_direct,cot0shot,cot0shot_goldreason_held_out_prompt_cot0shot_goldreason.jsonl"
    pruned_direct_prompt_file = f"{folder_path}/gsm8k_bottom_{sparsity}_direct,cot0shot,cot0shot_goldreason_held_out_prompt_direct.jsonl"
    
    print(f"  Processing folder: {folder_name}")
    print(f"    pq1={pq1}, pq2={pq2}, k={k}, u={u}, sparsity={sparsity}")
    
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
    
    # Add metadata
    results['metadata'] = {
        'folder_name': folder_name,
        'pq1': pq1,
        'pq2': pq2,
        'k': k,
        'u': u,
        'sparsity': sparsity
    }
    
    return results

def plot_results(all_results: Dict[str, Dict], pq_values: List[float]):
    """Generate plots for the results with k and u annotations."""
    # Filter out results without metadata (failed processing)
    valid_results = {k: v for k, v in all_results.items() if 'metadata' in v}
    
    if not valid_results:
        print("No valid results to plot!")
        return
    
    # Sort results by sparsity
    sorted_results = sorted(valid_results.items(), key=lambda x: x[1]['metadata']['sparsity'])
    
    # Prepare data for plotting
    categories = ["direct_success_cot_success", "direct_fail_cot_success"]
    colors = ['blue', 'red']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    for i, category in enumerate(categories):
        sparsities = []
        original_accuracies = []
        pruned_accuracies = []
        accuracy_changes = []
        annotations = []
        
        for folder_name, result in sorted_results:
            if category in result:
                sparsities.append(result['metadata']['sparsity'])
                original_accuracies.append(result[category]['original_accuracy'])
                pruned_accuracies.append(result[category]['pruned_accuracy'])
                accuracy_changes.append(result[category]['accuracy_change'])
                
                # Create annotation text
                k_val = result['metadata']['k']
                u_val = result['metadata']['u']
                annotations.append(f"k={k_val}\nu={u_val}")
        
        # Plot 1: Original vs Pruned Accuracy
        ax1.plot(sparsities, original_accuracies, 'o-', label=f'{category} (Original)', color=colors[i], alpha=0.7)
        ax1.plot(sparsities, pruned_accuracies, 's-', label=f'{category} (Pruned)', color=colors[i], alpha=0.7, linestyle='--')
        
        # Add annotations for k and u values
        for j, (sparsity, acc, annotation) in enumerate(zip(sparsities, original_accuracies, annotations)):
            ax1.annotate(annotation, (sparsity, acc), 
                        xytext=(5, 5), textcoords='offset points', 
                        fontsize=8, alpha=0.8,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        
        # Plot 2: Accuracy Change
        ax2.plot(sparsities, accuracy_changes, 'o-', label=category, color=colors[i], linewidth=2, markersize=8)
        
        # Add annotations for k and u values
        for j, (sparsity, change, annotation) in enumerate(zip(sparsities, accuracy_changes, annotations)):
            ax2.annotate(annotation, (sparsity, change), 
                        xytext=(5, 5), textcoords='offset points', 
                        fontsize=8, alpha=0.8,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    # Customize plots
    ax1.set_xlabel('Sparsity Ratio')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Original vs Pruned Accuracy by Sparsity Ratio\n(Annotated with k and u values)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.1, 1.1)
    
    ax2.set_xlabel('Sparsity Ratio')
    ax2.set_ylabel('Accuracy Change (Pruned - Original)')
    ax2.set_title('Accuracy Change by Sparsity Ratio\n(Annotated with k and u values)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'pq_ku_sparsity_analysis_500_{pq_values}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary table
    print("\n" + "="*100)
    print("SUMMARY TABLE")
    print("="*100)
    print(f"{'Folder':<30} {'Sparsity':<10} {'k':<6} {'u':<6} {'Category':<25} {'Original':<10} {'Pruned':<10} {'Change':<10}")
    print("-"*100)
    
    for folder_name, result in sorted_results:
        sparsity = result['metadata']['sparsity']
        k_val = result['metadata']['k']
        u_val = result['metadata']['u']
        
        for category in categories:
            if category in result:
                cat_result = result[category]
                print(f"{folder_name:<30} {sparsity:<10.6f} {k_val:<6.1f} {u_val:<6.1f} {category:<25} "
                      f"{cat_result['original_correct']:<10} {cat_result['pruned_correct']:<10} {cat_result['accuracy_change']:<10.3f}")
    
    # Save detailed results
    with open(f'pq_ku_sparsity_detailed_results_500_{pq_values}.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nDetailed results saved to: pq_ku_sparsity_detailed_results_500_{pq_values}.json")
    print(f"Plot saved to: pq_ku_sparsity_analysis_500_{pq_values}.png")

def main():
    """Main function to collect data for all pq_ku folders and generate plots."""
    base_dir = "out/llama2-7b-chat-hf/unstructured/wanda_3_set_difference_utility_weightonly/4_set_alpaca_cleaned_no_safety/prompt_direct,cot0shot,cot0shot_goldreason"
    
    pq_values = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
    for pq in pq_values:
        # Find all pq_ku folders
        pq_ku_folders = []
        for item in os.listdir(base_dir):
            item_path = os.path.join(base_dir, item)
            if os.path.isdir(item_path) and item.startswith(f'pq_{pq}') and '_k_' not in item and '_u_' not in item:
                pq_ku_folders.append(item_path)
        
        if not pq_ku_folders:
            raise FileNotFoundError("No pq_ku folders found in the specified directory")
        
        print(f"Found {len(pq_ku_folders)} pq_ku folders")
        print("="*50)
        
        all_results = {}
        
        for folder_path in pq_ku_folders:
            folder_name = os.path.basename(folder_path)
            print(f"Processing folder: {folder_name}")
            
            try:
                results = collect_data_for_pq_ku_folder(folder_path)
                all_results[folder_name] = results
                print(f"  ✓ Successfully collected data for {folder_name}")
                
                # Print summary for this folder
                for category in ["direct_success_cot_success", "direct_fail_cot_success"]:
                    if category in results:
                        result = results[category]
                        print(f"    {category}: {result['pruned_correct_count']} samples, "
                            f"Original: {result['original_accuracy']:.3f}, "
                            f"Pruned: {result['pruned_accuracy']:.3f}, "
                            f"Change: {result['accuracy_change']:+.3f}")
            except Exception as e:
                print(f"  ✗ Error processing folder {folder_name}: {e}")
                all_results[folder_name] = {}
        
        print("\nGenerating plots...")
        plot_results(all_results, pq)

if __name__ == "__main__":
    main()
