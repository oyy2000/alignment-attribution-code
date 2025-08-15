#!/usr/bin/env python3
"""
Example script demonstrating the use of the combined collect_data_for_all_sparsity_ratios function.
This function combines the functionality of discover_sparsity_ratios and collect_data_for_sparsity.
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple
import glob
import re

# Import the necessary functions from the main script
from collect_k_values_data_500_cot4shot_pqku import (
    load_jsonl, extract_sample_info, match_samples_by_question, 
    categorize_samples, analyze_category_performance, collect_data_for_original_model,
    plot_results, collect_data_for_all_sparsity_ratios
)

def example_usage():
    """
    Example usage of the combined function.
    """
    print("Example: Using combined collect_data_for_all_sparsity_ratios function")
    print("="*70)
    
    # Set the base directory
    base_dir = "/common/users/sl2148/Public/yang_ouyang/alignment-attribution-code/out/llama2-7b-chat-hf/unstructured/wanda_4_set_difference_cot4shot_weightonly/4_set_alpaca_cleaned_no_safety/prompt_direct,cot4shot/"
    
    # Step 1: Collect data for all sparsity ratios in one go
    print("Step 1: Collecting data for all sparsity ratios...")
    sparsity_results = collect_data_for_all_sparsity_ratios(base_dir)
    
    print(f"\nCollected data for {len(sparsity_results)} sparsity ratios:")
    for sparsity_ratio in sorted(sparsity_results.keys()):
        print(f"  - Sparsity {sparsity_ratio:.6f}: {len(sparsity_results[sparsity_ratio])} categories")
    
    # Step 2: Add original model data (0% sparsity)
    print("\nStep 2: Adding original model data (0% sparsity)...")
    try:
        original_results = collect_data_for_original_model()
        sparsity_results[0.0] = original_results
        print("  ✓ Successfully added original model data")
    except Exception as e:
        print(f"  ✗ Error adding original model data: {e}")
        sparsity_results[0.0] = {}
    
    # Step 3: Generate plots and save results
    print("\nStep 3: Generating plots and saving results...")
    plot_results(sparsity_results)
    
    # Step 4: Print summary statistics
    print("\nStep 4: Summary Statistics")
    print("="*50)
    
    categories = ["direct_success_cot_success", "direct_fail_cot_success"]
    sparsity_ratios = sorted(sparsity_results.keys())
    
    for category in categories:
        print(f"\n{category}:")
        print(f"{'Sparsity':<12} {'Original':<10} {'Pruned':<10} {'Change':<10}")
        print("-" * 45)
        
        for sparsity_ratio in sparsity_ratios:
            if category in sparsity_results[sparsity_ratio]:
                result = sparsity_results[sparsity_ratio][category]
                print(f"{sparsity_ratio:<12} {result['original_accuracy']:<10.3f} "
                      f"{result['pruned_accuracy']:<10.3f} {result['accuracy_change']:<10.3f}")
    
    return sparsity_results

def compare_approaches():
    """
    Compare the old approach (separate functions) vs new approach (combined function).
    """
    print("\nComparison: Old vs New Approach")
    print("="*50)
    
    base_dir = "/common/users/sl2148/Public/yang_ouyang/alignment-attribution-code/out/llama2-7b-chat-hf/unstructured/wanda_4_set_difference_cot4shot_weightonly/4_set_alpaca_cleaned_no_safety/prompt_direct,cot4shot/"
    
    # Old approach (separate functions)
    print("Old approach:")
    print("1. discover_sparsity_ratios(base_dir) - finds all sparsity ratios")
    print("2. For each sparsity_ratio:")
    print("   collect_data_for_sparsity(sparsity_ratio, base_dir) - processes one ratio")
    print("3. Multiple file reads and redundant processing")
    
    # New approach (combined function)
    print("\nNew approach:")
    print("1. collect_data_for_all_sparsity_ratios(base_dir) - does everything in one go")
    print("2. Single pass through directories")
    print("3. More efficient file handling")
    print("4. Better error handling and reporting")

if __name__ == "__main__":
    # Run the example
    results = example_usage()
    
    # Show comparison
    compare_approaches()
    
    print(f"\nExample completed successfully!")
    print(f"Results saved to: sparsity_ratios_detailed_results_500_cot4shot.json")
    print(f"Plot saved to: sparsity_ratios_analysis_500_cot4shot.png")
