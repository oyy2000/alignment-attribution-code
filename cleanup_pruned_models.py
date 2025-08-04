#!/usr/bin/env python3
"""
Script to clean up pruned model folders to save disk space.
This script can delete pruned model directories that are no longer needed.
"""

import os
import shutil
import argparse
from pathlib import Path
import glob


def get_folder_size(folder_path):
    """Calculate the total size of a folder in bytes."""
    total_size = 0
    try:
        for dirpath, dirnames, filenames in os.walk(folder_path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.exists(filepath):
                    total_size += os.path.getsize(filepath)
    except Exception as e:
        print(f"Error calculating size for {folder_path}: {e}")
    return total_size


def format_size(size_bytes):
    """Convert bytes to human readable format."""
    if size_bytes == 0:
        return "0B"
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    return f"{size_bytes:.2f}{size_names[i]}"


def find_pruned_model_folders(base_path, pattern=None):
    """Find all pruned model folders and wanda_score folders matching the pattern."""
    folders_to_delete = []
    
    # Search for pruned_model directories
    search_pattern = os.path.join(base_path, "**", "pruned_model")
    all_pruned_dirs = glob.glob(search_pattern, recursive=True)
    
    for pruned_dir in all_pruned_dirs:
        if os.path.isdir(pruned_dir):
            # Check if it contains pruned model subdirectories
            subdirs = [d for d in os.listdir(pruned_dir) 
                      if os.path.isdir(os.path.join(pruned_dir, d)) 
                      and d.startswith("pruned_model_")]
            
            if subdirs:
                for subdir in subdirs:
                    full_path = os.path.join(pruned_dir, subdir)
                    if pattern is None or pattern in full_path:
                        folders_to_delete.append(full_path)
    
    # Search for wanda_score directories (these use most disk space)
    wanda_score_pattern = os.path.join(base_path, "**", "wanda_score")
    all_wanda_score_dirs = glob.glob(wanda_score_pattern, recursive=True)
    
    for wanda_score_dir in all_wanda_score_dirs:
        if os.path.isdir(wanda_score_dir):
            if pattern is None or pattern in wanda_score_dir:
                folders_to_delete.append(wanda_score_dir)
    
    return folders_to_delete


def delete_pruned_models(base_path, pattern=None, dry_run=True, confirm=False):
    """Delete pruned model folders matching the pattern."""
    pruned_folders = find_pruned_model_folders(base_path, pattern)
    
    if not pruned_folders:
        print("No pruned model folders found matching the pattern.")
        return
    
    print(f"Found {len(pruned_folders)} pruned model folders:")
    total_size = 0
    
    for folder in pruned_folders:
        size = get_folder_size(folder)
        total_size += size
        print(f"  {folder} ({format_size(size)})")
    
    print(f"\nTotal size to be freed: {format_size(total_size)}")
    
    if dry_run:
        print("\nThis was a dry run. Use --execute to actually delete the folders.")
        return
    
    if not confirm:
        response = input(f"\nAre you sure you want to delete these {len(pruned_folders)} folders? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            print("Operation cancelled.")
            return
    
    deleted_count = 0
    freed_space = 0
    
    for folder in pruned_folders:
        try:
            size = get_folder_size(folder)
            shutil.rmtree(folder)
            print(f"Deleted: {folder} ({format_size(size)})")
            deleted_count += 1
            freed_space += size
        except Exception as e:
            print(f"Error deleting {folder}: {e}")
    
    print(f"\nSuccessfully deleted {deleted_count} folders.")
    print(f"Total space freed: {format_size(freed_space)}")


def main():
    parser = argparse.ArgumentParser(description="Clean up pruned model folders to save disk space")
    parser.add_argument("--base-path", default="out", 
                       help="Base path to search for pruned models (default: out)")
    parser.add_argument("--pattern", default=None,
                       help="Pattern to match in folder paths (e.g., 'GSM8K_cot0shot_120')")
    parser.add_argument("--execute", action="store_true",
                       help="Actually delete the folders (default is dry run)")
    parser.add_argument("--confirm", action="store_true",
                       help="Skip confirmation prompt")
    parser.add_argument("--list-only", action="store_true",
                       help="Only list folders without deleting")
    
    args = parser.parse_args()
    
    base_path = os.path.join(os.getcwd(), args.base_path)
    
    if not os.path.exists(base_path):
        print(f"Base path does not exist: {base_path}")
        return
    
    print(f"Searching for pruned model folders in: {base_path}")
    if args.pattern:
        print(f"Pattern filter: {args.pattern}")
    
    if args.list_only:
        pruned_folders = find_pruned_model_folders(base_path, args.pattern)
        if pruned_folders:
            print(f"\nFound {len(pruned_folders)} pruned model folders:")
            total_size = 0
            for folder in pruned_folders:
                size = get_folder_size(folder)
                total_size += size
                print(f"  {folder} ({format_size(size)})")
            print(f"\nTotal size: {format_size(total_size)}")
        else:
            print("No pruned model folders found.")
        return
    
    delete_pruned_models(
        base_path=base_path,
        pattern=args.pattern,
        dry_run=not args.execute,
        confirm=args.confirm
    )


if __name__ == "__main__":
    main() 