#!/usr/bin/env python3
import os
import shutil
import glob

def move_single_txt_folders():
    # Source directory
    source_dir = "out/llama2-7b-chat-hf/unstructured/wanda_3_set_difference_utility_weightonly/4_set_alpaca_cleaned_no_safety/prompt_direct,cot0shot,cot0shot_goldreason"
    
    # Destination directory
    dest_dir = "temp"
    
    # Ensure temp directory exists
    os.makedirs(dest_dir, exist_ok=True)
    
    # Get all pq_* folders
    pq_folders = glob.glob(os.path.join(source_dir, "pq_*"))
    
    moved_count = 0
    
    for folder_path in pq_folders:
        folder_name = os.path.basename(folder_path)
        
        # Get all files in the folder
        files = os.listdir(folder_path)
        
        # Count txt files
        txt_files = [f for f in files if f.endswith('.txt')]
        
        # If folder contains only one txt file and no other files
        if len(files) == 1 and len(txt_files) == 1:
            print(f"Moving {folder_name} (contains only: {txt_files[0]})")
            
            # Move the folder to temp
            dest_path = os.path.join(dest_dir, folder_name)
            shutil.move(folder_path, dest_path)
            moved_count += 1
    
    print(f"\nTotal folders moved: {moved_count}")

if __name__ == "__main__":
    move_single_txt_folders()
