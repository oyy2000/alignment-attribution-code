#!/bin/bash 
#SBATCH -J exp_product_wanda # 作业名 
#SBATCH -w c[32]
#SBATCH -p a5000ada # 分区（改成你想用的，比如 a6000）
#SBATCH --cpus-per-task=16 # CPU 核数（给 Python 线程用） 
#SBATCH --time=2-00:00:00 # 最长运行时间（2 天） 
#SBATCH --output=/home/%u/logs/%x-%j.out
#SBATCH --error=/home/%u/logs/%x-%j.err
source activate prune_llm 
nvidia-smi
python /home/youyang7/projects/alignment-attribution-code/exp_product/exp_script_main.py
# /home/youyang7/projects/alignment-attribution-code/exp_script_4_set_454_0.01_use_current_cards.py