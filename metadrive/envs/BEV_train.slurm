#!/bin/bash
#SBATCH --job-name=metadrive_bev_train
#SBATCH --partition=capella
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# ===== 0. 建议：让脚本遇错就停（避免悄悄失败）=====
set -e

# ===== 1. 进入你的工作目录（存日志的地方）=====
cd /data/horse/ws/sixi977f-sixi977f-topconference/metadrive/metadrive/envs || exit 1

# ===== 2. 激活 conda 环境（路径型环境）=====
source /software/genoa/r24.04/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate /data/horse/ws/sixi977f-sixi977f-topconference/metadrive_rl

# ===== 3. 限制 CPU 线程（推荐）=====
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

# ===== 4. 打印 GPU 信息（用于日志确认）=====
nvidia-smi

# ===== 5. （强烈建议）打印 Python / 环境信息，方便定位问题 =====
which python
python -V

# ===== 6. 运行你的训练脚本 =====
python BEV_train.py
