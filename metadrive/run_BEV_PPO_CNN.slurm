#!/bin/bash
#SBATCH --job-name=BEV_PPO_CNN_2ch
#SBATCH --output=BEV_PPO_CNN_2ch.out
#SBATCH --error=BEV_PPO_CNN_2ch.err
#SBATCH --time=48:00:00
#SBATCH --partition=capella
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --gres=gpu:1

module purge
module load release/24.04
module load Anaconda3/2023.09-0

# Repository and script absolute paths (adjust if your repo is mounted elsewhere)
REPO_ROOT="/data/horse/ws/sixi977f-metadrive_rl/metadrive/metadrive"
PYTHON_ABS="/data/horse/ws/sixi977f-metadrive_rl/envs/metadrive_env/bin/python"
TRAIN_SCRIPT_ABS="$REPO_ROOT/envs/BEV_train.py"

echo "Using repo root: $REPO_ROOT"
echo "Using python: $PYTHON_ABS"
echo "Training script: $TRAIN_SCRIPT_ABS"

# Go to repo root (not strictly required since we use absolute paths, but useful)
cd "$REPO_ROOT"

# Limit MKL/OpenBLAS/Numexpr threads to avoid oversubscription
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Debug: print environment info (optional)
$PYTHON_ABS -c "import sys, torch, numpy, stable_baselines3; print('>>> Python:', sys.executable); print('>>> NumPy:', numpy.__version__); print('>>> Torch:', torch.__version__, 'CUDA:', torch.version.cuda, 'Available:', torch.cuda.is_available()); print('>>> SB3:', stable_baselines3.__version__)"

# Run the training script with absolute paths and arguments
$PYTHON_ABS "$TRAIN_SCRIPT_ABS" --save_dir /data/horse/ws/sixi977f-metadrive_rl/ --n_envs 4 --timesteps 3000000 --device auto
