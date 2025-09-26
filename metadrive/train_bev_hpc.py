import os
import sys
import argparse
from functools import partial
from multiprocessing import set_start_method

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList

import torch

# Ensure repository root is on sys.path so local imports work when running from other cwds
repo_root = os.path.dirname(os.path.abspath(__file__))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# Import project modules using package paths to make imports robust in subprocesses
from metadrive.envs.top_down_env import TopDownMetaDrive
from metadrive.envs.distance_and_collision_callback import MetaDriveMetricsCallback
from metadrive.envs.BEV_CNN import CustomBEVCNN


def create_env(need_monitor=False):
    env = TopDownMetaDrive(dict(
        num_scenarios=200,
        start_seed=500,
        log_level=50,
        random_lane_width=True,
        random_lane_num=True,
        use_render=False,
    ))
    if need_monitor:
        env = Monitor(env)
    return env


def main(device: str | None = None, save_base: str = "/data/horse/ws/sixi977f-metadrive_rl/", n_envs: int = 4,
         total_timesteps: int = 3_000_000):
    # Device selection
    if device is None or device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Prepare save path
    path = os.path.join(save_base, "agent_model", "BEV_CNN_default_reward_2channel")
    try:
        os.makedirs(path, exist_ok=True)
    except Exception as e:
        print(f"Warning: could not create save path {path}: {e}")

    # Quick write permission check
    try:
        testfile = os.path.join(path, ".write_test")
        with open(testfile, "w") as f:
            f.write("ok")
        os.remove(testfile)
    except Exception as e:
        print(f"Warning: no write access to {path}: {e}")
        print("Please set --save_dir to a writable path on the cluster.")

    set_random_seed(0)

    # Create vectorized environments
    env_fns = [partial(create_env, True) for _ in range(n_envs)]
    train_env = SubprocVecEnv(env_fns)

    policy_kwargs = dict(
        features_extractor_class=CustomBEVCNN,
    )

    model = PPO(
        policy="CnnPolicy",
        env=train_env,
        n_steps=2048,
        verbose=1,
        device=device,
        policy_kwargs=policy_kwargs,
        tensorboard_log=path,
    )

    # Checkpoint callback
    checkpoint_cb = CheckpointCallback(save_freq=500_000, save_path=path, name_prefix='rl_model')
    callback_list = CallbackList([checkpoint_cb, MetaDriveMetricsCallback()])

    # Train (consider running a short smoke-test first with small total_timesteps)
    model.learn(
        total_timesteps=total_timesteps,
        log_interval=4,
        callback=callback_list,
    )

    # Final save
    try:
        model.save(os.path.join(path, "ppo_topdown_bev_final"))
    except Exception as e:
        print(f"Error saving final model: {e}")

    # Also save an extra copy (keeps compatibility with earlier naming)
    try:
        model.save(os.path.join(path, "ppo_topdown_bev"))
    except Exception:
        pass

    # Close envs
    try:
        train_env.close()
    except Exception:
        pass


if __name__ == "__main__":
    # Use spawn to be safe with CUDA + multiprocessing across platforms (no-op if already set)
    try:
        set_start_method("spawn")
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser(description="Train PPO on TopDownMetaDrive with BEV CNN (HPC-friendly)")
    parser.add_argument("--save_dir", type=str, default="/data/horse/ws/sixi977f-metadrive_rl/",
                        help="Base directory to save models and tensorboard logs (must be writable)")
    parser.add_argument("--n_envs", type=int, default=4, help="Number of SubprocVecEnv workers")
    parser.add_argument("--timesteps", type=int, default=3_000_000, help="Total training timesteps")
    parser.add_argument("--device", type=str, default="auto", help="Device to use: 'auto', 'cpu', or 'cuda'")

    args = parser.parse_args()

    main(device=args.device, save_base=args.save_dir, n_envs=args.n_envs, total_timesteps=args.timesteps)
