import os
from functools import partial

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList

from metadrive.envs.top_down_env import TopDownMetaDrive
from distance_and_collision_callback import MetaDriveMetricsCallback  # user callback
from BEV_CNN import CustomBEVCNN  # feature extractor


def create_env(need_monitor=False):
    env = TopDownMetaDrive(dict(
        map="OO",
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


def main():
    # Save path (adjust as you like)
    path = os.path.join(os.getcwd(), "agent_model", "BEV_CNN_default_reward")
    os.makedirs(path, exist_ok=True)

    set_random_seed(0)

    # Create vectorized environments
    n_envs = 4
    train_env = SubprocVecEnv([partial(create_env, True) for _ in range(n_envs)])

    policy_kwargs = dict(
        features_extractor_class=CustomBEVCNN,
    )

    model = PPO(
        policy="CnnPolicy",
        env=train_env,
        n_steps=2048,
        verbose=1,
        device="cuda",
        policy_kwargs=policy_kwargs,
        tensorboard_log=path,
    )

    # Checkpoint callback: save every 500,000 timesteps (50W = 500k)
    checkpoint_cb = CheckpointCallback(save_freq=500_000, save_path=path, name_prefix='rl_model')

    # Combine with your metrics callback
    callback_list = CallbackList([checkpoint_cb, MetaDriveMetricsCallback()])

    # Train
    total_timesteps = 3_000_000
    model.learn(
        total_timesteps=total_timesteps,
        log_interval=4,
        callback=callback_list,
    )

    # Final save
    model.save(os.path.join(path, "ppo_topdown_bev_final"))


if __name__ == "__main__":
    main()
