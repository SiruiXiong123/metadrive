import os
from functools import partial

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList

from metadrive.envs.top_down_env import TopDownMetaDrive
from distance_and_collision_callback import MetaDriveMetricsCallback  # user callback
from envs.Multi_BEV_CNN import CustomBEVCNN  # feature extractor
from sb3_contrib import RecurrentPPO


cfg = {
    "num_scenarios": 500,
    "start_seed": 123,
    "random_lane_width": True,
    "random_lane_num": False,
    "use_render": False,
    "traffic_density": 0.0,
    "traffic_mode": "hybrid",
    "manual_control": False,
    "controller": "keyboard",
    "vehicle_config": {
        "show_navi_mark": True,
        "show_line_to_dest": False,
        "show_line_to_navi_mark": True,
    },
    "distance": 30,
    "resolution_size": 128,
}

def create_env(need_monitor=False):
    env = TopDownMetaDrive(cfg)
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

    checkpoint_cb = CheckpointCallback(save_freq=100_000, save_path=path, name_prefix='rl_model')

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
