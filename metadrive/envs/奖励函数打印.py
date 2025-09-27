from metadrive.obs.state_obs import StateObservation, LidarStateObservation
from metadrive.obs.observation_base import BaseObservation
import os
import gymnasium as gym
from metadrive.envs.metadrive_env import MetaDriveEnv
from stable_baselines3 import TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from functools import partial
import numpy as np
from EgostateAndNavigation_obs import EgoStateNavigationobservation
import random
from stable_baselines3.common.callbacks import CheckpointCallback, EveryNTimesteps
from win32ui import ID_FILE_LOCATE
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise


cfg = {
    "map": "OO",
    # "num_scenarios": 500,
    # "start_seed": 123,
    "random_lane_width": True,
    "random_lane_num": False,
    "use_render": True,
    "traffic_density": 0.0,
    "traffic_mode": "hybrid",
    "manual_control": True,
    "controller": "keyboard",
    "vehicle_config": {
        "show_navi_mark": True,
        "show_line_to_dest": False,
        "show_line_to_navi_mark": True,
    },
}

env=MetaDriveEnv(cfg)

num_episodes = 3

for episode in range(num_episodes):
    obs = env.reset()
    done = False
    step_count = 0

    while not done:
        # 使用键盘控制车辆，等待用户输入
        action = env.action_space.sample()  # 这里的 action 会被键盘输入覆盖

        obs, reward, done, info, _ = env.step(action)
        # 打印当前时间步的奖励值
        step_count += 1   # 每步+1



# 关闭环境
env.close()