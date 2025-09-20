from metadrive.envs import MetaDriveEnv
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor
from functools import partial
import numpy as np
import os

from metadrive.component.sensors.rgb_camera import RGBCamera
from metadrive.constants import DEFAULT_AGENT
from EgostateAndNavigation_obs import EgoStateNavigationobservation

# 🚀 设置参数
sensor_size = (200, 100)
stack_size = 3               # 叠帧数
n_envs = 2                   # 并行环境数
buffer_size = 10_000         # replay buffer 大小
total_timesteps = 2_000_000  # 训练步数

# 🚀 确保工作目录在资源文件夹
os.chdir(r"/home/h1/sixi977f/agent_model/sac_metadrive")

def create_env(need_monitor=False):
    env = MetaDriveEnv(dict(
        num_scenarios=500,
        start_seed=500,
        traffic_density=0.0,
        log_level=50,
        image_observation=True,
        random_lane_width=True,
        random_lane_num=True,
        vehicle_config=dict(image_source="rgb_camera"),
        sensors={"rgb_camera": (RGBCamera, *sensor_size)},
        stack_size=stack_size,
    ))
    if need_monitor:
        env = Monitor(env)
    return env


if __name__ == "__main__":
    # 🚀 保存路径
    path = r"C:\Users\37945\OneDrive\Desktop\sac_metadrive"
    set_random_seed(0)

    # 🚀 启动并行环境
    train_env = SubprocVecEnv([partial(create_env, True) for _ in range(n_envs)])

    # 🚀 初始化 SAC
    model = SAC(
        "MultiInputPolicy",
        train_env,
        buffer_size=buffer_size,
        verbose=1,
        device="cuda",
        tensorboard_log=path
    )

    # 🚀 训练
    model.learn(total_timesteps=total_timesteps, log_interval=4)

    # 🚀 保存
    model.save(path)
