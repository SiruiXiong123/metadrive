from metadrive.envs import MetaDriveEnv
from stable_baselines3 import TD3
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from functools import partial
import numpy as np
from metadrive.component.sensors.rgb_camera import RGBCamera
from metadrive.constants import DEFAULT_AGENT

# 自定义 observation （如果需要的话）
from EgostateAndNavigation_obs import EgoStateNavigationobservation

sensor_size = (200, 100)

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
        stack_size=3,
    ))
    if need_monitor:
        env = Monitor(env)
    return env

if __name__ == "__main__":
    # 保存路径
    path = r"C:\Users\37945\OneDrive\Desktop"
    #path = r"/home/h1/sixi977f/agent_model/td3_metadrive"
    set_random_seed(0)

    # 启动多进程环境
    train_env = SubprocVecEnv([partial(create_env, True) for _ in range(2)])

    # 确定动作空间维度
    n_actions = train_env.action_space.shape[-1]
    # 定义动作噪声（TD3 需要）
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    # 初始化 TD3
    model = TD3(
        "MultiInputPolicy",
        train_env,
        buffer_size=10000,
        action_noise=action_noise,
        verbose=1,
        device="cuda",
        tensorboard_log=path,
    )

    # 训练
    model.learn(total_timesteps=2_000_000, log_interval=4)
    model.save(path)
