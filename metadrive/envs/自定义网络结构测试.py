from metadrive.obs.state_obs import StateObservation, LidarStateObservation
from metadrive.obs.observation_base import BaseObservation
import os
import gymnasium as gym
from metadrive.envs.metadrive_env import MetaDriveEnv
from EgostateAndNavigation_obs import EgoStateNavigationobservation
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from functools import partial
from stable_baselines3 import PPO
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
import random
from metadrive.component.sensors.rgb_camera import RGBCamera
from envs.only_VIT_singleframe import CustomViTPolicy

random.seed(123)


sensor_size = (224, 224)

def create_env(need_monitor=False):
    env = MetaDriveEnv(cfg)
    if need_monitor:
        env = Monitor(env)
    return env

if __name__ == '__main__':
    cfg = dict(
        map="CC",
        num_scenarios=100,
        start_seed=0,
        random_lane_width=True,
        use_render=True,
        traffic_density=0.0,
        #discrete_action=True,
        #use_multi_discrete=True, 
        #traffic_mode="hybrid",
        image_observation=True,
        vehicle_config=dict(image_source="rgb_camera"),
        sensors={"rgb_camera": (RGBCamera, *sensor_size)},
        stack_size=4,
    )

    def create_env_for_testing():
        def _env_fn():
            return MetaDriveEnv(cfg)
        return DummyVecEnv([_env_fn])

    env = create_env_for_testing()
    PPO_Path = r"C:\Users\37945\OneDrive\Desktop\PPO_BASE_CCC.zip"


    model = PPO.load(PPO_Path, env=env)
    # episode_rewards, episode_infos = evaluate_policy(
    #     model,
    #     env,
    #     n_eval_episodes=100,  # 评估 100 次
    #     deterministic=True,
    #     render=False,
    #     return_episode_rewards=True  # 返回每个 episode 的奖励和信息
    # )




    episodes = 5
    for episode in range(1, episodes + 1):
        obs = env.reset()  # VecEnv 的 reset 返回直接是 obs
        done = False
        score = 0

        while not done:
            env.render(mode="topdown")
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)  # VecEnv 的 step 返回 dones
            score += reward
            if done:
                break
    env.close()