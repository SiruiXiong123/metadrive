from metadrive.obs.state_obs import StateObservation, LidarStateObservation
from metadrive.obs.observation_base import BaseObservation
import os
import gymnasium as gym
from metadrive.envs.metadrive_env import MetaDriveEnv
from EgostateAndNavigation_obs import EgoStateNavigationobservation
from sb3_contrib import RecurrentPPO  # ✅ 改为 RecurrentPPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from functools import partial
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
import random
import numpy as np
from metadrive.envs.top_down_env import TopDownMetaDrive
from Multi_BEV_CNN import ImageNetBEVCNN  # ✅ 导入自定义特征提取器





def create_env(need_monitor=False):
    env = TopDownMetaDrive(cfg)
    if need_monitor:
        env = Monitor(env)
    return env

if __name__ == '__main__':
    # ✅ 使用与训练时相同的配置
    cfg = dict(
        #map="OO",
        num_scenarios=100,
        start_seed=0,
        random_lane_width=True,
        use_render=True,
        traffic_density=0.0,
        resolution_size=128,  # ✅ 改为与训练时相同的分辨率
        traffic_mode="hybrid",  # ✅ 添加训练时使用的交通模式
        vehicle_config={  # ✅ 添加训练时的vehicle_config
            "show_navi_mark": True,
            "show_line_to_dest": False,
            "show_line_to_navi_mark": True,
        },
        distance=20,  # ✅ 添加训练时的距离配置
    )

    def create_env_for_testing():
        def _env_fn():
            return TopDownMetaDrive(cfg)
        return DummyVecEnv([_env_fn])

    env = create_env_for_testing()
    
    # ✅ 模型路径检查和设置
    # 方案1：如果模型在工作区外部
    external_model_path = r"C:\Users\37945\OneDrive\Desktop\1. 有无奖励函数对比\有奖励函数\recurrent_ppo_mlp_final"
    
    # 方案2：检查工作区内的模型
    internal_model_path = os.path.join(os.getcwd(), "agent_model", "BEV_MlpLstmPolicy", "recurrent_ppo_mlp_final")
    
    # 检查哪个路径存在
    if os.path.exists(external_model_path + ".zip"):
        model_path = external_model_path
        print(f"✅ 找到外部模型: {model_path}.zip")
    elif os.path.exists(external_model_path):
        model_path = external_model_path  
        print(f"✅ 找到外部模型: {model_path}")
    elif os.path.exists(internal_model_path + ".zip"):
        model_path = internal_model_path
        print(f"✅ 找到内部模型: {model_path}.zip")
    elif os.path.exists(internal_model_path):
        model_path = internal_model_path
        print(f"✅ 找到内部模型: {model_path}")
    else:
        print("❌ 未找到模型文件！请检查路径：")
        print(f"   外部路径: {external_model_path}")
        print(f"   内部路径: {internal_model_path}")
        exit(1)

    # ✅ 使用RecurrentPPO加载，并指定正确的策略参数
    policy_kwargs = dict(
        features_extractor_class=ImageNetBEVCNN,
        features_extractor_kwargs=dict(features_dim=275), 
        normalize_images=False,
        lstm_hidden_size=256,
        n_lstm_layers=1,
        shared_lstm=False,
    )
    
    try:
        print(f"🔄 正在加载模型: {model_path}")
        model = RecurrentPPO.load(
            model_path, 
            env=env,
            policy_kwargs=policy_kwargs,  # ✅ 必须提供相同的策略参数
            device="cpu"  # 先用CPU，避免CUDA问题
        )
        print("✅ 模型加载成功！")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        exit(1)
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
        
        # ✅ 重置LSTM状态
        lstm_states = None
        episode_starts = np.ones((env.num_envs,), dtype=bool)

        while not done:
            env.render(mode="topdown")
            # ✅ 对于RecurrentPPO，需要传递LSTM状态
            action, lstm_states = model.predict(
                obs, 
                state=lstm_states,
                episode_start=episode_starts,
                deterministic=True
            )
            obs, reward, done, info = env.step(action)
            episode_starts = done  # 下一步的episode_start状态
            score += reward
            if done:
                print(f"Episode {episode} finished with score: {score}")
                break
    env.close()