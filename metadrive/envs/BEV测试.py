from metadrive.obs.state_obs import StateObservation, LidarStateObservation
from metadrive.obs.observation_base import BaseObservation
import os
import time
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
from typing import List
from metadrive.envs.top_down_env import TopDownMetaDrive
from Multi_BEV_CNN import ImageNetBEVCNN  # ✅ 导入自定义特征提取器


def get_bev_hwc(obs):
    """Return HWC uint8 array with values in [0,255].
    Accepts either the raw observation array or a dict with key "image".
    """
    x = obs["image"] if isinstance(obs, dict) and "image" in obs else obs
    x = np.asarray(x)
    
    # Handle 4D case (batch dimension): take first sample
    if x.ndim == 4:
        x = x[0]  # Remove batch dimension
    
    if x.ndim != 3:
        raise ValueError(f"Expect 3D BEV after removing batch dim, got {x.shape}")
    
    # If channels-first (C,H,W) -> transpose to HWC
    if x.shape[0] in (2, 3, 4, 5) and x.shape[-1] not in (2, 3, 4, 5):
        x = np.transpose(x, (1, 2, 0))
    
    # If float [0,1] -> convert to uint8
    if x.dtype == np.float32 or x.max() <= 1.0:
        x_u8 = (np.clip(x, 0.0, 1.0) * 255).astype(np.uint8)
    else:
        x_u8 = x.astype(np.uint8)
    return x_u8


def make_output_path(base_dir='recordings'):
    os.makedirs(base_dir, exist_ok=True)
    ts = time.strftime('%Y%m%d_%H%M%S')
    filename = f"BEV测试_录制_{ts}.npz"
    return os.path.join(base_dir, filename)


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
    external_model_path = r"C:\Users\37945\OneDrive\Desktop\2.stage2奖励函数修改\base+ckpt\rl_model_400000_steps"
    # 方案2：检查工作区内的模型
    internal_model_path = os.path.join(os.getcwd(), "agent_model", "BEV_MlpLstmPolicy", "recurrent_ppo_mlp_final")
    
    # 检查哪个路径存在
    if os.path.exists(external_model_path + ".zip"):
        model_path = external_model_path
        # print(f"✅ 找到外部模型: {model_path}.zip")
    elif os.path.exists(external_model_path):
        model_path = external_model_path  
        # print(f"✅ 找到外部模型: {model_path}")
    elif os.path.exists(internal_model_path + ".zip"):
        model_path = internal_model_path
        # print(f"✅ 找到内部模型: {model_path}.zip")
    elif os.path.exists(internal_model_path):
        model_path = internal_model_path
        # print(f"✅ 找到内部模型: {model_path}")
    else:
        # print("❌ 未找到模型文件！请检查路径：")
        # print(f"   外部路径: {external_model_path}")
        # print(f"   内部路径: {internal_model_path}")
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
        # print(f"🔄 正在加载模型: {model_path}")
        model = RecurrentPPO.load(
            model_path, 
            env=env,
            policy_kwargs=policy_kwargs,  # ✅ 必须提供相同的策略参数
            device="cpu"  # 先用CPU，避免CUDA问题
        )
        # print("✅ 模型加载成功！")
    except Exception as e:
        # print(f"❌ 模型加载失败: {e}")
        exit(1)
    # episode_rewards, episode_infos = evaluate_policy(
    #     model,
    #     env,
    #     n_eval_episodes=100,  # 评估 100 次
    #     deterministic=True,
    #     render=False,
    #     return_episode_rewards=True  # 返回每个 episode 的奖励和信息
    # )




    # 初始化数据收集列表
    out_file = make_output_path()
    frames: List[np.ndarray] = []
    rewards: List[float] = []
    dones: List[bool] = []
    infos: List[object] = []
    actions: List[np.ndarray] = []
    
    # print(f'开始测试并记录数据，将保存到: {out_file}')
    
    try:
        episodes = 5
        total_step = 0
        
        for episode in range(1, episodes + 1):
            obs = env.reset()  # VecEnv 的 reset 返回直接是 obs
            done = False
            score = 0
            
            # ✅ 重置LSTM状态
            lstm_states = None
            episode_starts = np.ones((env.num_envs,), dtype=bool)
            
            print(f'开始Episode {episode}')

            while not done:
                env.render(mode="topdown")
                
                # ✅ 对于RecurrentPPO，需要传递LSTM状态
                action, lstm_states = model.predict(
                    obs, 
                    state=lstm_states,
                    episode_start=episode_starts,
                    deterministic=True
                )
                
                # 保存当前观测值
                try:
                    # 对于VecEnv，obs是一个数组，需要取第一个环境的观测
                    current_obs = obs[0] if isinstance(obs, (list, np.ndarray)) and len(obs) > 0 else obs
                    
                    # 添加调试信息
                    if total_step == 1:  # 只在第一步打印详细信息
                        print(f"调试信息 - 观测值类型: {type(current_obs)}")
                        if isinstance(current_obs, dict):
                            print(f"观测值字典键: {list(current_obs.keys())}")
                            for key, value in current_obs.items():
                                if hasattr(value, 'shape'):
                                    print(f"  {key}: shape={value.shape}, dtype={getattr(value, 'dtype', 'unknown')}")
                                    if hasattr(value, 'min'):
                                        print(f"    range: [{np.min(value):.3f}, {np.max(value):.3f}]")
                        else:
                            print(f"观测值形状: {getattr(current_obs, 'shape', 'No shape')}")
                            if hasattr(current_obs, 'dtype'):
                                print(f"观测值范围: [{np.min(current_obs):.3f}, {np.max(current_obs):.3f}]")
                    
                    bev = get_bev_hwc(current_obs)
                    
                    # 检查转换后的结果
                    if total_step == 1:
                        print(f"转换后BEV形状: {bev.shape}, 范围: [{bev.min()}, {bev.max()}]")
                    
                    # 只保留RGB三个通道
                    if bev.shape[-1] > 3:
                        bev = bev[..., :3]
                    frames.append(bev.copy())
                    
                except Exception as e:
                    print(f'Failed to convert observation to RGB BEV: {e}')
                    print(f'观测值类型: {type(current_obs)}, 内容: {current_obs}')
                    # 如果转换失败，添加一个空白帧
                    frames.append(np.zeros((128, 128, 3), dtype=np.uint8))
                
                obs, reward, done, info = env.step(action)
                episode_starts = done  # 下一步的episode_start状态
                
                # 保存其他数据
                current_reward = reward[0] if isinstance(reward, (list, np.ndarray)) and len(reward) > 0 else reward
                current_done = done[0] if isinstance(done, (list, np.ndarray)) and len(done) > 0 else done
                current_info = info[0] if isinstance(info, (list, np.ndarray)) and len(info) > 0 else info
                current_action = action[0] if isinstance(action, (list, np.ndarray)) and len(action) > 0 else action
                
                rewards.append(float(current_reward) if current_reward is not None else 0.0)
                dones.append(bool(current_done))
                infos.append(current_info if current_info is not None else {})
                actions.append(current_action.copy() if isinstance(current_action, np.ndarray) else current_action)
                
                score += current_reward
                total_step += 1
                
                print(f"Episode {episode}, Step {total_step}, Reward: {current_reward:.4f}, Done: {current_done}")
                
                if current_done:
                    print(f"Episode {episode} finished with score: {score}")
                    break
                    
    except KeyboardInterrupt:
        print('用户中断 (Ctrl+C)。正在保存已收集的数据...')
    finally:
        env.close()
        
        # 保存收集到的数据
        if len(frames) == 0:
            print('没有收集到帧数据，不保存文件。')
        else:
            try:
                arr = np.stack(frames, axis=0)  # T,H,W,3 uint8
                np.savez_compressed(
                    out_file, 
                    frames=arr, 
                    rewards=np.array(rewards), 
                    dones=np.array(dones), 
                    infos=infos,
                    actions=np.array(actions) if len(actions) > 0 else []
                )
                print(f'✅ 保存了 {arr.shape[0]} 帧数据到: {out_file}')
                print(f'   - 帧形状: {arr.shape}')
                print(f'   - 奖励数量: {len(rewards)}')
                print(f'   - 平均奖励: {np.mean(rewards):.4f}')
            except Exception as e:
                print(f'❌ 保存数据时出错: {e}')