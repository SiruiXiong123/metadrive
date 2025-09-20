import os
from functools import partial
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

from metadrive.envs.top_down_env import TopDownMetaDrive
from distance_and_collision_callback import MetaDriveMetricsCallback  # 你自定义的 callback
from BEV_CNN import CustomBEVCNN  # 你刚写的模块


# === 创建 BEV 环境 ===
def create_env(need_monitor=False):
    env = TopDownMetaDrive(dict(
        num_scenarios=200,
        start_seed=500,
        log_level=50,
        use_render=False,
        random_lane_width=True,
        random_lane_num=True,

    ))
    if need_monitor:
        env = Monitor(env)
    return env

if __name__ == "__main__":
    # import matplotlib.pyplot as plt

    # env = create_env(need_monitor=False)
    # obs, _ = env.reset()
    # print("type of obs:", type(obs))

    # # 如果是 dict 就取 "image"
    # if isinstance(obs, dict) and "image" in obs:
    #     bev = obs["image"]  # shape = (H, W, C)
    # else:
    #     bev = obs  # shape = (84, 84, 5)

    # print("BEV shape:", bev.shape)

    # # --- 可视化所有通道 ---
    # n_channels = bev.shape[-1]
    # plt.figure(figsize=(15, 3))
    # for i in range(n_channels):
    #     plt.subplot(1, n_channels, i+1)
    #     plt.imshow(bev[..., i], cmap="gray")   # 每个通道单独显示
    #     plt.title(f"Channel {i}")
    #     plt.axis("off")
    # plt.show()

    # x = 1/0  # 断点

    path = "./ppo_topdown_bev_logs_分支注意力"
    os.makedirs(path, exist_ok=True)

    set_random_seed(0)

    # 使用 SubprocVecEnv 并行创建环境
    train_env = SubprocVecEnv([partial(create_env, True) for _ in range(4)])  # 4 个并行环境

    policy_kwargs = dict(
    features_extractor_class=CustomBEVCNN,)
    
    # 使用默认 CnnPolicy
    model = PPO(
        policy="CnnPolicy",
        env=train_env,
        n_steps=2048,
        verbose=1,
        device="cuda",
        policy_kwargs=policy_kwargs,
        tensorboard_log=path
    )

    # 训练
    model.learn(
        total_timesteps=3000000,
        log_interval=4,
        callback=MetaDriveMetricsCallback()
    )

    # 保存模型
    model.save(os.path.join(path, "ppo_topdown_bev"))
