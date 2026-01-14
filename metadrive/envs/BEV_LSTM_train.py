import os
from functools import partial
import numpy as np

# 设置环境变量允许 PyTorch 使用较新的 CUDA 架构
os.environ["TORCH_CUDA_ARCH_LIST"] = "5.0;6.0;6.1;7.0;7.5;8.0;8.6;9.0;12.0"

from sb3_contrib import RecurrentPPO  # ✅ Recurrent PPO (PPO+LSTM)

from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback  # ✅ 每 N 步自动保存

from metadrive.envs.top_down_env import TopDownMetaDrive
from distance_and_collision_callback import MetaDriveMetricsCallback  # 你自定义的 callback
from BEV_CNN import RobustBEVCNN  # 你写的模块


# === 创建 BEV 环境 ===
def create_env(need_monitor=False):
    env = TopDownMetaDrive(
        dict(
            num_scenarios=2,
            map="OO",
            start_seed=500,
            log_level=50,
            use_render=False,
            traffic_density=0.0,
            random_lane_width=True,
            random_lane_num=True,
        )
    )
    if need_monitor:
        env = Monitor(env)
    return env


if __name__ == "__main__":
    # ✅ 统一存放：tensorboard 日志 + checkpoints + 最终模型
    path = "/data/horse/ws/sixi977f-sixi977f-topconference/CNN_BEV(LSTM)_test"
    os.makedirs(path, exist_ok=True)

    set_random_seed(0)

    # 使用 SubprocVecEnv 并行创建环境
    train_env = SubprocVecEnv([partial(create_env, True) for _ in range(4)])  # 4 个并行环境

    # ✅ policy 参数
    policy_kwargs = dict(
        features_extractor_class=RobustBEVCNN,
        lstm_hidden_size=256,
        n_lstm_layers=1,
        shared_lstm=True,
        enable_critic_lstm=True,
    )

    # ✅ 每 10 万步保存一次模型（注意：save_freq 是“环境步”，不是“迭代次数”）
    checkpoint_callback = CheckpointCallback(
        save_freq=100_000,
        save_path=path,               # checkpoints 也放同一路径
        name_prefix="ppo_bev_lstm",    # 生成 ppo_bev_lstm_100000_steps.zip 这种文件
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    # ✅ RecurrentPPO (CnnLstmPolicy)
    model = RecurrentPPO(
        policy="CnnLstmPolicy",
        env=train_env,
        n_steps=128,
        verbose=1,
        device="cuda",
        policy_kwargs=policy_kwargs,
        tensorboard_log=path,          # tensorboard 日志也在同一路径下
    )

    # ✅ 组合 callback：既记录指标又定期保存 checkpoint
    model.learn(
        total_timesteps=1_000_000,
        log_interval=4,
        callback=[checkpoint_callback, MetaDriveMetricsCallback()],
    )

    # ✅ 保存最终模型（建议）
    model.save(os.path.join(path, "ppo_bev_lstm_final"))
