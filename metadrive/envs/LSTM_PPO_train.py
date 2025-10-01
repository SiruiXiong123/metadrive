import os
from functools import partial
import numpy as np

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList

from metadrive.envs.top_down_env import TopDownMetaDrive
from distance_and_collision_callback import MetaDriveMetricsCallback
from Multi_BEV_CNN import ImageNetBEVCNN 



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


# ==================== 主程序 ====================
def main():
    # 📁 模型保存路径
    save_path = os.path.join(os.getcwd(), "agent_model", "BEV_MlpLstmPolicy")
    os.makedirs(save_path, exist_ok=True)

    set_random_seed(0)

    # ✅ 多环境并行
    n_envs = 4
    train_env = SubprocVecEnv([partial(create_env, True) for _ in range(n_envs)])

    # Diagnostic: inspect vehicle_config used by the subprocess envs (take first env)
    try:
        first_cfg = train_env.get_attr('config')[0]
        vc = first_cfg.get('vehicle_config', {})
        side_det = vc.get('side_detector', None)
        lane_det = vc.get('lane_line_detector', None)
        print("[DIAG] vehicle_config (first env):", vc)
        print(f"[DIAG] side_detector={side_det}, lane_line_detector={lane_det}")
    except Exception as e:
        print("[DIAG] Failed to read env config from SubprocVecEnv:", e)

    # ✅ LSTM PPO 策略参数
    policy_kwargs = dict(
        features_extractor_class=ImageNetBEVCNN,          # 你的特征提取器
        # extractor returns image features (256) + state (19) = 275, make LSTM expect the full vector
        features_extractor_kwargs=dict(features_dim=275), # CNN输出长度（拼接后=275）
        normalize_images=False,                        # 已在提取器中归一化
        lstm_hidden_size=256,                          # LSTM 隐藏层维度
        n_lstm_layers=1,
        shared_lstm=False,
    )

    # ✅ 构建 RecurrentPPO 模型
    model = RecurrentPPO(
        policy="MlpLstmPolicy",      # ✅ 向量输入策略
        env=train_env,
        policy_kwargs=policy_kwargs,
        n_steps=256,                 # 每环境采样步数 (LSTM 序列长度)
        batch_size=256,              # 整除 n_envs * n_steps
        n_epochs=10,
        gamma=0.99,
        learning_rate=3e-4,
        gae_lambda=0.95,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        device="cuda",
        tensorboard_log=save_path,
    )

    # Model diagnostics: print feature extractor and LSTM sizes to detect mismatches
    try:
        fe = model.policy.features_extractor
        print("[MODEL DIAG] features_extractor:", fe)
        print("[MODEL DIAG] features_extractor._features_dim:", getattr(fe, '_features_dim', None))
        print("[MODEL DIAG] features_extractor.output_dim:", getattr(fe, 'output_dim', None))
    except Exception as e:
        print("[MODEL DIAG] Failed to inspect features_extractor:", e)

    try:
        lstm_actor = getattr(model.policy, 'lstm_actor', None)
        print("[MODEL DIAG] lstm_actor:", lstm_actor)
        if lstm_actor is not None:
            print("[MODEL DIAG] lstm_actor.input_size:", getattr(lstm_actor, 'input_size', None))
    except Exception as e:
        print("[MODEL DIAG] Failed to inspect lstm_actor:", e)

    try:
        print("[MODEL DIAG] train_env.observation_space:", train_env.observation_space)
    except Exception:
        pass

    # ✅ 回调：模型保存 + 自定义指标
    checkpoint_cb = CheckpointCallback(save_freq=100_000, save_path=save_path, name_prefix='rl_model')
    metrics_cb = MetaDriveMetricsCallback()
    callback_list = CallbackList([checkpoint_cb, metrics_cb])

    # ✅ 训练
    total_timesteps = 3_000_000
    model.learn(
        total_timesteps=total_timesteps,
        log_interval=4,
        callback=callback_list,
    )

    # ✅ 保存模型
    model.save(os.path.join(save_path, "recurrent_ppo_mlp_final"))
    print(f"✅ Training complete! Model saved at {save_path}")


if __name__ == "__main__":
    main()
