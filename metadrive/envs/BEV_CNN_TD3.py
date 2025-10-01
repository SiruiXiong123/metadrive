import os
from functools import partial

from stable_baselines3 import TD3
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from stable_baselines3.common.noise import NormalActionNoise

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


def main():
    # 保存路径
    save_path = os.path.join(os.getcwd(), "agent_model", "BEV_TD3_CNN")
    os.makedirs(save_path, exist_ok=True)

    set_random_seed(0)

    # 并行环境
    n_envs = 4
    train_env = SubprocVecEnv([partial(create_env, True) for _ in range(n_envs)])

    # 尝试诊断 env 配置
    try:
        first_cfg = train_env.get_attr('config')[0]
        vc = first_cfg.get('vehicle_config', {})
        print('[DIAG] vehicle_config (first env):', vc)
    except Exception as e:
        print('[DIAG] Failed to read env config from SubprocVecEnv:', e)

    # policy kwargs - 使用自定义特征提取器
    policy_kwargs = dict(
        features_extractor_class=ImageNetBEVCNN,
        features_extractor_kwargs=dict(features_dim=275),  # 256 image + 19 state
        normalize_images=False,
    )

    # 构建作用噪声（TD3 需要）
    try:
        action_dim = train_env.action_space.shape[0]
    except Exception:
        # fallback: instantiate a temp env to query
        tmp = create_env(False)
        action_dim = tmp.action_space.shape[0]
        tmp.close()

    action_noise = NormalActionNoise(mean=[0.0] * action_dim, sigma=[0.1] * action_dim)

    # 构建 TD3 模型
    model = TD3(
        policy="MultiInputPolicy",
        env=train_env,
        policy_kwargs=policy_kwargs,
        learning_rate=1e-4,
        buffer_size=500000,
        batch_size=100,
        learning_starts=1000,
        action_noise=action_noise,
        verbose=1,
        device="cuda",
        tensorboard_log=save_path,
    )

    # Model diagnostics
    try:
        fe = model.policy.features_extractor
        print('[MODEL DIAG] features_extractor:', fe)
        print('[MODEL DIAG] features_extractor._features_dim:', getattr(fe, '_features_dim', None))
        print('[MODEL DIAG] features_extractor.output_dim:', getattr(fe, 'output_dim', None))
    except Exception as e:
        print('[MODEL DIAG] Failed to inspect features_extractor:', e)

    try:
        print('[MODEL DIAG] train_env.observation_space:', train_env.observation_space)
        print('[MODEL DIAG] train_env.action_space:', train_env.action_space)
    except Exception:
        pass

    # callbacks
    checkpoint_cb = CheckpointCallback(save_freq=100_000, save_path=save_path, name_prefix='td3_model')
    metrics_cb = MetaDriveMetricsCallback()
    callback_list = CallbackList([checkpoint_cb, metrics_cb])

    # 训练
    total_timesteps = 3_000_000
    model.learn(total_timesteps=total_timesteps, callback=callback_list, log_interval=4)

    # 保存模型
    model.save(os.path.join(save_path, 'td3_cnn_final'))
    print(f"✅ TD3 training complete! Model saved at {save_path}")


if __name__ == '__main__':
    main()
