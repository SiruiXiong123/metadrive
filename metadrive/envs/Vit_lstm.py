import os
from functools import partial

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList

from metadrive.envs.top_down_env import TopDownMetaDrive
from metadrive.constants import DEFAULT_AGENT

# user callback optional
try:
    from distance_and_collision_callback import MetaDriveMetricsCallback
except Exception:
    from stable_baselines3.common.callbacks import BaseCallback

    class MetaDriveMetricsCallback(BaseCallback):
        def __init__(self):
            super().__init__()

        def _on_step(self) -> bool:
            return True

# Try to import the custom ViT extractor from the local envs module
try:
    # when running from envs/ directory
    from only_VIT_singleframe import CustomCombinedExtractor
except Exception:
    try:
        # when running from repository root
        from envs.only_VIT_singleframe import CustomCombinedExtractor
    except Exception:
        # fallback: import by full module path
        from metadrive.envs.only_VIT_singleframe import CustomCombinedExtractor


# Environment config: resolution must match ViT expected input (e.g., 224)
cfg = {
    "use_render": False,
    "num_scenarios": 500,
    "start_seed": 123,
    "distance": 30,
    "resolution_size": 224,
    "traffic_density": 0.0,
    "vehicle_config": {"show_navi_mark": True, "show_line_to_navi_mark": True},
}


def create_env(need_monitor=False):
    env = TopDownMetaDrive(cfg)
    # Enable color RGB observation (debug_color True) so extractor receives HWC RGB
    try:
        obs_obj = env.observations[DEFAULT_AGENT]
        if hasattr(obs_obj, 'debug_color'):
            obs_obj.debug_color = True
    except Exception:
        pass

    if need_monitor:
        env = Monitor(env)
    return env


def main():
    save_path = os.path.join(os.getcwd(), "agent_model", "BEV_ViT_LSTM")
    os.makedirs(save_path, exist_ok=True)

    set_random_seed(0)

    # number of parallel envs
    n_envs = 4
    train_env = SubprocVecEnv([partial(create_env, True) for _ in range(n_envs)])

    # Policy kwargs for MlpLstmPolicy: use our ViT extractor
    policy_kwargs = dict(
        features_extractor_class=CustomCombinedExtractor,
        features_extractor_kwargs=dict(frames=1, temporal_pool='mean'),
        normalize_images=False,  # env already normalizes pixels to [0,1]
        lstm_hidden_size=256,
        n_lstm_layers=1,
        shared_lstm=False,
    )

    # Build RecurrentPPO model
    model = RecurrentPPO(
        policy="MlpLstmPolicy",
        env=train_env,
        policy_kwargs=policy_kwargs,
        n_steps=256,            # sequence length per env for LSTM
        batch_size=256,         # must divide n_envs * n_steps
        n_epochs=8,
        learning_rate=3e-4,
        gae_lambda=0.95,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        device="cuda",
        tensorboard_log=save_path,
    )

    # Diagnostics: print extractor feature dim so we can confirm LSTM input size
    try:
        fe = model.policy.features_extractor
        print("[MODEL DIAG] features_extractor:", fe)
        print("[MODEL DIAG] features_extractor._features_dim:", getattr(fe, '_features_dim', None))
    except Exception as e:
        print("[MODEL DIAG] Failed to inspect features_extractor:", e)

    # Callbacks
    checkpoint_cb = CheckpointCallback(save_freq=100_000, save_path=save_path, name_prefix='rl_model')
    metrics_cb = MetaDriveMetricsCallback()
    callback_list = CallbackList([checkpoint_cb, metrics_cb])

    # Train
    total_timesteps = 1_000_000  # adjust as needed
    model.learn(
        total_timesteps=total_timesteps,
        log_interval=4,
        callback=callback_list,
    )

    # Save
    model.save(os.path.join(save_path, "recurrent_ppo_vit_lstm_final"))


if __name__ == '__main__':
    main()
