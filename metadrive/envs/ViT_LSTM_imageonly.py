import os
from functools import partial

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList

from metadrive.envs.top_down_env import TopDownMetaDrive
from metadrive.constants import DEFAULT_AGENT

try:
    # Try relative import when running as a package/module
    from only_VIT_singleframe import CustomCombinedExtractor
except Exception:
    # Fallback for direct script execution
    try:
        from only_VIT_singleframe import CustomCombinedExtractor
    except Exception:
        # Last resort: import by module name in case of path differences
        from only_VIT_singleframe import CustomCombinedExtractor


# Environment configuration: use 224 resolution which matches ViT expectation
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
    # enable debug_color on per-agent observation object if available so the
    # observation is RGB and easier to visualize. The extractor accepts HWC.
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
    # Save path
    path = os.path.join(os.getcwd(), "agent_model", "BEV_ViT_LSTM_imageonly")
    os.makedirs(path, exist_ok=True)

    set_random_seed(0)

    # Create vectorized environments
    n_envs = 4
    train_env = SubprocVecEnv([partial(create_env, True) for _ in range(n_envs)])

    # Policy kwargs for RecurrentPPO using MlpLstmPolicy
    # We use CustomCombinedExtractor which sets _features_dim=256 internally,
    # so do NOT pass a features_dim kw here (the extractor signature doesn't accept it).
    policy_kwargs = dict(
        features_extractor_class=CustomCombinedExtractor,
        features_extractor_kwargs=dict(frames=1, temporal_pool='mean'),
        normalize_images=False,
        lstm_hidden_size=256,
        n_lstm_layers=1,
        shared_lstm=False,
    )

    # Build RecurrentPPO model
    model = RecurrentPPO(
        policy="MlpLstmPolicy",
        env=train_env,
        policy_kwargs=policy_kwargs,
        n_steps=256,                 # sequence length per env
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        learning_rate=3e-4,
        gae_lambda=0.95,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        device="cuda",
        tensorboard_log=path,
    )

    # Diagnostics: print feature extractor and LSTM sizes to detect mismatches
    try:
        fe = model.policy.features_extractor
        print("[MODEL DIAG] features_extractor:", fe)
        print("[MODEL DIAG] features_extractor._features_dim:", getattr(fe, '_features_dim', None))
    except Exception as e:
        print("[MODEL DIAG] Failed to inspect features_extractor:", e)

    try:
        lstm_actor = getattr(model.policy, 'lstm_actor', None)
        print("[MODEL DIAG] lstm_actor:", lstm_actor)
        if lstm_actor is not None:
            # Different SB3 contrib versions may expose different attribute names
            print("[MODEL DIAG] lstm_actor.input_size:", getattr(lstm_actor, 'input_size', None))
    except Exception as e:
        print("[MODEL DIAG] Failed to inspect lstm_actor:", e)

    try:
        print("[MODEL DIAG] train_env.observation_space:", train_env.observation_space)
    except Exception:
        pass

    # Callbacks: checkpoint + user metrics if available
    checkpoint_cb = CheckpointCallback(save_freq=100_000, save_path=path, name_prefix='rl_model')
    # Try to import user callback if available
    try:
        from distance_and_collision_callback import MetaDriveMetricsCallback
        metrics_cb = MetaDriveMetricsCallback()
        callback_list = CallbackList([checkpoint_cb, metrics_cb])
    except Exception:
        callback_list = CallbackList([checkpoint_cb])

    # Train
    total_timesteps = 3_000_000
    model.learn(
        total_timesteps=total_timesteps,
        log_interval=4,
        callback=callback_list,
    )

    # Final save
    model.save(os.path.join(path, "recurrent_ppo_vit_imageonly_final"))


if __name__ == "__main__":
    main()
