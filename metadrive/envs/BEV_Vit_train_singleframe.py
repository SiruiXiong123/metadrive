import os
from functools import partial

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList

from metadrive.envs.top_down_env import TopDownMetaDrive
from metadrive.constants import DEFAULT_AGENT
try:
    from distance_and_collision_callback import MetaDriveMetricsCallback  # user callback
except Exception:
    # Fallback: provide a no-op compatible callback if the user's callback
    # module is not present. This allows importing/running quick checks.
    from stable_baselines3.common.callbacks import BaseCallback

    class MetaDriveMetricsCallback(BaseCallback):
        """No-op callback substitute used when distance_and_collision_callback
        cannot be imported. It implements the minimal interface expected by
        CallbackList/Stable Baselines so training can proceed (but without
        user metrics).
        """

        def __init__(self):
            super().__init__()

        def _on_step(self) -> bool:
            return True
try:
    # Prefer relative import when used as a package: python -m envs.BEV_Vit
    from only_VIT_singleframe import CustomViTPolicy
except Exception:
    # Fallback for direct script execution: python envs\BEV_Vit.py
    try:
        from only_VIT_singleframe import CustomViTPolicy
    except Exception:
        # As a last resort, try importing using package-qualified name
        from only_VIT_singleframe import CustomViTPolicy




def create_env(need_monitor=False):
    cfg = {
        # "map": "SS",
        "use_render": False,
        "num_scenarios": 500,
        "start_seed": 123,
        "distance": 30,
        "resolution_size": 224,
        "traffic_density": 0.0,
        "vehicle_config": {"show_navi_mark": True, "show_line_to_navi_mark": True},
    }

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
    # Save path (adjust as you like)
    path = os.path.join(os.getcwd(), "agent_model", "BEV_ViT_default_reward")
    os.makedirs(path, exist_ok=True)

    set_random_seed(0)

    # Create vectorized environments
    n_envs = 4
    train_env = SubprocVecEnv([partial(create_env, True) for _ in range(n_envs)])

    # Use the custom ViT policy class directly
    policy = CustomViTPolicy

    model = PPO(
        policy=policy,
        env=train_env,
        n_steps=2048,
        verbose=1,
        device="cuda",
        tensorboard_log=path,
    )

    checkpoint_cb = CheckpointCallback(save_freq=100_000, save_path=path, name_prefix='rl_model')

    # Combine with your metrics callback
    callback_list = CallbackList([checkpoint_cb, MetaDriveMetricsCallback()])

    # Train
    total_timesteps = 3_000_000
    model.learn(
        total_timesteps=total_timesteps,
        log_interval=4,
        callback=callback_list,
    )

    # Final save
    model.save(os.path.join(path, "ppo_topdown_vit_final"))


if __name__ == "__main__":
    main()
