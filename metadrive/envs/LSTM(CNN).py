import os
import time
import argparse
import numpy as np
import gymnasium as gym

try:
    from sb3_contrib import RecurrentPPO
except Exception:
    raise ImportError("sb3_contrib is required. Install with `pip install sb3-contrib`.")

from stable_baselines3.common.vec_env import DummyVecEnv
from metadrive.envs.top_down_env import TopDownMetaDrive

# The training used this feature extractor; keep the same import so loading finds the class
try:
    from Multi_BEV_CNN import ImageNetBEVCNN
except Exception:
    # If import fails, still allow load attempt — RecurrentPPO.load will need the class in scope
    ImageNetBEVCNN = None


cfg = {
    "num_scenarios": 500,
    "start_seed": 123,
    "random_lane_width": True,
    "random_lane_num": False,
    # For viewing we enable render by default; you can override with CLI args
    "use_render": True,
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


def make_env():
    class ImageOnlyWrapper(gym.ObservationWrapper):
        """Convert an env that returns a Dict observation {'image':..., 'state':...} into one
        that returns only the image (Box) so it matches models trained on image-only obs.
        """
        def __init__(self, env):
            super().__init__(env)
            obs_space = getattr(env, 'observation_space', None)
            # gymnasium.spaces.Dict uses attribute .spaces
            try:
                image_space = obs_space.spaces['image']
            except Exception:
                # fallback: if already Box, keep it
                image_space = obs_space
            self.observation_space = image_space

        def observation(self, observation):
            if isinstance(observation, dict):
                return observation.get('image', observation)
            return observation

    def _init():
        env = TopDownMetaDrive(cfg)

        # Compatibility wrapper: some envs (older gym-style) don't accept `options` kw in reset
        class ResetCompatWrapper(gym.Wrapper):
            def reset(self, seed=None, options=None, **kwargs):
                # Try to call reset with the new signature, fallback to older signatures, and
                # normalize return to (obs, info)
                try:
                    res = self.env.reset(seed=seed, options=options, **kwargs)
                except TypeError:
                    try:
                        res = self.env.reset(seed=seed, **kwargs)
                    except TypeError:
                        res = self.env.reset(**kwargs)

                # Normalize to (obs, info)
                if isinstance(res, tuple) and len(res) == 2:
                    return res
                return res, {}

        # Wrap env so DummyVecEnv (which calls reset(..., options=...)) doesn't fail
        env = ResetCompatWrapper(env)

        # If env returns a Dict observation, wrap it to return only the image channel
        try:
            if hasattr(env, 'observation_space') and hasattr(env.observation_space, 'spaces') and 'image' in env.observation_space.spaces:
                env = ImageOnlyWrapper(env)
        except Exception:
            pass

        return env

    return DummyVecEnv([_init])


def run(model_path: str, episodes: int = 5, device: str = "cpu", render_mode: str = "topdown", sleep: float = 0.03):
    env = make_env()

    # Load the recurrent PPO model. sb3_contrib.RecurrentPPO expects feature extractor class to be importable.
    print(f"Loading model from: {model_path} (device={device})")
    model = RecurrentPPO.load(model_path, env=env, device=device)

    # Single environment (DummyVecEnv with one env)
    n_envs = env.num_envs if hasattr(env, 'num_envs') else 1

    for ep in range(1, episodes + 1):
        obs = env.reset()

        # For recurrent policies provide initial states and episode_starts mask
        states = None
        episode_starts = np.ones((n_envs,), dtype=bool)

        done = np.zeros((n_envs,), dtype=bool)
        ep_rewards = np.zeros((n_envs,), dtype=float)

        print(f"Starting episode {ep}")
        while True:
            # Render (Top-down renderer used by TopDownMetaDrive)
            try:
                env.render(mode=render_mode)
            except Exception:
                # fallback to default render call
                try:
                    env.render()
                except Exception:
                    pass

            # Predict action with recurrent state and episode_starts mask
            action, states = model.predict(obs, state=states, episode_start=episode_starts, deterministic=True)

            obs, reward, terminated, info = env.step(action)

            # VecEnv returns arrays; handle both vectorized and single-env shapes
            reward_arr = np.array(reward).reshape(-1)
            ep_rewards += reward_arr

            # terminated may be array or bool
            if isinstance(terminated, (list, tuple, np.ndarray)):
                done = np.array(terminated).reshape(-1)
            else:
                done = np.array([bool(terminated)])

            # episode_starts should be True where episodes just began; set for next step
            episode_starts = done.copy()

            if np.any(done):
                for i, d in enumerate(done):
                    if d:
                        print(f"Env {i} finished - episode reward: {ep_rewards[i]:.3f}")
                        ep_rewards[i] = 0.0
                break

            # Small sleep so rendering is watchable
            time.sleep(sleep)

    env.close()
    print("Finished running.")


def parse_args():
    parser = argparse.ArgumentParser(description="Play a trained RecurrentPPO LSTM(CNN) agent in TopDownMetaDrive")
    parser.add_argument("--model-path", type=str, default=os.path.join(os.getcwd(), "agent_model", "BEV_MlpLstmPolicy", "recurrent_ppo_mlp_final.zip"), help="Path to the trained model (.zip)")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to run")
    parser.add_argument("--device", type=str, default="cpu", help="Device to load the model on (cpu or cuda)")
    parser.add_argument("--render-mode", type=str, default="topdown", help="Render mode passed to env.render(), e.g., 'topdown'")
    parser.add_argument("--sleep", type=float, default=0.03, help="Seconds to sleep between steps to control playback speed")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run(model_path=args.model_path, episodes=args.episodes, device=args.device, render_mode=args.render_mode, sleep=args.sleep)
