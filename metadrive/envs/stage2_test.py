import os
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib import RecurrentPPO
from metadrive.envs.top_down_env import TopDownMetaDrive
from Multi_BEV_CNN import ImageNetBEVCNN
from metadrive.constants import TerminationState


def create_env():
    cfg = dict(
        num_scenarios=100,
        start_seed=0,
        use_render=False,
        traffic_density=0.0,
        random_lane_width=True,
        resolution_size=128,
        traffic_mode="hybrid",
        distance=20,
        vehicle_config=dict(
            show_navi_mark=True,
            show_line_to_dest=False,
            show_line_to_navi_mark=True,
        ),
    )

    def _env_fn():
        return TopDownMetaDrive(cfg)
    return DummyVecEnv([_env_fn])


def evaluate_model(model, env, n_episodes=100):
    cumulative_rewards, success_flags = [], []

    for ep in range(n_episodes):
        obs = env.reset()
        lstm_states = None
        episode_starts = np.ones((env.num_envs,), dtype=bool)
        done, total_reward = False, 0.0

        while not done:
            action, lstm_states = model.predict(
                obs, state=lstm_states, episode_start=episode_starts, deterministic=True
            )
            obs, reward, done, infos = env.step(action)
            total_reward += float(reward[0])
            episode_starts = done
            if done[0]:
                info = infos[0]
                success = info.get("arrive_dest", False) or \
                          info.get(TerminationState.SUCCESS, False)
                success_flags.append(1 if success else 0)
                break

        cumulative_rewards.append(total_reward)
        print(f"Episode {ep+1}/{n_episodes} | Reward: {total_reward:.2f} | Success: {success_flags[-1]}")

    # 计算统计量
    mean_reward = np.mean(cumulative_rewards)
    std_reward = np.std(cumulative_rewards)
    mean_success = np.mean(success_flags)
    std_success = np.std(success_flags)

    print("\n=== Evaluation Summary ===")
    print(f"Average Cumulative Reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"Success Rate: {mean_success*100:.2f}% ± {std_success*100:.2f}%")

    return {
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "mean_success": mean_success,
        "std_success": std_success,
    }


if __name__ == "__main__":
    env = create_env()

    model_path = r"C:\Users\37945\OneDrive\Desktop\stage2论文\Case 4\recurrent_ppo_mlp_final.zip"
    policy_kwargs = dict(
        features_extractor_class=ImageNetBEVCNN,
        features_extractor_kwargs=dict(features_dim=275),
        normalize_images=False,
        lstm_hidden_size=256,
        n_lstm_layers=1,
        shared_lstm=False,
    )

    print("Loading model...")
    model = RecurrentPPO.load(model_path, env=env, policy_kwargs=policy_kwargs, device="cpu")
    print("Model loaded successfully!")

    results = evaluate_model(model, env, n_episodes=100)
