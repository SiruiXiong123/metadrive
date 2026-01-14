import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from metadrive.envs.top_down_env import TopDownMetaDrive

# ✅ 与训练对齐的 cfg
cfg = dict(
    num_scenarios=2,
    map="O",
    start_seed=500,
    log_level=50,
    use_render=True,          # 测试想看画面就 True；想纯评估就 False
    traffic_density=0.0,
    random_lane_width=True,
    random_lane_num=True,
)

def make_env():
    env = TopDownMetaDrive(cfg)
    env = Monitor(env)
    return env

if __name__ == "__main__":
    env = DummyVecEnv([make_env])

    model_path = r"C:\Users\37945\OneDrive\Desktop\BEV_Train_test_5(速度奖励调整+导航加粗)\ppo_topdown_bev_final.zip"
    model = PPO.load(model_path, env=env)

    episodes = 5
    for ep in range(episodes):
        obs = env.reset()
        done = False
        ep_return = 0.0

        while not done:
            # ✅ VecEnv 渲染：取里面那个真实 env
            env.envs[0].render()

            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, infos = env.step(action)

            ep_return += float(reward[0])
            done = bool(dones[0])

        print(f"Episode {ep+1}: return={ep_return:.2f}")

    env.close()
