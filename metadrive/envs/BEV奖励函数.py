import os
import time
import argparse
import numpy as np

from metadrive.envs.top_down_env import TopDownMetaDriveEnvV2


def collect_and_save_obs(save_dir: str, num_episodes: int = 1, steps_per_episode: int = 500, manual: bool = True):
    os.makedirs(save_dir, exist_ok=True)

    cfg = dict(
        map="OO",
        num_scenarios=1,
        use_render=False,
        start_seed=5000,
        distance=30,
        # 不要自动启动 manual control
        manual_control=False,
    )

    # 如果需要手动控制，启用 keyboard controller 和渲染窗口
    if manual:
        print("[INFO] Manual mode enabled: keyboard control active, rendering window will open")
        cfg.update({
            "manual_control": True,
            "use_render": True,
            "controller": "keyboard",
        })

    env = TopDownMetaDriveEnvV2(cfg)

    file_index = 0
    try:
        for ep in range(num_episodes):
            obs = env.reset()
            # reset() 有可能返回 env-specific observation or tuple
            # 统一处理：若为 tuple 则取第一个元素
            if isinstance(obs, (list, tuple)):
                obs = obs[0]

            # 保存初始观测（完整多通道）
            ts = int(time.time() * 1000)
            fname = os.path.join(save_dir, f"obs_ep{ep:03d}_step0000_{ts}.npz")
            try:
                # 保存为 .npz 以便保留完整数据结构
                np.savez_compressed(fname, observation=obs)
            except Exception as e:
                print(f"[WARN] Failed to save initial obs: {e}")

            done = False
            step = 0
            while not done and step < steps_per_episode:
                # 在手动模式下，env.render() 会显示窗口并处理键盘输入
                if manual:
                    env.render()
                    # 在手动模式下，让 MetaDrive 的键盘控制处理动作
                    # 不需要主动传入 action，MetaDrive 会从键盘读取
                    action = None
                else:
                    action = env.action_space.sample()
                
                step_ret = env.step(action)
                # MetaDrive 旧版本可能返回 4 或 5 元组
                # 常见格式: (obs, reward, terminated, truncated, info) 或 (obs, reward, done, info)
                if len(step_ret) == 5:
                    obs, reward, terminated, truncated, info = step_ret
                    done = bool(terminated or truncated)
                elif len(step_ret) == 4:
                    obs, reward, done, info = step_ret
                else:
                    # 兼容性：取第一个元素作为 obs，最后一个作为 info
                    obs = step_ret[0]
                    done = False

                if isinstance(obs, (list, tuple)):
                    obs = obs[0]

                ts = int(time.time() * 1000)
                fname = os.path.join(save_dir, f"obs_ep{ep:03d}_step{step+1:04d}_{ts}.npz")
                try:
                    # 保存为 .npz 以便保留完整数据结构
                    np.savez_compressed(fname, observation=obs)
                except Exception as e:
                    print(f"[WARN] Failed to save obs at step {step+1}: {e}")

                file_index += 1
                step += 1

            print(f"Episode {ep} finished, saved {step+1} frames.")
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description="Collect BEV observations from TopDown env and save as .npy files")
    parser.add_argument("--save_dir", type=str,
                        default=r"C:\Users\37945\OneDrive\Desktop\obs\obs",
                        help="Directory to save observations")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to collect")
    parser.add_argument("--steps", type=int, default=500, help="Max steps per episode")
    parser.add_argument("--no-manual", action="store_true", help="Disable manual keyboard control (auto mode)")

    args = parser.parse_args()

    # 默认启用手动模式（除非显式指定 --no-manual）
    manual_mode = not args.no_manual
    collect_and_save_obs(args.save_dir, num_episodes=args.episodes, steps_per_episode=args.steps, manual=manual_mode)


if __name__ == '__main__':
    main()
