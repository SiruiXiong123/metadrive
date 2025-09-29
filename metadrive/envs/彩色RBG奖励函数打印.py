"""
彩色RBG奖励函数打印.py

功能：
1) 允许用户通过键盘手动控制智能体（使用 MetaDrive 的键盘控制）。
2) 每一帧保存彩色 RGB BEV 的原始数据（NumPy 数组，不保存为图片）。
3) 在终端打印每一步的 reward（奖励）。

用法：
    python envs/彩色RBG奖励函数打印.py

运行时按常用键盘控制驾驶（请确保窗口有焦点）。按 Ctrl+C 中断并保存采集到的数据。

输出：
    recordings/彩色RBG_reward_<TIMESTAMP>.npz
    包含 keys: "frames" (T,H,W,3 uint8), "rewards" (T,), "dones" (T,) , "infos" (list)

"""

import os
import time
import numpy as np
from typing import List

from metadrive.envs.top_down_env import TopDownMetaDrive
from metadrive.constants import DEFAULT_AGENT


def get_bev_hwc(obs):
    """Return HWC uint8 array with values in [0,255].
    Accepts either the raw observation array or a dict with key "image".
    """
    x = obs["image"] if isinstance(obs, dict) and "image" in obs else obs
    x = np.asarray(x)
    if x.ndim != 3:
        raise ValueError(f"Expect 3D BEV, got {x.shape}")
    # If channels-first (C,H,W) -> transpose to HWC
    if x.shape[0] in (2, 3, 4, 5) and x.shape[-1] not in (2, 3, 4, 5):
        x = np.transpose(x, (1, 2, 0))
    # If float [0,1] -> convert to uint8
    if x.dtype == np.float32 or x.max() <= 1.0:
        x_u8 = (np.clip(x, 0.0, 1.0) * 255).astype(np.uint8)
    else:
        x_u8 = x.astype(np.uint8)
    return x_u8


def make_output_path(base_dir='recordings'):
    os.makedirs(base_dir, exist_ok=True)
    ts = time.strftime('%Y%m%d_%H%M%S')
    filename = f"彩色RBG_reward_{ts}.npz"
    return os.path.join(base_dir, filename)


def main():
    cfg = {
        'map': 'OO',
        'random_lane_width': True,
        'random_lane_num': False,
        'use_render': True,  # render so keyboard works
        'traffic_density': 0.0,
        'traffic_mode': 'hybrid',
        'manual_control': True,
        'controller': 'keyboard',
        'vehicle_config': {
            'show_navi_mark': True,
            'show_line_to_dest': False,
            'show_line_to_navi_mark': True,
        },
        # BEV params
        'distance': 25,
        'resolution_size': 180,
    }

    env = TopDownMetaDrive(cfg)
    out_file = make_output_path()
    frames: List[np.ndarray] = []
    rewards: List[float] = []
    dones: List[bool] = []
    infos: List[object] = []

    try:
        obs, info = env.reset()
        # ensure color output (TopDownMultiChannel defaults to color by our branch,
        # but force the property if present for safety)
        obs_obj = env.observations.get(DEFAULT_AGENT)
        if obs_obj is not None and hasattr(obs_obj, 'debug_color'):
            obs_obj.debug_color = True

        print('Env reset. Use keyboard to control. Close window or Ctrl+C to stop and save.')

        step = 0
        while True:
            action = None
            try:
                # Some configs accept None for manual keyboard control; fallback to sample if not
                result = env.step(action)
            except Exception:
                try:
                    action = env.action_space.sample()
                    result = env.step(action)
                except Exception as e:
                    print('env.step failed:', e)
                    break

            # Support Gym / Gymnasium signatures
            if len(result) == 5:
                obs, reward, terminated, truncated, info = result
                done = bool(terminated or truncated)
            elif len(result) == 4:
                obs, reward, done, info = result
            else:
                obs = result[0]
                reward = float('nan')
                done = False
                info = {}

            # Convert observation to HWC uint8 RGB and append
            try:
                bev = get_bev_hwc(obs)
                # Keep only RGB channels if extra channels present (e.g., stacked channels),
                # prefer last dimension to be channels
                if bev.shape[-1] > 3:
                    bev = bev[..., :3]
                frames.append(bev.copy())
            except Exception as e:
                print('Failed to convert observation to RGB BEV:', e)

            rewards.append(float(reward) if reward is not None else 0.0)
            dones.append(bool(done))
            infos.append(info if info is not None else {})

            print(f"step={step}, reward={reward}, done={done}")

            step += 1
            if done:
                print('Episode finished, resetting...')
                try:
                    env.reset()
                except Exception:
                    break

    except KeyboardInterrupt:
        print('Interrupted by user (Ctrl+C). Saving collected frames...')
    finally:
        env.close()
        if len(frames) == 0:
            print('No frames collected; nothing to save.')
            return
        arr = np.stack(frames, axis=0)  # T,H,W,3 uint8
        np.savez_compressed(out_file, frames=arr, rewards=np.array(rewards), dones=np.array(dones), infos=infos)
        print(f'Saved {arr.shape[0]} frames to: {out_file}')


if __name__ == '__main__':
    main()
