import os
import random
import numpy as np
from matplotlib import pyplot as plt
from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.component.sensors.rgb_camera import RGBCamera
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv

# 固定随机种子
random.seed(123)

# 摄像头和图像设置
sensor_size = (84, 60)
stack_size = 3
base_path = r'C:\Users\37945\OneDrive\Desktop'

# 配置环境参数
cfg = {
    "map": "OO",
    # "num_scenarios": 500,
    # "start_seed": 123,
    "random_lane_width": True,
    "random_lane_num": False,
    "use_render": True,
    "traffic_density": 0.0,
    "traffic_mode": "hybrid",
    "manual_control": True,
    "controller": "keyboard",
    "vehicle_config": {
        "show_navi_mark": True,
        "show_line_to_dest": False,
        "show_line_to_navi_mark": True,
    },
}

# 创建 DummyVecEnv 包裹的单环境
def create_env_for_testing():
    def _env_fn():
        return MetaDriveEnv(cfg)
    return DummyVecEnv([_env_fn])

if __name__ == '__main__':
    env = create_env_for_testing()


    # 获取一帧观测
    reset_ret = env.reset()
    # support both (obs, info) and obs returns from env.reset()
    if isinstance(reset_ret, tuple) and len(reset_ret) >= 1:
        obs = reset_ret[0]
    else:
        obs = reset_ret

    # 尝试从 obs 中找到 image 数组（兼容 dict、vec env）
    img_arr = None
    if isinstance(obs, dict):
        if "image" in obs:
            img_arr = obs["image"]
        else:
            vals = [v for v in obs.values() if isinstance(v, np.ndarray)]
            img_arr = vals[0] if vals else None
    elif isinstance(obs, np.ndarray):
        img_arr = obs
    else:
        raise RuntimeError("Unsupported observation type: %r" % type(obs))

    if img_arr is None:
        raise RuntimeError("No image found in observation")

    # 如果是 vectorized env，通常第一维是环境 batch 大小（1）
    if isinstance(img_arr, np.ndarray) and img_arr.ndim == 4 and img_arr.shape[0] == 1:
        img_arr = img_arr[0]

    # 如果还有 4 维（可能是 time x H x W x C 或 batch x H x W x C），尝试取最后一帧
    if isinstance(img_arr, np.ndarray) and img_arr.ndim == 4:
        # 假设最后一维是 channel，如果是 (T,H,W,C) 则取最后一帧
        if img_arr.shape[-1] in (1, 3, 4):
            img_arr = img_arr[-1]
        else:
            # 其他罕见布局：尝试压缩第一个轴
            img_arr = img_arr[0]

    # 此时 img_arr 应为 2D (H,W) 或 3D (H,W,C) 或 (C,H,W)
    if img_arr.ndim == 2:
        img = img_arr
    elif img_arr.ndim == 3:
        # 如果是 channel-first (C,H,W)，把它转成 (H,W,C)
        if img_arr.shape[0] in (1, 3, 4) and img_arr.shape[-1] not in (1, 3, 4):
            img = np.transpose(img_arr, (1, 2, 0))
        else:
            img = img_arr
    else:
        raise RuntimeError(f"Unexpected image array shape: {img_arr.shape}")

    # 显示图像
    plt.imshow(img)
    plt.title("RGB Camera Last Frame")
    plt.axis('off')
    plt.show()

    env.close()
