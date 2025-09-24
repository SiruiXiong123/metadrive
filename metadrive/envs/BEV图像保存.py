# # python.py
# import os
# import numpy as np
# import matplotlib.pyplot as plt
# from metadrive.envs.top_down_env import TopDownMetaDrive

# def get_bev_hwc(obs):
#     """统一返回 HWC 格式 (H, W, C)，范围 [0,1]"""
#     x = obs["image"] if isinstance(obs, dict) and "image" in obs else obs
#     x = np.asarray(x)
#     if x.ndim != 3:
#         raise ValueError(f"Expect 3D BEV, got {x.shape}")
#     # 如果是 CHW 格式 -> 转 HWC
#     if x.shape[0] in (3,4,5) and x.shape[-1] not in (3,4,5):
#         x = np.transpose(x, (1,2,0))
#     if x.max() > 1.0:
#         x = x / 255.0
#     return np.clip(x, 0.0, 1.0)

# def save_bev_multi(bev_hwc, save_path="figures/bev_frame.png"):
#     """保存论文风格的多通道拼接图"""
#     channel_names = [
#         "Road and navigation",
#         "Ego now and previous pos",
#         "Neighbor at step t",
#         "Neighbor at step t-1",
#         "Neighbor at step t-2"
#     ]
#     C = bev_hwc.shape[-1]
#     titles = channel_names[:C] + [f"Channel {i}" for i in range(len(channel_names), C)]
#     plt.figure(figsize=(3*C, 4), dpi=150)
#     for i in range(C):
#         plt.subplot(1, C, i+1)
#         plt.imshow(bev_hwc[..., i], cmap="gray")
#         plt.title(titles[i])
#         plt.axis("off")
#     plt.suptitle("Multi-channels Top-down Observation")
#     os.makedirs(os.path.dirname(save_path), exist_ok=True)
#     plt.savefig(save_path, dpi=300, bbox_inches="tight")
#     plt.close()
#     print(f"[OK] Saved BEV image to {save_path}")

# if __name__ == "__main__":
#     env = TopDownMetaDrive(dict(
#         num_scenarios=1,
#         start_seed=123,
#         use_render=False
#     ))
#     try:
#         obs, _ = env.reset()
#         bev_hwc = get_bev_hwc(obs)
#         print("BEV shape:", bev_hwc.shape)
#         save_bev_multi(bev_hwc, "figures/bev_frame.png")
#     finally:
#         env.close()
# python_show_bev.py
import numpy as np
import matplotlib.pyplot as plt
from metadrive.envs.top_down_env import TopDownMetaDrive


def get_bev_hwc(obs):
    """统一返回 HWC 格式 (H, W, C)，范围 [0,1]"""
    x = obs["image"] if isinstance(obs, dict) and "image" in obs else obs
    x = np.asarray(x)
    if x.ndim != 3:
        raise ValueError(f"Expect 3D BEV, got {x.shape}")
    # 如果是 CHW 格式 -> 转 HWC
    if x.shape[0] in (2, 3, 4, 5) and x.shape[-1] not in (2, 3, 4, 5):
        x = np.transpose(x, (1, 2, 0))
    if x.max() > 1.0:
        x = x / 255.0
    return np.clip(x, 0.0, 1.0)


def show_bev_multi(bev_hwc):
    """显示多通道 BEV 图像（不保存）"""
    channel_names = [
    "Road network",
    "road lines",
    "past_pos",
    ]   

    C = bev_hwc.shape[-1]
    titles = channel_names[:C] + [f"Channel {i}" for i in range(len(channel_names), C)]
    plt.figure(figsize=(3 * C, 4), dpi=150)
    for i in range(C):
        plt.subplot(1, C, i + 1)
        plt.imshow(bev_hwc[..., i], cmap="gray")
        plt.title(titles[i])
        plt.axis("off")
    plt.suptitle("Multi-channels Top-down Observation")
    plt.show()


if __name__ == "__main__":
    env = TopDownMetaDrive(dict(
        num_scenarios=1,
        # use_render=True,
        start_seed=123,

    ))
    try:
        obs, _ = env.reset()
        bev_hwc = get_bev_hwc(obs)
        print("BEV shape:", bev_hwc.shape)
        show_bev_multi(bev_hwc)  # ✅ 直接显示
    finally:
        env.close()
