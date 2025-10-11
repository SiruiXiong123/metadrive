import os
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

# === 根目录 ===
base_dir = r"C:\Users\37945\OneDrive\Desktop\Stage 1"

# === 实验配置 ===
log_paths = {
    "CNN (Image only)": os.path.join(base_dir, "CNN (Image only)"),
    "CNN + State Fusion": os.path.join(base_dir, "CNN + State Fusion"),
    "ViT (Image only)": os.path.join(base_dir, "ViT (Image only)"),
    "ViT + State Fusion": os.path.join(base_dir, "ViT + State Fusion"),
    "CNN_SAC": os.path.join(base_dir, "CNN_SAC"),
    "CNN_TD3": os.path.join(base_dir, "CNN_TD3"),
}

# === 颜色方案 ===
colors = {
    "CNN (Image only)": "#FF4500",   # 橙红
    "CNN + State Fusion": "#FF8C00", # 深橙
    "ViT (Image only)": "#1E90FF",   # 蓝
    "ViT + State Fusion": "#32CD32", # 绿
    "CNN_SAC": "#000000",            # 黑
    "CNN_TD3": "#808080",            # 灰
}

# === 对比指标 ===
tags = {
    "rollout/ep_len_mean": "Average Episode Length",
    "rollout/ep_rew_mean": "Average Episode Reward",
    "rollout/success_rate": "Success Rate",
    "time/fps": "FPS"
}

# === 平滑函数 ===
def moving_average(data, window_size=10):
    """计算滑动平均"""
    if len(data) < window_size:
        return data
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# === 绘图主循环 ===
window_size = 15  # 平滑窗口大小（可根据曲线噪声调整）

for tag, title in tags.items():
    plt.figure(figsize=(8, 5))

    for name, path in log_paths.items():
        event_files = [f for f in os.listdir(path) if f.startswith("events.out.tfevents")]
        if not event_files:
            print(f"⚠️ No event files found in {path}")
            continue

        event_path = os.path.join(path, event_files[0])
        ea = event_accumulator.EventAccumulator(event_path)
        ea.Reload()

        if tag not in ea.Tags().get('scalars', []):
            print(f"⚠️ Tag '{tag}' not found in {name}")
            continue

        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]

        # 平滑处理
        smoothed_values = moving_average(values, window_size=window_size)
        smoothed_steps = steps[:len(smoothed_values)]

        plt.plot(smoothed_steps, smoothed_values, label=name, color=colors[name], linewidth=2)

    plt.title(f"{title} Comparison", fontsize=13)
    plt.xlabel("Training Steps", fontsize=11)
    plt.ylabel("Value", fontsize=11)
    plt.legend(loc="best", fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
