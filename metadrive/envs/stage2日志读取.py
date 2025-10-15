import os
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

# === 根目录 ===
base_dir = r"C:\Users\37945\OneDrive\Desktop\stage2论文"

# 自动列出该目录下的所有子文件夹（每个子文件夹代表一个实验）
log_paths = {
    name: os.path.join(base_dir, name)
    for name in os.listdir(base_dir)
    if os.path.isdir(os.path.join(base_dir, name))
}

# === 自动分配颜色 ===
color_list = ["#FF4500", "#FF8C00", "#1E90FF", "#32CD32", "#800080", "#000000", "#808080"]
colors = {name: color_list[i % len(color_list)] for i, name in enumerate(log_paths)}

# === 对比指标（TensorBoard scalar tags）===
tags = {
    "rollout/ep_len_mean": "Average Episode Length",
    "rollout/ep_rew_mean": "Average Episode Reward",
    "rollout/success_rate": "Success Rate",
    "time/fps": "FPS"
}

# === 定义滑动平均函数 ===
def moving_average(values, window_size=10):
    """Compute moving average with given window size."""
    if len(values) < window_size:
        return values
    return np.convolve(values, np.ones(window_size) / window_size, mode='valid')

# === 遍历每个指标绘图 ===
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

        if tag not in ea.Tags().get("scalars", []):
            print(f"⚠️ Tag '{tag}' not found in {name}")
            continue

        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]

        # 应用滑动平均
        smooth_values = moving_average(values, window_size=10)
        smooth_steps = steps[len(steps) - len(smooth_values):]

        plt.plot(smooth_steps, smooth_values, label=name, color=colors[name], linewidth=2)

    # === 图像样式设置 ===
    plt.title(f"{title} Comparison", fontsize=13)   # ✅ 去掉 (Moving Avg = 10)
    plt.xlabel("Training Steps", fontsize=11)
    plt.ylabel("Value", fontsize=11)

    # ✅ legend 固定在右下角，带半透明背景
    plt.legend(loc="lower right", fontsize=9, framealpha=0.8)

    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
