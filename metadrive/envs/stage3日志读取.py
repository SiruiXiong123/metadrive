import os
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

# === 根目录 ===
base_dir = r"C:\Users\37945\OneDrive\Desktop\stage3"

# === 递归查找所有事件文件 ===
event_files = []
for root, _, files in os.walk(base_dir):
    for f in files:
        if f.startswith("events.out.tfevents"):
            event_files.append(os.path.join(root, f))

if not event_files:
    print("❌ 未找到任何 TensorBoard 日志文件！")
else:
    print(f"✅ 找到 {len(event_files)} 个日志文件：\n" + "\n".join(event_files))

# === 对比指标 ===
tags = {
    "rollout/ep_len_mean": "Average Episode Length",
    "rollout/ep_rew_mean": "Average Episode Reward",
    "rollout/success_rate": "Success Rate",
    "time/fps": "FPS"
}

# === 滑动平均 ===
def moving_average(values, window_size=10):
    if len(values) < window_size:
        return values
    return np.convolve(values, np.ones(window_size)/window_size, mode="valid")

# === 自动分配颜色 ===
color_list = ["#FF4500", "#1E90FF", "#32CD32", "#800080"]
colors = {f: color_list[i % len(color_list)] for i, f in enumerate(event_files)}

# === 绘图 ===
for tag, title in tags.items():
    plt.figure(figsize=(8, 5))
    for f in event_files:
        ea = event_accumulator.EventAccumulator(f)
        ea.Reload()

        if tag not in ea.Tags().get("scalars", []):
            print(f"⚠️ {f} 中未找到 tag '{tag}'")
            continue

        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]
        smooth_values = moving_average(values, window_size=10)
        smooth_steps = steps[-len(smooth_values):]

        label_name = os.path.basename(os.path.dirname(f))  # 取父文件夹名，如 APF / TTC
        plt.plot(smooth_steps, smooth_values, label=label_name, color=colors[f], linewidth=2)

    plt.title(f"{title} Comparison", fontsize=13)
    plt.xlabel("Training Steps", fontsize=11)
    plt.ylabel("Value", fontsize=11)
    plt.legend(loc="lower right", fontsize=9, framealpha=0.8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
