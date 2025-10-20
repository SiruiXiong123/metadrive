import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorboard.backend.event_processing import event_accumulator

# === 根目录 ===
base_dir = r"C:\Users\37945\OneDrive\Desktop\stage2论文"

# === 查找所有 TensorBoard 日志文件 ===
event_files = []
for root, _, files in os.walk(base_dir):
    for f in files:
        if f.startswith("events.out.tfevents"):
            event_files.append(os.path.join(root, f))

if not event_files:
    print("❌ 未找到任何 TensorBoard 日志文件！")
    exit()
else:
    print(f"✅ 找到 {len(event_files)} 个日志文件：\n" + "\n".join(event_files))

# === 需要提取的指标 ===
tags = {
    "rollout/ep_len_mean": "Average Episode Length",
    "rollout/ep_rew_mean": "Average Episode Reward",
    "rollout/success_rate": "Success Rate",
    "time/fps": "FPS",
}

# === 滑动平均函数 ===
def moving_average(values, window_size=10):
    if len(values) < window_size:
        return np.array(values)
    return np.convolve(values, np.ones(window_size) / window_size, mode="valid")

# === 绘图风格 ===
sns.set(style="whitegrid", font_scale=1.4)
save_dir = os.path.join(base_dir, "Figures")
os.makedirs(save_dir, exist_ok=True)

# === 固定 Case 对应颜色 ===
color_map = {
    "Case 1": "#E41A1C",   # 红
    "Case 2": "#377EB8",   # 蓝
    "Case 3": "#4DAF4A",   # 绿
    "Case 4": "#FF7F00",   # 橙
}

# === 主循环 ===
for tag, title in tags.items():
    plt.figure(figsize=(8, 5))
    plt.title(f"{title} Comparison", fontsize=16, weight="bold")
    plt.xlabel("Training Steps", fontsize=13)
    plt.ylabel("Value", fontsize=13)

    # === 聚合每个实验组 ===
    group_data = {}
    for f in event_files:
        folder_name = os.path.basename(os.path.dirname(f))
        group_data.setdefault(folder_name, []).append(f)

    for folder_name, file_list in sorted(group_data.items()):
        color = color_map.get(folder_name, "#984EA3")  # 默认紫

        all_steps, all_values = [], []
        for f in file_list:
            ea = event_accumulator.EventAccumulator(f)
            try:
                ea.Reload()
            except Exception as e:
                print(f"⚠️ 无法读取 {f}: {e}")
                continue

            if tag not in ea.Tags().get("scalars", []):
                continue

            events = ea.Scalars(tag)
            steps = np.array([e.step for e in events])
            values = np.array([e.value for e in events])
            smooth_values = moving_average(values, window_size=10)
            all_steps.append(steps[-len(smooth_values):])
            all_values.append(smooth_values)

        if not all_values:
            print(f"⚠️ {folder_name} 无该指标 {tag}")
            continue

        # === 多日志取均值±标准差 ===
        min_len = min(len(v) for v in all_values)
        all_values = np.array([v[-min_len:] for v in all_values])
        all_steps = np.array([s[-min_len:] for s in all_steps])
        mean_steps = np.mean(all_steps, axis=0)
        mean_values = np.mean(all_values, axis=0)
        std_values = np.std(all_values, axis=0)

        plt.plot(mean_steps, mean_values, label=folder_name, color=color, linewidth=2.5)
        plt.fill_between(mean_steps, mean_values - std_values, mean_values + std_values,
                         color=color, alpha=0.25)

    # === 动态调整图例位置 ===
    legend_loc = "lower right" if "Reward" in title or "Success" in title else "upper right"
    plt.legend(loc=legend_loc, fontsize=11, framealpha=0.95, facecolor="white", edgecolor="gray")

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"{title.replace(' ', '_')}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ 已保存图像：{save_path}")
