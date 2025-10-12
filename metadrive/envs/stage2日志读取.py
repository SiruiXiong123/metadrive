# import os
# import matplotlib.pyplot as plt
# from tensorboard.backend.event_processing import event_accumulator

# # 根目录
# base_dir = r"C:\Users\37945\OneDrive\Desktop\Stage 1"

# # 六种实验配置
# log_paths = {
#     "CNN (Image only)": os.path.join(base_dir, "CNN (Image only)"),
#     "CNN + State Fusion": os.path.join(base_dir, "CNN + State Fusion"),
#     "ViT (Image only)": os.path.join(base_dir, "ViT (Image only)"),
#     "ViT + State Fusion": os.path.join(base_dir, "ViT + State Fusion"),
#     "CNN_SAC": os.path.join(base_dir, "CNN_SAC"),
#     "CNN_TD3": os.path.join(base_dir, "CNN_TD3"),
# }

# # 自定义颜色
# colors = {
#     "CNN (Image only)": "#FF4500",       # 橙红
#     "CNN + State Fusion": "#FF8C00",     # 深橙
#     "ViT (Image only)": "#1E90FF",       # 蓝
#     "ViT + State Fusion": "#32CD32",     # 绿
#     "CNN_SAC": "#000000",                # 黑
#     "CNN_TD3": "#808080",                # 灰
# }

# # 对比指标（TensorBoard scalar tags）
# tags = {
#     "rollout/ep_len_mean": "Average Episode Length",
#     "rollout/ep_rew_mean": "Average Episode Reward",
#     "rollout/success_rate": "Success Rate",
#     "time/fps": "FPS"
# }

# # 遍历每个指标单独绘图
# for tag, title in tags.items():
#     plt.figure(figsize=(8, 5))

#     for name, path in log_paths.items():
#         event_files = [f for f in os.listdir(path) if f.startswith("events.out.tfevents")]
#         if not event_files:
#             print(f"⚠️ No event files found in {path}")
#             continue

#         event_path = os.path.join(path, event_files[0])
#         ea = event_accumulator.EventAccumulator(event_path)
#         ea.Reload()

#         if tag not in ea.Tags().get('scalars', []):
#             print(f"⚠️ Tag '{tag}' not found in {name}")
#             continue

#         events = ea.Scalars(tag)
#         steps = [e.step for e in events]
#         values = [e.value for e in events]

#         plt.plot(steps, values, label=name, color=colors[name], linewidth=2)

#     plt.title(f"{title} Comparison", fontsize=13)
#     plt.xlabel("Training Steps", fontsize=11)
#     plt.ylabel("Value", fontsize=11)
#     plt.legend(loc="best", fontsize=9)
#     plt.grid(alpha=0.3)
#     plt.tight_layout() 
#     plt.show()
import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

# 根目录
base_dir = r"C:\Users\37945\OneDrive\Desktop\2.stage2奖励函数修改"

# 自动列出该目录下的所有子文件夹（假设每个子文件夹中都有一个 TensorBoard 日志）
log_paths = {
    name: os.path.join(base_dir, name)
    for name in os.listdir(base_dir)
    if os.path.isdir(os.path.join(base_dir, name))
}

# 自动分配颜色（可根据需要手动改）
color_list = ["#FF4500", "#FF8C00", "#1E90FF", "#32CD32", "#800080", "#000000", "#808080"]
colors = {name: color_list[i % len(color_list)] for i, name in enumerate(log_paths)}

# 对比指标（TensorBoard scalar tags）
tags = {
    "rollout/ep_len_mean": "Average Episode Length",
    "rollout/ep_rew_mean": "Average Episode Reward",
    "rollout/success_rate": "Success Rate",
    "time/fps": "FPS"
}

# 遍历每个指标，分别绘图
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

        plt.plot(steps, values, label=name, color=colors[name], linewidth=2)

    plt.title(f"{title} Comparison", fontsize=13)
    plt.xlabel("Training Steps", fontsize=11)
    plt.ylabel("Value", fontsize=11)
    plt.legend(loc="best", fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
