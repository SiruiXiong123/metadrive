import os, glob
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei','SimHei','Noto Sans CJK SC','Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False

# 你的四个实验目录（保持不变）
log_dirs = [
    r"C:\Users\37945\OneDrive\Desktop\Vit模型日志\Discrete action + two-layer neural network\PPO_1",
    r"C:\Users\37945\OneDrive\Desktop\Vit模型日志\Discrete action + single-layer neural network\PPO_1",
    r"C:\Users\37945\OneDrive\Desktop\Vit模型日志\Continuous action + two-layer neural network\PPO_1",
    r"C:\Users\37945\OneDrive\Desktop\Vit模型日志\Continuous action + single-layer neural network\PPO_1",
]

# 读取日志
runs = {}
for d in log_dirs:
    if not os.path.isdir(d): 
        print(f"⚠️ 路径不存在：{d}"); continue
    ea = event_accumulator.EventAccumulator(d)
    ea.Reload()
    runs[os.path.basename(os.path.dirname(d))] = ea

if not runs:
    raise SystemExit("未找到有效日志。")

# 求公共 scalar 指标
tag_sets = [set(ea.Tags()['scalars']) for ea in runs.values()]
common_tags = set.intersection(*tag_sets)
print(f"公共指标数量: {len(common_tags)}")

os.makedirs("比较结果", exist_ok=True)

def smooth(vals, k=1):
    if k <= 1: return vals
    out = []
    s = 0.0
    for i,v in enumerate(vals):
        s += v
        if i >= k: s -= vals[i-k]
        out.append(s / min(i+1, k))
    return out

SMOOTH_K = 1  # 想平滑就改成 5/10

# 逐指标绘图
for tag in sorted(common_tags):
    plt.figure(figsize=(9,6))
    for name, ea in runs.items():
        evs = ea.Scalars(tag)
        steps = [e.step for e in evs]
        vals  = [e.value for e in evs]
        plt.plot(steps, smooth(vals, SMOOTH_K), label=name)
    plt.xlabel("Steps")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    out = os.path.join("比较结果", f"{tag.replace('/','_')}.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.show()

print("✅ 已生成所有公共指标对比图，见 ./比较结果/")
