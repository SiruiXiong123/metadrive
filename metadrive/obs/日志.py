from tensorboard.backend.event_processing import event_accumulator

log_file = r"C:\Users\37945\OneDrive\Desktop\dingding\events.out.tfevents.1757665964.c113.1699681.0"

ea = event_accumulator.EventAccumulator(log_file)
ea.Reload()

# 找出所有 scalar
print("所有 scalar 标签:", ea.Tags()["scalars"])

# 过滤出和 fps 有关的
fps_tags = [t for t in ea.Tags()["scalars"] if "fps" in t.lower()]
print("检测到的 fps 标签:", fps_tags)

# 打印前 20 个数据点
for tag in fps_tags:
    events = ea.Scalars(tag)
    print(f"\nTag: {tag}")
    for e in events[:20]:
        print(f"Step {e.step}: {e.value}")
