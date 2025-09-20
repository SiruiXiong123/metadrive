import os
import numpy as np
from PIL import Image
from metadrive.component.sensors.rgb_camera import RGBCamera
from metadrive.envs.safe_metadrive_env import SafeMetaDriveEnv

# 保存路径
save_path = "./obs_image_200x100.png"

# 创建环境
env = SafeMetaDriveEnv(dict(
    use_render=False,
    image_observation=True,
    log_level=50,
    start_seed=0,
    num_scenarios=1,
    sensors={"rgb_camera": (RGBCamera, 224, 224)},
    vehicle_config=dict(image_source="rgb_camera"),
    stack_size=1,
    image_on_cuda=False,
))

# 获取观测
obs, _ = env.reset()
image_stack = obs["image"] if isinstance(obs, dict) else obs

# 如果是 cupy 转为 numpy
if hasattr(image_stack, "get"):
    image_stack = image_stack.get()

# 适配你的格式：image_stack.shape = (100, 200, 3, 1)
if image_stack.shape[-1] == 1:
    image = image_stack[:, :, :, -1]  # shape (100, 200, 3)
else:
    raise ValueError(f"Unexpected image shape: {image_stack.shape}")

# 转换 dtype
if image.dtype != np.uint8:
    image = (image * 255).clip(0, 255).astype(np.uint8)

# 保存图片
Image.fromarray(image).save(save_path)
print(f"✅ 成功保存一帧图像: {os.path.abspath(save_path)}")

env.close()
