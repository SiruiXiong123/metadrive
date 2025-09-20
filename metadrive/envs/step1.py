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
cfg = dict(
    num_scenarios=1,
    start_seed=0,
    random_lane_width=True,
    random_lane_num=False,
    use_render=False,  # 关闭窗口渲染，防止报错
    traffic_density=0.0,
    image_observation=True,
    vehicle_config=dict(image_source="rgb_camera"),
    sensors={"rgb_camera": (RGBCamera, *sensor_size)},
    stack_size=stack_size,
)

# 创建 DummyVecEnv 包裹的单环境
def create_env_for_testing():
    def _env_fn():
        return MetaDriveEnv(cfg)
    return DummyVecEnv([_env_fn])

if __name__ == '__main__':
    env = create_env_for_testing()

    # 加载 SAC 模型
    SAC_Path = os.path.join(base_path, 'sac_metadrive.zip')
    model = SAC.load(SAC_Path, env=env, deterministic=True)

    # 获取一帧观测
    obs = env.reset()
    action, _ = model.predict(obs, deterministic=True)
    print("🚗 Predicted action:", action)

    # ✅ 修复：正确提取图像（取第一个环境，最后一帧）
    img = obs["image"][0][:, :, :, -1]  # shape (H, W, C) or (C, H, W)
    if img.shape[0] == 3:  # channel-first 需要转置
        img = np.transpose(img, (1, 2, 0))

    # # ✅ 显示图像
    # plt.imshow(img)
    # plt.title("RGB Camera Last Frame")
    # plt.axis('off')
    # plt.show()

    env.close()
