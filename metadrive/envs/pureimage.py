import gymnasium as gym
import numpy as np
from metadrive.obs.observation_base import BaseObservation
from metadrive.component.sensors.base_camera import BaseCamera
from metadrive.component.sensors.point_cloud_lidar import PointCloudLidar

_cuda_enable = True
try:
    import cupy as cp
except ImportError:
    _cuda_enable = False

class PureImageObservation(BaseObservation):
    """
    Only return image observation (no dictionary). Compatible with SB3 CnnPolicy.
    """

    def __init__(self, config):
        super(PureImageObservation, self).__init__(config)
        self.image_source = config["vehicle_config"]["image_source"]
        self.stack_size = config.get("stack_size", 1)
        self.norm_pixel = config.get("norm_pixel", False)  # 修改：默认 False，以匹配 SB3 uint8 [0,255]
        self.enable_cuda = config.get("image_on_cuda", False)

        if self.enable_cuda:
            assert _cuda_enable, "CuPy not found! Install it or set image_on_cuda=False."

        sensor_cls = config["sensors"][self.image_source][0]
        assert sensor_cls == "MainCamera" or issubclass(sensor_cls, BaseCamera), "Sensor must be BaseCamera"
        channel = sensor_cls.num_channels if sensor_cls != "MainCamera" else 3

        self.channel = channel
        self.height = config["sensors"][self.image_source][2]
        self.width = config["sensors"][self.image_source][1]

        # 修改：使用 channels-last 形状 (H, W, C * stack_size)
        self.obs_shape = (self.height, self.width, self.channel * self.stack_size)
        
        # 修改：根据 norm_pixel 设置 dtype 和初始 state
        dtype = np.float32 if self.norm_pixel else np.uint8
        self.state = np.zeros(self.obs_shape, dtype=dtype)
        if self.enable_cuda:
            self.state = cp.asarray(self.state)

    @property
    def observation_space(self):
        # 修改：根据 norm_pixel 设置 low/high/dtype，以匹配 SB3 期望
        if self.norm_pixel:
            low, high, dtype = 0.0, 1.0, np.float32
        else:
            low, high, dtype = 0, 255, np.uint8
        return gym.spaces.Box(low, high, shape=self.obs_shape, dtype=dtype)

    def observe(self, *args, **kwargs):
        # 修改：perceive() 返回 (H, W, C)，无需 transpose（保持 channels-last）
        new_img = self.engine.get_sensor(self.image_source).perceive(self.norm_pixel)  # shape: (H, W, C)
        
        if self.enable_cuda:
            new_img = cp.asarray(new_img)
            self.state = cp.roll(self.state, -self.channel, axis=-1)  # roll 沿通道轴
            self.state[..., -self.channel:] = new_img  # 赋值到最后 C 通道
        else:
            self.state = np.roll(self.state, -self.channel, axis=-1)
            self.state[..., -self.channel:] = new_img

        return self.state

    def reset(self, env, vehicle=None):
        dtype = np.float32 if self.norm_pixel else np.uint8
        self.state = np.zeros(self.obs_shape, dtype=dtype)
        if self.enable_cuda:
            self.state = cp.asarray(self.state)

    def destroy(self):
        super().destroy()
        self.state = None