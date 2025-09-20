import gymnasium as gym
from metadrive.component.sensors.base_camera import BaseCamera
from metadrive.component.sensors.point_cloud_lidar import PointCloudLidar
import numpy as np

from metadrive.component.vehicle.base_vehicle import BaseVehicle
from metadrive.obs.observation_base import BaseObservation
from metadrive.obs.state_obs import StateObservation

_cuda_enable = True
try:
    import cupy as cp
except ImportError:
    _cuda_enable = False


class ImageStateObservation(BaseObservation):
    """
    Use ego state info, navigation info and front cam image/top down image as input
    The shape needs special handling
    """
    IMAGE = "image"
    STATE = "state"

    def __init__(self, config):
        super(ImageStateObservation, self).__init__(config)
        self.img_obs = ImageObservation(config, config["vehicle_config"]["image_source"], config["norm_pixel"])
        self.state_obs = StateObservation(config)

    @property
    def observation_space(self):
        return gym.spaces.Dict(
            {
                self.IMAGE: self.img_obs.observation_space,
                self.STATE: self.state_obs.observation_space
            }
        )
    #opencv加路径(直接画点)viewpoints 一个函数取viewpoint,一个函数画，info
    def observe(self, vehicle: BaseVehicle):
        return {self.IMAGE: self.img_obs.observe(), self.STATE: self.state_obs.observe(vehicle)}

    def destroy(self):
        super(ImageStateObservation, self).destroy()
        self.img_obs.destroy()
        self.state_obs.destroy()


class ImageObservation(BaseObservation):
    """
    Use only image info as input
    """
    STACK_SIZE = 3  # use continuous 3 image as the input

    def __init__(self, config, image_source: str, clip_rgb: bool):
        self.enable_cuda = config["image_on_cuda"]
        if self.enable_cuda:
            assert _cuda_enable, "CuPy is not enabled. Fail to set up image_on_cuda. Hint: pip install cupy-cuda11x or pip install cupy-cuda12x"
        self.STACK_SIZE = config["stack_size"]
        self.image_source = image_source
        super(ImageObservation, self).__init__(config)
        self.norm_pixel = clip_rgb
        self.state = np.zeros(self.observation_space.shape, dtype=np.float32 if self.norm_pixel else np.uint8)
        if self.enable_cuda:
            self.state = cp.asarray(self.state)

    @property
    def observation_space(self):
        sensor_cls = self.config["sensors"][self.image_source][0]
        assert sensor_cls == "MainCamera" or issubclass(sensor_cls, BaseCamera), "Sensor should be BaseCamera"
        channel = sensor_cls.num_channels if sensor_cls != "MainCamera" else 3
        shape = (self.config["sensors"][self.image_source][2],
                 self.config["sensors"][self.image_source][1]) + (channel, self.STACK_SIZE)
        if sensor_cls is PointCloudLidar:
            return gym.spaces.Box(-np.inf, np.inf, shape=shape, dtype=np.float32)
        if self.norm_pixel:
            return gym.spaces.Box(-0.0, 1.0, shape=shape, dtype=np.float32)
        else:
            return gym.spaces.Box(0, 255, shape=shape, dtype=np.uint8)

    def observe(self, new_parent_node=None, position=None, hpr=None):
        """
        Get the image Observation. By setting new_parent_node and the reset parameters, it can capture a new image from
        a different position and pose
        """
        new_obs = self.engine.get_sensor(self.image_source).perceive(self.norm_pixel, new_parent_node, position, hpr)
        self.state = cp.roll(self.state, -1, axis=-1) if self.enable_cuda else np.roll(self.state, -1, axis=-1)
        self.state[..., -1] = new_obs
        return self.state

    def get_image(self):
        return self.state.copy()[:, :, -1]

    def reset(self, env, vehicle=None):
        """
        Clear stack
        :param env: MetaDrive
        :param vehicle: BaseVehicle
        :return: None
        """
        self.state = np.zeros(self.observation_space.shape, dtype=np.float32)
        if self.enable_cuda:
            self.state = cp.asarray(self.state)

    def destroy(self):
        """
        Clear memory
        """
        super(ImageObservation, self).destroy()
        self.state = None




# import gymnasium as gym
# from metadrive.component.sensors.base_camera import BaseCamera
# from metadrive.component.sensors.point_cloud_lidar import PointCloudLidar
# import numpy as np

# from metadrive.component.vehicle.base_vehicle import BaseVehicle
# from metadrive.obs.observation_base import BaseObservation
# from metadrive.obs.state_obs import StateObservation

# _cuda_enable = True
# try:
#     import cupy as cp
# except ImportError:
#     _cuda_enable = False

# class ImageStateObservation(BaseObservation):
#     """
#     Use ego state info, navigation info and front cam image/top down image as input
#     The shape needs special handling
#     """
#     IMAGE = "image"
#     STATE = "state"

#     def __init__(self, config):
#         super(ImageStateObservation, self).__init__(config)
#         self.img_obs = ImageObservation(config, config["vehicle_config"]["image_source"], config["norm_pixel"])
#         #self.state_obs = StateObservation(config)

#     @property
#     def observation_space(self):
#         # 修改：直接返回图像的 Box 空间，非 Dict
#         return self.img_obs.observation_space

#     def observe(self, vehicle: BaseVehicle):
#         # 修改：返回纯图像数组
#         return self.img_obs.observe()

#     def destroy(self):
#         super(ImageStateObservation, self).destroy()
#         self.img_obs.destroy()
#         #self.state_obs.destroy()

# class ImageObservation(BaseObservation):
#     """
#     Use only image info as input
#     """
#     STACK_SIZE = 1  # use continuous 3 image as the input

#     def __init__(self, config, image_source: str, clip_rgb: bool):
#         self.enable_cuda = config["image_on_cuda"]
#         if self.enable_cuda:
#             assert _cuda_enable, "CuPy is not enabled. Fail to set up image_on_cuda. Hint: pip install cupy-cuda11x or pip install cupy-cuda12x"
#         self.STACK_SIZE = config["stack_size"]
#         self.image_source = image_source
#         super(ImageObservation, self).__init__(config)
#         self.norm_pixel = clip_rgb
#         # 修改：初始化 state 时用 3D 形状
#         self.state = np.zeros((self.config["sensors"][self.image_source][2], 
#                                self.config["sensors"][self.image_source][1], 
#                                3 * self.STACK_SIZE),  # 假设 channel=3
#                               dtype=np.float32 if self.norm_pixel else np.uint8)
#         if self.enable_cuda:
#             self.state = cp.asarray(self.state)

#     @property
#     def observation_space(self):
#         sensor_cls = self.config["sensors"][self.image_source][0]
#         assert sensor_cls == "MainCamera" or issubclass(sensor_cls, BaseCamera), "Sensor should be BaseCamera"
#         channel = sensor_cls.num_channels if sensor_cls != "MainCamera" else 3
#         height = self.config["sensors"][self.image_source][2]
#         width = self.config["sensors"][self.image_source][1]
#         # 修改：3D Box (C, H, W)，channels-first
#         shape = (channel * self.STACK_SIZE, height, width)
#         if sensor_cls is PointCloudLidar:
#             return gym.spaces.Box(-np.inf, np.inf, shape=shape, dtype=np.float32)
#         if self.norm_pixel:
#             return gym.spaces.Box(0.0, 1.0, shape=shape, dtype=np.float32)
#         else:
#             return gym.spaces.Box(0, 255, shape=shape, dtype=np.uint8)

#     def observe(self, new_parent_node=None, position=None, hpr=None):
#         new_obs = self.engine.get_sensor(self.image_source).perceive(self.norm_pixel, new_parent_node, position, hpr)
#         self.state = cp.roll(self.state, -3, axis=-1) if self.enable_cuda else np.roll(self.state, -3, axis=-1)  # 假设 channel=3
#         self.state[..., -3:] = new_obs  # 假设 channel=3
#         if self.enable_cuda:
#             return cp.transpose(self.state, (2, 0, 1))  # (H, W, C) → (C, H, W)
#         else:
#             return np.transpose(self.state, (2, 0, 1))

#     def get_image(self):
#         return self.state.copy()[:, :, -3:]  # 假设 channel=3

#     def reset(self, env, vehicle=None):
#         self.state = np.zeros(self.observation_space.shape, dtype=np.float32 if self.norm_pixel else np.uint8)
#         if self.enable_cuda:
#             self.state = cp.asarray(self.state)

#     def destroy(self):
#         super(ImageObservation, self).destroy()
#         #self.state = None
