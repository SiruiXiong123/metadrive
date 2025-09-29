from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.obs.state_obs import LidarStateObservation
from metadrive.component.sensors.rgb_camera import RGBCamera
import os
sensor_size = (224, 224)

env = MetaDriveEnv(config=dict(
    use_render=False,
    agent_observation=LidarStateObservation,
    image_observation=True,
    norm_pixel=False,
    vehicle_config=dict(image_source="main_camera",
                        show_navi_mark=True,
                        show_line_to_dest=True,),
    sensors=dict(rgb_camera=(RGBCamera, *sensor_size)),
))

obs, info = env.reset()

print("Observation shape: ", obs.shape)

image = env.engine.get_sensor("rgb_camera").perceive(to_float=False)
image = image[..., [2, 1, 0]]


import matplotlib.pyplot as plt
plt.imshow(image)
plt.show()