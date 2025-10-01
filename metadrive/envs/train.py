from metadrive.envs import MetaDriveEnv
from metadrive.envs.safe_metadrive_env import SafeMetaDriveEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor
from functools import partial
from distance_and_collision_callback import MetaDriveMetricsCallback
#from ViT import CustomCombinedExtractor
#sensor_size = (128, 72)
import os
from metadrive.component.sensors.rgb_camera import RGBCamera
from stable_baselines3.common.vec_env import DummyVecEnv
from envs.only_VIT_singleframe import CustomViTPolicy



ppo_params = dict(
    learning_rate = 3e-4,
    batch_size = 64,
    n_epochs = 10,
    gamma = 0.99,
    gae_lambda = 0.95,
    clip_range = 0.2,
    ent_coef = 0.01,
    vf_coef = 0.5,
    max_grad_norm = 0.5,
)


sensor_size = (224, 224)
# sensor_size = (200, 100)



def create_env(need_monitor=False):
    env = MetaDriveEnv(dict(
        map="CCCCCCCC",
        multi_thread_render=True, 
        num_scenarios=200,
        start_seed=500,
        traffic_density=0.0,
        log_level=50,
        use_render=False,
        discrete_action=True,
        use_multi_discrete=True,      # 打开多离散模式
        image_observation=True,
        random_lane_width=True,
        random_lane_num=True,
        vehicle_config=dict(image_source="rgb_camera"),
        sensors={"rgb_camera": (RGBCamera, *sensor_size)},
        stack_size=4,
        image_on_cuda=False,
        ))
    if need_monitor:
        env = Monitor(env)
    return env


if __name__ == "__main__":
    from metadrive.component.sensors.rgb_camera import RGBCamera
    from metadrive.constants import DEFAULT_AGENT
    import numpy as np

    path = r"C:\Users\37945\OneDrive\Desktop\sac_metadrive"
    #sensor_size = (84, 60)
    set_random_seed(0)
    train_env = SubprocVecEnv([partial(create_env, True) for _ in range(1)])
    # policy_kwargs = dict(
    # features_extractor_class=CustomCombinedExtractor,  # 使用自定义特征提取器
    # features_extractor_kwargs=dict(
    #     observation_space=create_env().observation_space  # 传递 observation_space
    # ))
    # policy_kwargs = dict(
    # features_extractor_class=CustomResNetExtractor, 
    # net_arch=dict(pi=[128, 128], vf=[128, 128]),  )
    
    
    
    # policy_kwargs = dict(
    #     features_extractor_class=CustomCNN,
    #     features_extractor_kwargs=dict(features_dim=128))
    #model = PPO("MultiInputPolicy", env=train_env, n_steps=4096, 
    #            verbose=1, device="cuda", tensorboard_log=path, policy_kwargs=policy_kwargs)
    # model = PPO("CnnPolicy", env=train_env, n_steps=4096, 
    #             verbose=1, device="cuda", tensorboard_log=path)
    model = PPO(policy=CustomViTPolicy, env=train_env, n_steps=4096, verbose=1, device="cuda", **ppo_params,tensorboard_log=path)
    # #model = PPO("MlpPolicy", train_env, n_steps=4096, verbose=1, device="cuda",tensorboard_log=path)
    model.learn(total_timesteps=3000000, log_interval=4,callback=MetaDriveMetricsCallback())
    model.save(path) 

    # #conclusion
    # #加大模型训练步数
    # 暂时先用 DummyVecEnv 验证
    # train_env = DummyVecEnv([partial(create_env, True)])
    # obs, info = train_env.reset()  # 不报错再切回 SubprocVecEnv
