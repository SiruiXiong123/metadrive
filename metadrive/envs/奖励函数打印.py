from metadrive.obs.state_obs import StateObservation, LidarStateObservation
from metadrive.obs.observation_base import BaseObservation
from metadrive.obs.top_down_obs_multi_channel import TopDownMultiChannel
import os
import gymnasium as gym
from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.envs.top_down_env import TopDownMetaDrive  # 添加TopDown环境
from stable_baselines3 import TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from functools import partial
import numpy as np
# from EgostateAndNavigation_obs import EgoStateNavigationobservation  # 暂时注释掉以避免导入错误
import random
from stable_baselines3.common.callbacks import CheckpointCallback, EveryNTimesteps
from win32ui import ID_FILE_LOCATE
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
import time


cfg = {
    "map": "OO",
    # "num_scenarios": 500,
    # "start_seed": 123,
    "random_lane_width": True,
    "random_lane_num": False,
    "use_render": True,
    "traffic_density": 0.0,
    "traffic_mode": "hybrid",
    "manual_control": True,
    "controller": "keyboard",
    "vehicle_config": {
        "show_navi_mark": True,
        "show_line_to_navi_mark": True,
    },
    # TopDownMetaDrive的特定配置
    "frame_skip": 5,
    "frame_stack": 3,
    "post_stack": 5,
    "norm_pixel": True,
    "resolution_size": 100,
    "distance": 50,  # 增加检测距离以便更好地测试
    "image_observation": True,
}

# 使用TopDownMetaDrive环境来获得BEV可行域检查功能
print("=== 可行域检测功能验证 ===")
print("创建环境中...")
env = TopDownMetaDrive(cfg)
print("✓ 环境创建成功")

print("\n使用键盘控制车辆测试可行域检测功能:")
print("↑ 加速  ↓ 减速  ← 左转  → 右转")
print("尝试驶出道路边界来测试检测功能")
print("-" * 50)

num_episodes = 3

for episode in range(num_episodes):
    print(f"\n=== Episode {episode + 1}/{num_episodes} ===")
    obs = env.reset()
    done = False
    step_count = 0
    
    # 获取车辆对象
    vehicle = env.agents["default_agent"]
    
    # 检查是否有TopDownMultiChannel观察实例
    obs_instance = None
    
    # 尝试多种方式获取观察实例
    # 方法1: 通过vehicle.observation
    obs_manager = getattr(vehicle, 'observation', None)
    if obs_manager is not None:
        print(f"找到观察管理器，可用观察: {list(obs_manager.observations.keys())}")
        
        for obs_name, obs_obj in obs_manager.observations.items():
            print(f"检查观察 '{obs_name}': {type(obs_obj).__name__}")
            
            if isinstance(obs_obj, TopDownMultiChannel):
                obs_instance = obs_obj
                print(f"✓ 找到TopDownMultiChannel实例: {obs_name}")
                break
    
    # 方法2: 直接从环境获取观察
    if obs_instance is None:
        try:
            env_obs = env.observations.get("default_agent", None)
            if env_obs and isinstance(env_obs, TopDownMultiChannel):
                obs_instance = env_obs
                print("✓ 从环境直接获取到TopDownMultiChannel实例")
        except:
            pass
    
    # 检查最终结果
    if obs_instance is not None:
        if hasattr(obs_instance, 'check_agent_in_drivable_area'):
            print("✓ check_agent_in_drivable_area 方法存在")
        else:
            print("❌ check_agent_in_drivable_area 方法不存在")
    else:
        print("❌ 未找到TopDownMultiChannel实例，将跳过详细检测")
    
    # 统计变量
    detection_stats = {
        "in_drivable": 0,
        "out_drivable": 0,
        "errors": 0,
        "consistency_warnings": 0
    }

    while not done and step_count < 200:  # 限制最大步数
        # 使用键盘控制车辆，等待用户输入
        action = env.action_space.sample()  # 这里的 action 会被键盘输入覆盖

        obs, reward, done, info, _ = env.step(action)
        
        # 获取车辆状态信息
        vehicle_pos = vehicle.position
        vehicle_heading = vehicle.heading_theta
        vehicle_speed = vehicle.speed
        
        # 环境状态信息 - 处理不同的info类型
        if isinstance(info, dict):
            out_of_road = info.get("out_of_road", False)
            crash_vehicle = info.get("crash_vehicle", False)
            crash_object = info.get("crash_object", False)
            arrive_dest = info.get("arrive_dest", False)
        else:
            # 如果info不是字典，则使用默认值
            out_of_road = False
            crash_vehicle = False
            crash_object = False
            arrive_dest = False
        
        # 每5步输出一次详细信息
        if step_count % 5 == 0:
            print(f"\n--- Step {step_count:3d} ---")
            print(f"车辆位置: ({vehicle_pos[0]:6.2f}, {vehicle_pos[1]:6.2f})")
            print(f"车辆朝向: {np.degrees(vehicle_heading):6.1f}°")
            print(f"车辆速度: {vehicle_speed:6.2f} m/s")
            print(f"环境状态: out_of_road={out_of_road}, crash_vehicle={crash_vehicle}")
            
            # 调用可行域检测函数
            if obs_instance is not None and hasattr(obs_instance, 'check_agent_in_drivable_area'):
                try:
                    print("--- 可行域检测详情 ---")
                    
                    # 详细检测过程（手动执行检测逻辑获取更多信息）
                    canvas_pix = obs_instance.canvas_background.vec2pix([vehicle_pos[0], vehicle_pos[1]])
                    x, y = int(round(canvas_pix[0])), int(round(canvas_pix[1]))
                    canvas_size = obs_instance.canvas_background.get_size()
                    
                    print(f"坐标转换: ({vehicle_pos[0]:.2f}, {vehicle_pos[1]:.2f}) -> 像素({x}, {y})")
                    print(f"画布大小: {canvas_size}")
                    
                    if 0 <= x < canvas_size[0] and 0 <= y < canvas_size[1]:
                        pixel_color = obs_instance.canvas_background.get_at((x, y))
                        rgb = (pixel_color.r, pixel_color.g, pixel_color.b)
                        print(f"像素颜色: RGB{rgb}")
                        
                        # 调用实际检测函数
                        is_in_drivable = obs_instance.check_agent_in_drivable_area(vehicle)
                        print(f"检测结果: {'✓ 在可行域内' if is_in_drivable else '✗ 不在可行域内'}")
                        
                        # 更新统计
                        if is_in_drivable:
                            detection_stats["in_drivable"] += 1
                        else:
                            detection_stats["out_drivable"] += 1
                        
                        # 一致性检查
                        if out_of_road and is_in_drivable:
                            print("⚠️  不一致: 环境检测out_of_road=True, 但BEV检测在可行域内")
                            detection_stats["consistency_warnings"] += 1
                        elif not out_of_road and not is_in_drivable:
                            print("⚠️  可能差异: BEV检测不在可行域，但环境out_of_road=False")
                            print("    (这可能是正常的，BEV检测可能更敏感)")
                        else:
                            print("✓ 环境状态与BEV检测基本一致")
                    else:
                        print("❌ 车辆位置超出画布范围")
                        detection_stats["out_drivable"] += 1
                        
                except Exception as e:
                    print(f"❌ 可行域检测出错: {e}")
                    detection_stats["errors"] += 1
                    import traceback
                    traceback.print_exc()
            else:
                print("❌ 无法进行可行域检测")

        step_count += 1
        
        # 添加短暂延迟，便于观察输出
        time.sleep(0.05)
    
    # Episode结束统计
    total_detections = detection_stats["in_drivable"] + detection_stats["out_drivable"]
    print(f"\n=== Episode {episode + 1} 统计 ===")
    if total_detections > 0:
        print(f"总检测次数: {total_detections}")
        print(f"在可行域内: {detection_stats['in_drivable']} ({detection_stats['in_drivable']/total_detections*100:.1f}%)")
        print(f"不在可行域内: {detection_stats['out_drivable']} ({detection_stats['out_drivable']/total_detections*100:.1f}%)")
        print(f"检测错误: {detection_stats['errors']}")
        print(f"一致性警告: {detection_stats['consistency_warnings']}")
    else:
        print("本轮未进行任何检测")
    
    if done:
        print(f"Episode结束原因: {info}")

print(f"\n=== 全部测试完成 ===")
print("如果检测功能正常，您应该看到:")
print("1. 车辆在道路内时显示'在可行域内'")
print("2. 车辆偏离道路时显示'不在可行域内'")
print("3. 像素颜色信息显示白色背景(非可行域)和有色区域(可行域)")
print("4. 与环境的out_of_road状态基本一致")

# 关闭环境
env.close()