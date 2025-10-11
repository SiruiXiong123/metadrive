#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可行域检测功能验证脚本
用于测试TopDownMultiChannel类中的check_agent_in_drivable_area方法
"""

import os
import time
import numpy as np
from metadrive.envs.top_down_env import TopDownMetaDrive
from metadrive.obs.top_down_obs_multi_channel import TopDownMultiChannel

def test_drivable_area_detection():
    """测试可行域检测功能"""
    
    # 配置环境
    cfg = {
        "map": "OO",  # 使用O型地图便于测试
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
        # 确保使用TopDownMultiChannel观察
        "image_observation": True,
        "interface_type": "image",
        "image_obs_type": "TopDownMultiChannel",
    }
    
    print("=== 可行域检测功能验证开始 ===")
    print("使用键盘控制车辆，尝试驶入和驶出道路区域来测试检测功能")
    print("键盘控制：↑加速 ↓减速 ←左转 →右转")
    print("-" * 50)
    
    # 创建环境
    env = TopDownMetaDrive(cfg)
    
    try:
        num_episodes = 3
        
        for episode in range(num_episodes):
            print(f"\n=== Episode {episode + 1} ===")
            obs = env.reset()
            done = False
            step_count = 0
            
            # 记录检测结果统计
            detection_stats = {
                "in_drivable": 0,
                "out_drivable": 0,
                "detection_errors": 0
            }
            
            while not done and step_count < 500:  # 限制最大步数
                # 键盘控制（如果有输入会覆盖随机动作）
                action = env.action_space.sample()
                
                # 执行动作
                obs, reward, done, info, _ = env.step(action)
                
                # 获取车辆对象
                vehicle = env.agents["default_agent"]
                
                # 获取车辆当前状态信息
                vehicle_pos = vehicle.position
                vehicle_heading = vehicle.heading_theta
                vehicle_speed = vehicle.speed
                
                print(f"\nStep {step_count:3d}:")
                print(f"  车辆位置: ({vehicle_pos[0]:6.2f}, {vehicle_pos[1]:6.2f})")
                print(f"  车辆朝向: {vehicle_heading:6.2f}°")
                print(f"  车辆速度: {vehicle_speed:6.2f} m/s")
                
                # 查找TopDownMultiChannel观察实例
                obs_instance = None
                obs_manager = getattr(vehicle, 'observation', None)
                
                if obs_manager is not None:
                    for obs_name, obs_obj in obs_manager.observations.items():
                        if isinstance(obs_obj, TopDownMultiChannel):
                            obs_instance = obs_obj
                            break
                
                if obs_instance is not None:
                    try:
                        # 调用可行域检测函数
                        print("  --- 可行域检测结果 ---")
                        is_in_drivable = obs_instance.check_agent_in_drivable_area(vehicle)
                        
                        # 更新统计
                        if is_in_drivable:
                            detection_stats["in_drivable"] += 1
                        else:
                            detection_stats["out_drivable"] += 1
                            
                        print(f"  ✓ 检测完成: {'在可行域内' if is_in_drivable else '不在可行域内'}")
                        
                        # 如果车辆不在可行域内，给出警告
                        if not is_in_drivable:
                            print("  ⚠️  警告: 车辆可能偏离了道路!")
                        
                    except Exception as e:
                        print(f"  ❌ 检测出错: {e}")
                        detection_stats["detection_errors"] += 1
                        
                else:
                    print("  ❌ 未找到TopDownMultiChannel观察实例")
                
                # 额外的验证：检查是否与环境内置的碰撞检测一致
                try:
                    # 检查车辆是否与边界碰撞
                    crash_vehicle = info.get("crash_vehicle", False)
                    crash_object = info.get("crash_object", False) 
                    out_of_road = info.get("out_of_road", False)
                    
                    print(f"  环境状态: crash_vehicle={crash_vehicle}, crash_object={crash_object}, out_of_road={out_of_road}")
                    
                    # 如果环境检测到out_of_road，我们的检测应该也检测到不在可行域
                    if out_of_road and obs_instance is not None:
                        try:
                            our_detection = obs_instance.check_agent_in_drivable_area(vehicle)
                            if our_detection:
                                print("  ⚠️  不一致: 环境检测到out_of_road，但我们的检测显示在可行域内")
                            else:
                                print("  ✓ 一致: 环境和我们的检测都显示车辆不在可行域")
                        except:
                            pass
                            
                except Exception as e:
                    print(f"  状态检查出错: {e}")
                
                step_count += 1
                
                # 添加短暂延迟使输出更容易阅读
                time.sleep(0.1)
            
            # 显示本轮统计
            total_detections = detection_stats["in_drivable"] + detection_stats["out_drivable"]
            print(f"\n=== Episode {episode + 1} 统计 ===")
            print(f"总检测次数: {total_detections}")
            print(f"在可行域内: {detection_stats['in_drivable']} ({detection_stats['in_drivable']/max(total_detections,1)*100:.1f}%)")
            print(f"不在可行域内: {detection_stats['out_drivable']} ({detection_stats['out_drivable']/max(total_detections,1)*100:.1f}%)")
            print(f"检测错误: {detection_stats['detection_errors']}")
            
            if done:
                print(f"Episode结束原因: {info}")
    
    except KeyboardInterrupt:
        print("\n用户中断测试")
    except Exception as e:
        print(f"\n测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        env.close()
        print("\n=== 测试完成 ===")

def test_pixel_color_analysis():
    """测试像素颜色分析功能"""
    print("\n=== 像素颜色分析测试 ===")
    
    cfg = {
        "map": "OO",
        "use_render": True,
        "traffic_density": 0.0,
        "manual_control": False,  # 自动驾驶以便测试不同位置
        "image_observation": True,
        "interface_type": "image", 
        "image_obs_type": "TopDownMultiChannel",
    }
    
    env = TopDownMetaDrive(cfg)
    
    try:
        obs = env.reset()
        vehicle = env.agents["default_agent"]
        
        # 获取TopDownMultiChannel实例
        obs_instance = None
        obs_manager = getattr(vehicle, 'observation', None)
        if obs_manager is not None:
            for obs_name, obs_obj in obs_manager.observations.items():
                if isinstance(obs_obj, TopDownMultiChannel):
                    obs_instance = obs_obj
                    break
        
        if obs_instance is not None:
            print("找到TopDownMultiChannel实例，开始像素分析测试...")
            
            # 测试几个不同的位置
            test_positions = [
                vehicle.position,  # 当前位置（应该在道路上）
                (vehicle.position[0] + 10, vehicle.position[1]),  # 向前10米
                (vehicle.position[0], vehicle.position[1] + 5),   # 向右5米
                (vehicle.position[0] - 10, vehicle.position[1]),  # 向后10米
                (vehicle.position[0], vehicle.position[1] - 5),   # 向左5米
            ]
            
            for i, test_pos in enumerate(test_positions):
                print(f"\n测试位置 {i+1}: ({test_pos[0]:.2f}, {test_pos[1]:.2f})")
                
                # 临时修改车辆位置进行测试
                original_pos = vehicle.position
                try:
                    # 直接设置位置（仅用于测试）
                    vehicle.set_position(test_pos)
                    
                    # 检测可行域
                    is_in_drivable = obs_instance.check_agent_in_drivable_area(vehicle)
                    print(f"  检测结果: {'在可行域' if is_in_drivable else '不在可行域'}")
                    
                except Exception as e:
                    print(f"  测试位置时出错: {e}")
                finally:
                    # 恢复原始位置
                    vehicle.set_position(original_pos)
        else:
            print("未找到TopDownMultiChannel实例")
            
    except Exception as e:
        print(f"像素分析测试出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        env.close()

if __name__ == "__main__":
    print("可行域检测功能验证工具")
    print("1. 交互式测试（键盘控制）")
    print("2. 像素分析测试（自动）")
    
    choice = input("请选择测试模式 (1/2): ").strip()
    
    if choice == "1":
        test_drivable_area_detection()
    elif choice == "2":
        test_pixel_color_analysis()
    else:
        print("无效选择，运行默认的交互式测试...")
        test_drivable_area_detection()