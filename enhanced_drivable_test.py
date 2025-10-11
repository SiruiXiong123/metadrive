#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版可行域检测功能验证
专门用于验证check_agent_in_drivable_area函数的准确性
"""

import os
import time
import numpy as np
from metadrive.envs.top_down_env import TopDownMetaDrive
from metadrive.obs.top_down_obs_multi_channel import TopDownMultiChannel
from metadrive.utils import import_pygame

# 导入pygame
pygame = import_pygame()

def detailed_drivable_area_test():
    """详细的可行域检测测试"""
    
    cfg = {
        "map": "OO", 
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
        "image_observation": True,
        "interface_type": "image",
        "image_obs_type": "TopDownMultiChannel",
    }
    
    print("=== 详细可行域检测测试 ===")
    print("使用键盘控制车辆测试，按ESC或关闭窗口退出")
    print("键盘控制：↑加速 ↓减速 ←左转 →右转")
    print("-" * 60)
    
    env = TopDownMetaDrive(cfg)
    
    try:
        obs = env.reset()
        vehicle = env.agents["default_agent"]
        done = False
        step_count = 0
        
        # 找到TopDownMultiChannel实例
        obs_instance = None
        obs_manager = getattr(vehicle, 'observation', None)
        if obs_manager is not None:
            for obs_name, obs_obj in obs_manager.observations.items():
                if isinstance(obs_obj, TopDownMultiChannel):
                    obs_instance = obs_obj
                    print(f"✓ 找到TopDownMultiChannel实例: {obs_name}")
                    break
        
        if obs_instance is None:
            print("❌ 未找到TopDownMultiChannel实例，无法进行测试")
            return
        
        # 验证画布是否正确初始化
        print("\n=== 画布初始化检查 ===")
        try:
            canvas_size = obs_instance.canvas_background.get_size()
            print(f"✓ 背景画布大小: {canvas_size}")
            print(f"✓ 画布缩放: {obs_instance.canvas_background.scaling}")
            print(f"✓ 观察分辨率: {obs_instance.resolution}")
            print(f"✓ 最大距离: {obs_instance.max_distance}")
        except Exception as e:
            print(f"❌ 画布检查失败: {e}")
            return
        
        print("\n=== 开始逐步测试 ===")
        
        while not done and step_count < 1000:
            # 执行一步
            action = env.action_space.sample()
            obs, reward, done, info, _ = env.step(action)
            
            # 获取车辆状态
            vehicle_pos = vehicle.position
            vehicle_heading = vehicle.heading_theta
            vehicle_speed = vehicle.speed
            
            print(f"\n--- Step {step_count:4d} ---")
            print(f"车辆位置: ({vehicle_pos[0]:7.2f}, {vehicle_pos[1]:7.2f})")
            print(f"车辆朝向: {np.degrees(vehicle_heading):6.1f}°")
            print(f"车辆速度: {vehicle_speed:6.2f} m/s")
            
            # 环境状态信息
            crash_vehicle = info.get("crash_vehicle", False)
            crash_object = info.get("crash_object", False) 
            out_of_road = info.get("out_of_road", False)
            arrive_dest = info.get("arrive_dest", False)
            
            print(f"环境状态: crash_vehicle={crash_vehicle}, crash_object={crash_object}")
            print(f"          out_of_road={out_of_road}, arrive_dest={arrive_dest}")
            
            # 执行可行域检测
            try:
                print("--- 可行域检测详情 ---")
                
                # 手动执行检测逻辑来获得详细信息
                canvas_pix = obs_instance.canvas_background.vec2pix([vehicle_pos[0], vehicle_pos[1]])
                x, y = int(round(canvas_pix[0])), int(round(canvas_pix[1]))
                canvas_size = obs_instance.canvas_background.get_size()
                
                print(f"世界坐标转换: ({vehicle_pos[0]:.2f}, {vehicle_pos[1]:.2f}) -> 像素({x}, {y})")
                print(f"画布范围检查: 0 <= {x} < {canvas_size[0]}, 0 <= {y} < {canvas_size[1]}")
                
                if 0 <= x < canvas_size[0] and 0 <= y < canvas_size[1]:
                    pixel_color = obs_instance.canvas_background.get_at((x, y))
                    rgb = (pixel_color.r, pixel_color.g, pixel_color.b)
                    
                    print(f"像素颜色: RGB{rgb}")
                    
                    # 分析颜色
                    is_white = (rgb[0] > 240 and rgb[1] > 240 and rgb[2] > 240)
                    is_black = (rgb[0] < 15 and rgb[1] < 15 and rgb[2] < 15)
                    
                    print(f"颜色分析: 是否白色={is_white}, 是否黑色={is_black}")
                    
                    # 调用原始检测函数
                    is_in_drivable = obs_instance.check_agent_in_drivable_area(vehicle)
                    print(f"最终检测结果: {'✓ 在可行域内' if is_in_drivable else '✗ 不在可行域内'}")
                    
                    # 交叉验证
                    if out_of_road and is_in_drivable:
                        print("⚠️  不一致警告: 环境检测out_of_road=True, 但BEV检测在可行域内")
                    elif not out_of_road and not is_in_drivable:
                        print("⚠️  可能的检测差异: 环境out_of_road=False, 但BEV检测不在可行域")
                    else:
                        print("✓ 环境状态与BEV检测基本一致")
                        
                    # 周围像素检查（检测边界情况）
                    print("--- 周围像素检查 ---")
                    offsets = [(0,0), (-1,0), (1,0), (0,-1), (0,1), (-1,-1), (1,1), (-1,1), (1,-1)]
                    for dx, dy in offsets:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < canvas_size[0] and 0 <= ny < canvas_size[1]:
                            neighbor_color = obs_instance.canvas_background.get_at((nx, ny))
                            neighbor_rgb = (neighbor_color.r, neighbor_color.g, neighbor_color.b)
                            neighbor_white = (neighbor_rgb[0] > 240 and neighbor_rgb[1] > 240 and neighbor_rgb[2] > 240)
                            print(f"  ({dx:2d},{dy:2d}): RGB{neighbor_rgb} {'(白色)' if neighbor_white else '(有色)'}")
                    
                else:
                    print("❌ 车辆位置超出画布范围")
                    is_in_drivable = False
                    
            except Exception as e:
                print(f"❌ 检测过程出错: {e}")
                import traceback
                traceback.print_exc()
            
            step_count += 1
            
            # 控制输出频率（每5步详细输出一次）
            if step_count % 5 == 0:
                print("=" * 50)
                time.sleep(0.2)  # 减缓输出速度便于观察
            
    except KeyboardInterrupt:
        print("\n用户中断测试")
    except Exception as e:
        print(f"\n测试出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        env.close()
        print("\n=== 测试结束 ===")

def canvas_visualization_test():
    """画布可视化测试 - 保存画布图像用于调试"""
    
    cfg = {
        "map": "OO",
        "use_render": True, 
        "traffic_density": 0.0,
        "manual_control": False,
        "image_observation": True,
        "interface_type": "image",
        "image_obs_type": "TopDownMultiChannel",
    }
    
    print("=== 画布可视化测试 ===")
    
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
        
        if obs_instance is None:
            print("未找到TopDownMultiChannel实例")
            return
        
        # 运行几步确保画布正确绘制
        for i in range(10):
            action = env.action_space.sample() 
            obs, reward, done, info, _ = env.step(action)
        
        # 保存画布图像
        debug_dir = "debug_canvas"
        os.makedirs(debug_dir, exist_ok=True)
        
        try:
            # 保存背景画布
            pygame.image.save(obs_instance.canvas_background, os.path.join(debug_dir, "canvas_background.png"))
            print(f"✓ 保存背景画布: {debug_dir}/canvas_background.png")
            
            # 保存道路网络画布
            pygame.image.save(obs_instance.canvas_road_network, os.path.join(debug_dir, "canvas_road_network.png"))
            print(f"✓ 保存道路网络画布: {debug_dir}/canvas_road_network.png")
            
            # 保存道路线画布
            pygame.image.save(obs_instance.canvas_road_lines, os.path.join(debug_dir, "canvas_road_lines.png"))
            print(f"✓ 保存道路线画布: {debug_dir}/canvas_road_lines.png")
            
            print(f"\n画布图像已保存到 {debug_dir}/ 目录")
            print("您可以查看这些图像来理解可行域检测的工作原理")
            print("- canvas_background.png: 包含完整道路网络的背景画布")
            print("- canvas_road_network.png: 道路网络画布")  
            print("- canvas_road_lines.png: 仅包含道路线的画布")
            
        except Exception as e:
            print(f"保存画布图像时出错: {e}")
        
        # 测试几个不同位置的像素值
        print(f"\n=== 位置像素值分析 ===")
        test_positions = [
            vehicle.position,
            (vehicle.position[0] + 5, vehicle.position[1]),
            (vehicle.position[0] - 5, vehicle.position[1]),
            (vehicle.position[0], vehicle.position[1] + 5),
            (vehicle.position[0], vehicle.position[1] - 5),
        ]
        
        for i, pos in enumerate(test_positions):
            canvas_pix = obs_instance.canvas_background.vec2pix([pos[0], pos[1]])
            x, y = int(round(canvas_pix[0])), int(round(canvas_pix[1]))
            
            if 0 <= x < obs_instance.canvas_background.get_size()[0] and 0 <= y < obs_instance.canvas_background.get_size()[1]:
                pixel_color = obs_instance.canvas_background.get_at((x, y))
                rgb = (pixel_color.r, pixel_color.g, pixel_color.b)
                is_white = (rgb[0] > 240 and rgb[1] > 240 and rgb[2] > 240)
                
                print(f"位置 {i+1}: ({pos[0]:6.2f}, {pos[1]:6.2f}) -> 像素({x:4d}, {y:4d}) -> RGB{rgb} {'(白色背景)' if is_white else '(有色区域)'}")
            else:
                print(f"位置 {i+1}: ({pos[0]:6.2f}, {pos[1]:6.2f}) -> 超出画布范围")
        
    except Exception as e:
        print(f"可视化测试出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        env.close()

if __name__ == "__main__":
    print("可行域检测功能验证工具 - 增强版")
    print("1. 详细交互式测试（推荐）")
    print("2. 画布可视化测试")
    print("3. 运行所有测试")
    
    choice = input("请选择测试模式 (1/2/3): ").strip()
    
    if choice == "1":
        detailed_drivable_area_test()
    elif choice == "2":
        canvas_visualization_test()
    elif choice == "3":
        print("运行画布可视化测试...")
        canvas_visualization_test()
        print("\n" + "="*50)
        print("运行详细交互式测试...")
        detailed_drivable_area_test()
    else:
        print("无效选择，运行默认的详细测试...")
        detailed_drivable_area_test()