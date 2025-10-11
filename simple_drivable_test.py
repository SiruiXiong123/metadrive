#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版可行域检测验证脚本
专门用于验证check_agent_in_drivable_area函数
"""

import sys
import os
import time

# 添加metadrive路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'metadrive'))

try:
    from metadrive.envs.top_down_env import TopDownMetaDrive
    from metadrive.obs.top_down_obs_multi_channel import TopDownMultiChannel
    import numpy as np
    
    print("✓ MetaDrive模块导入成功")
except ImportError as e:
    print(f"❌ 导入MetaDrive模块失败: {e}")
    print("请确保您在正确的conda环境中运行此脚本")
    sys.exit(1)

def simple_test():
    """简单的可行域检测测试"""
    
    print("=== 简化版可行域检测测试 ===")
    
    # 基本配置
    cfg = {
        "map": "OO",
        "use_render": True,
        "traffic_density": 0.0,
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
    
    print("创建环境中...")
    try:
        env = TopDownMetaDrive(cfg)
        print("✓ 环境创建成功")
    except Exception as e:
        print(f"❌ 环境创建失败: {e}")
        return
    
    try:
        print("重置环境中...")
        obs = env.reset()
        vehicle = env.agents["default_agent"]
        print("✓ 环境重置成功")
        
        # 查找TopDownMultiChannel实例
        obs_instance = None
        obs_manager = getattr(vehicle, 'observation', None)
        
        if obs_manager is not None:
            print(f"找到观察管理器，观察类型: {list(obs_manager.observations.keys())}")
            
            for obs_name, obs_obj in obs_manager.observations.items():
                print(f"检查观察 {obs_name}: {type(obs_obj)}")
                if isinstance(obs_obj, TopDownMultiChannel):
                    obs_instance = obs_obj
                    print(f"✓ 找到TopDownMultiChannel实例: {obs_name}")
                    break
                elif hasattr(obs_obj, 'check_agent_in_drivable_area'):
                    obs_instance = obs_obj
                    print(f"✓ 找到带有可行域检测功能的观察实例: {obs_name}")
                    break
        else:
            print("❌ 未找到观察管理器")
        
        if obs_instance is None:
            print("❌ 未找到可行域检测功能的观察实例")
            return
        
        print(f"观察实例类型: {type(obs_instance)}")
        print(f"是否有check_agent_in_drivable_area方法: {hasattr(obs_instance, 'check_agent_in_drivable_area')}")
        
        # 运行几步测试
        print("\n=== 开始测试 ===")
        print("使用键盘控制车辆：↑加速 ↓减速 ←左转 →右转")
        print("按Ctrl+C停止测试")
        
        step_count = 0
        done = False
        
        while not done and step_count < 100:  # 限制步数避免无限循环
            # 执行一步
            action = env.action_space.sample()
            obs, reward, done, info, _ = env.step(action)
            
            # 获取车辆状态
            vehicle_pos = vehicle.position
            vehicle_speed = vehicle.speed
            
            print(f"\n--- Step {step_count:2d} ---")
            print(f"车辆位置: ({vehicle_pos[0]:6.2f}, {vehicle_pos[1]:6.2f})")
            print(f"车辆速度: {vehicle_speed:5.2f} m/s")
            
            # 环境状态
            out_of_road = info.get("out_of_road", False)
            crash_vehicle = info.get("crash_vehicle", False)
            
            print(f"环境状态: out_of_road={out_of_road}, crash_vehicle={crash_vehicle}")
            
            # 测试可行域检测
            try:
                if hasattr(obs_instance, 'check_agent_in_drivable_area'):
                    is_in_drivable = obs_instance.check_agent_in_drivable_area(vehicle)
                    print(f"BEV检测结果: {'✓ 在可行域内' if is_in_drivable else '✗ 不在可行域内'}")
                    
                    # 对比环境状态
                    if out_of_road and is_in_drivable:
                        print("⚠️  警告: 环境检测out_of_road=True, 但BEV检测在可行域内")
                    elif not out_of_road and not is_in_drivable:
                        print("⚠️  注意: BEV检测不在可行域，但环境out_of_road=False")
                    else:
                        print("✓ 环境状态与BEV检测一致")
                else:
                    print("❌ 观察实例没有check_agent_in_drivable_area方法")
                    
            except Exception as e:
                print(f"❌ 可行域检测出错: {e}")
                import traceback
                traceback.print_exc()
            
            step_count += 1
            time.sleep(0.1)  # 短暂暂停便于观察
            
        print(f"\n测试完成，共运行 {step_count} 步")
        
    except KeyboardInterrupt:
        print("\n用户中断测试")
    except Exception as e:
        print(f"❌ 测试过程中出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("关闭环境...")
        env.close()
        print("✓ 环境已关闭")

def function_inspection():
    """检查函数实现"""
    print("=== 检查可行域检测函数实现 ===")
    
    try:
        from metadrive.obs.top_down_obs_multi_channel import TopDownMultiChannel
        
        # 检查方法是否存在
        if hasattr(TopDownMultiChannel, 'check_agent_in_drivable_area'):
            print("✓ check_agent_in_drivable_area 方法存在")
            
            # 获取方法的源码信息
            import inspect
            method = getattr(TopDownMultiChannel, 'check_agent_in_drivable_area')
            signature = inspect.signature(method)
            print(f"方法签名: {signature}")
            
            # 尝试获取文档字符串
            docstring = method.__doc__
            if docstring:
                print(f"方法文档:\n{docstring}")
            else:
                print("方法没有文档字符串")
                
        else:
            print("❌ check_agent_in_drivable_area 方法不存在")
            
        # 列出所有可用方法
        methods = [m for m in dir(TopDownMultiChannel) if not m.startswith('_')]
        print(f"TopDownMultiChannel 可用方法: {methods}")
        
    except Exception as e:
        print(f"检查函数实现时出错: {e}")

if __name__ == "__main__":
    print("可行域检测功能验证工具")
    print("1. 运行简单测试")
    print("2. 检查函数实现")
    print("3. 运行所有检查")
    
    try:
        choice = input("请选择 (1/2/3): ").strip()
    except:
        choice = "1"  # 默认选择
    
    if choice == "1":
        simple_test()
    elif choice == "2":
        function_inspection()
    elif choice == "3":
        function_inspection()
        print("\n" + "="*50 + "\n")
        simple_test()
    else:
        print("无效选择，运行默认测试...")
        simple_test()