import copy
from metadrive.component.navigation_module.node_network_navigation import NodeNetworkNavigation
from typing import Union

import numpy as np
import math
import os
import sys

# 添加TTC和EPF模块的导入
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from ttc_simple import calculate_ttc_collision_risk
    TTC_AVAILABLE = True
except ImportError:
    # print("Warning: TTC module not found")
    TTC_AVAILABLE = False

try:
    from epf_simple import calculate_epf_collision_risk
    EPF_AVAILABLE = True
except ImportError:
    # print("Warning: EPF module not found")
    EPF_AVAILABLE = False


from metadrive.component.algorithm.blocks_prob_dist import PGBlockDistConfig
from metadrive.component.map.base_map import BaseMap
from metadrive.component.map.pg_map import parse_map_config, MapGenerateMethod
from metadrive.component.pgblock.first_block import FirstPGBlock
from metadrive.constants import DEFAULT_AGENT, TerminationState
from metadrive.envs.base_env import BaseEnv
from metadrive.manager.traffic_manager import TrafficMode
from metadrive.utils import clip, Config

METADRIVE_DEFAULT_CONFIG = dict(
    # ===== Generalization =====
    start_seed=0,
    num_scenarios=1,

    # ===== PG Map Config =====
    map=2,  # int or string: an easy way to fill map_config
    block_dist_config=PGBlockDistConfig,
    random_lane_width=False,
    random_lane_num=False,
    map_config={
        BaseMap.GENERATE_TYPE: MapGenerateMethod.BIG_BLOCK_NUM,
        BaseMap.GENERATE_CONFIG: None,  # it can be a file path / block num / block ID sequence
        BaseMap.LANE_WIDTH: 3.5,
        BaseMap.LANE_NUM: 3,
        "exit_length": 50,
        "start_position": [0, 0],
    },
    store_map=True,

    # ===== Traffic =====
    traffic_density=0.1,
    need_inverse_traffic=False,
    traffic_mode=TrafficMode.Trigger,  # "Respawn", "Trigger"
    random_traffic=False,  # Traffic is randomized at default.
    # this will update the vehicle_config and set to traffic
    traffic_vehicle_config=dict(
        show_navi_mark=False,
        show_dest_mark=False,
        enable_reverse=False,
        show_lidar=False,
        show_lane_line_detector=False,
        show_side_detector=False,
    ),

    # ===== Object =====
    accident_prob=0.,  # accident may happen on each block with this probability, except multi-exits block
    static_traffic_object=True,  # object won't react to any collisions

    # ===== Others =====
    use_AI_protector=False,
    save_level=0.5,
    horizon=1000,

    # ===== Agent =====
    random_spawn_lane_index=True,
    vehicle_config=dict(navigation_module=NodeNetworkNavigation),
    agent_configs={
        DEFAULT_AGENT: dict(
            use_special_color=True,
            spawn_lane_index=(FirstPGBlock.NODE_1, FirstPGBlock.NODE_2, 0),
        )
    },

    # ===== Reward Scheme =====
    # See: https://github.com/metadriverse/metadrive/issues/283
    success_reward=100.0,
    out_of_road_penalty=10.0,
    crash_vehicle_penalty=10.0,
    crash_object_penalty=10.0,
    crash_sidewalk_penalty=10.0,
    driving_reward=1.0,
    speed_reward=0.1,
    use_lateral_reward=False,

    # ===== Cost Scheme =====
    crash_vehicle_cost=1.0,
    crash_object_cost=1.0,
    out_of_road_cost=1.0,

    # ===== Termination Scheme =====
    out_of_route_done=False,
    out_of_road_done=True,
    on_continuous_line_done=True,
    on_broken_line_done=False,
    crash_vehicle_done=True,
    crash_object_done=True,
    crash_human_done=True,
)


class MetaDriveEnv(BaseEnv):
    @classmethod
    def default_config(cls) -> Config:
        config = super(MetaDriveEnv, cls).default_config()
        config.update(METADRIVE_DEFAULT_CONFIG)
        config.register_type("map", str, int)
        config["map_config"].register_type("config", None)
        return config

    def __init__(self, config: Union[dict, None] = None):
        self.default_config_copy = Config(self.default_config(), unchangeable=True)
        super(MetaDriveEnv, self).__init__(config)

        # scenario setting
        self.start_seed = self.start_index = self.config["start_seed"]
        self.env_num = self.num_scenarios
        
        # TTC功能可用性标志
        self.ttc_available = TTC_AVAILABLE

    def _post_process_config(self, config):
        config = super(MetaDriveEnv, self)._post_process_config(config)
        if not config["norm_pixel"]:
            self.logger.warning(
                "You have set norm_pixel = False, which means the observation will be uint8 values in [0, 255]. "
                "Please make sure you have parsed them later before feeding them to network!"
            )

        config["map_config"] = parse_map_config(
            easy_map_config=config["map"], new_map_config=config["map_config"], default_config=self.default_config_copy
        )
        config["vehicle_config"]["norm_pixel"] = config["norm_pixel"]
        config["vehicle_config"]["random_agent_model"] = config["random_agent_model"]
        target_v_config = copy.deepcopy(config["vehicle_config"])
        if not config["is_multi_agent"]:
            target_v_config.update(config["agent_configs"][DEFAULT_AGENT])
            config["agent_configs"][DEFAULT_AGENT] = target_v_config
        return config

    def done_function(self, vehicle_id: str):
        vehicle = self.agents[vehicle_id]
        done = False
        max_step = self.config["horizon"] is not None and self.episode_lengths[vehicle_id] >= self.config["horizon"]
        is_success = self._is_arrive_destination(vehicle)
        done_info = {
            TerminationState.CRASH_VEHICLE: vehicle.crash_vehicle,
            TerminationState.CRASH_OBJECT: vehicle.crash_object,
            TerminationState.CRASH_BUILDING: vehicle.crash_building,
            TerminationState.CRASH_HUMAN: vehicle.crash_human,
            TerminationState.CRASH_SIDEWALK: vehicle.crash_sidewalk,
            TerminationState.OUT_OF_ROAD: self._is_out_of_road(vehicle),
            TerminationState.SUCCESS: self._is_arrive_destination(vehicle),
            TerminationState.MAX_STEP: max_step,
            TerminationState.ENV_SEED: self.current_seed,
            "is_success": is_success if self._is_arrive_destination(vehicle) else False,
        }

        done_info[TerminationState.CRASH] = (
            done_info[TerminationState.CRASH_VEHICLE] or done_info[TerminationState.CRASH_OBJECT]
            or done_info[TerminationState.CRASH_BUILDING] or done_info[TerminationState.CRASH_SIDEWALK]
            or done_info[TerminationState.CRASH_HUMAN]
        )

        if done_info[TerminationState.SUCCESS]:
            done = True
            self.logger.info(
                "Episode ended! Scenario Index: {} Reason: arrive_dest.".format(self.current_seed),
                extra={"log_once": True},
            )
        if done_info[TerminationState.OUT_OF_ROAD] and self.config["out_of_road_done"]:
            done = True
            self.logger.info(
                "Episode ended! Scenario Index: {} Reason: out_of_road.".format(self.current_seed),
                extra={"log_once": True}
            )
        if done_info[TerminationState.CRASH_VEHICLE] and self.config["crash_vehicle_done"]:
            done = True
            self.logger.info(
                "Episode ended! Scenario Index: {} Reason: crash vehicle ".format(self.current_seed),
                extra={"log_once": True}
            )
        if done_info[TerminationState.CRASH_OBJECT] and self.config["crash_object_done"]:
            done = True
            self.logger.info(
                "Episode ended! Scenario Index: {} Reason: crash object ".format(self.current_seed),
                extra={"log_once": True}
            )
        if done_info[TerminationState.CRASH_BUILDING]:
            done = True
            self.logger.info(
                "Episode ended! Scenario Index: {} Reason: crash building ".format(self.current_seed),
                extra={"log_once": True}
            )
        if done_info[TerminationState.CRASH_HUMAN] and self.config["crash_human_done"]:
            done = True
            self.logger.info(
                "Episode ended! Scenario Index: {} Reason: crash human".format(self.current_seed),
                extra={"log_once": True}
            )
        if done_info[TerminationState.MAX_STEP]:
            if self.config["truncate_as_terminate"]:
                done = True
            self.logger.info(
                "Episode ended! Scenario Index: {} Reason: max step ".format(self.current_seed),
                extra={"log_once": True}
            )
        return done, done_info

    def cost_function(self, vehicle_id: str):
        vehicle = self.agents[vehicle_id]
        step_info = dict()
        step_info["cost"] = 0
        if self._is_out_of_road(vehicle):
            step_info["cost"] = self.config["out_of_road_cost"]
        elif vehicle.crash_vehicle:
            step_info["cost"] = self.config["crash_vehicle_cost"]
        elif vehicle.crash_object:
            step_info["cost"] = self.config["crash_object_cost"]
        return step_info['cost'], step_info

    @staticmethod
    def _is_arrive_destination(vehicle):
        """
        Args:
            vehicle: The BaseVehicle instance.

        Returns:
            flag: Whether this vehicle arrives its destination.
        """
        long, lat = vehicle.navigation.final_lane.local_coordinates(vehicle.position)
        flag = (vehicle.navigation.final_lane.length - 5 < long < vehicle.navigation.final_lane.length + 5) and (
            vehicle.navigation.get_current_lane_width() / 2 >= lat >=
            (0.5 - vehicle.navigation.get_current_lane_num()) * vehicle.navigation.get_current_lane_width()
        )
        return flag

    def _is_out_of_road(self, vehicle):
        # A specified function to determine whether this vehicle should be done.
        # return vehicle.on_yellow_continuous_line or (not vehicle.on_lane) or vehicle.crash_sidewalk
        ret = not vehicle.on_lane
        if self.config["out_of_route_done"]:
            ret = ret or vehicle.out_of_route
        elif self.config["on_continuous_line_done"]:
            ret = ret or vehicle.on_yellow_continuous_line or vehicle.on_white_continuous_line or vehicle.crash_sidewalk
        if self.config["on_broken_line_done"]:
            ret = ret or vehicle.on_broken_line
        return ret

    def denormalize_other_vehicles_info(self, obs, vehicle):
        """
        反归一化其他车辆信息，从观测中提取并转换为实际物理值
        :param obs: 完整观测向量 (43维)
        :param vehicle: 当前车辆实例
        :return: 包含4个车辆信息的列表，每个车辆包含6个特征的字典
        """
        # 观测结构：前19维是基础状态，后24维是4个车辆的信息（每车6维）
        other_vehicles_info = []
        start_idx = 19  # 其他车辆信息从第19维开始
        
        perceive_distance = vehicle.config["lidar"]["distance"]  # 50米
        max_speed = vehicle.max_speed_km_h  # 最大速度
        max_length = getattr(vehicle, 'MAX_LENGTH', 10.0)  # 最大长度，默认10米
        max_width = getattr(vehicle, 'MAX_WIDTH', 5.0)     # 最大宽度，默认5米
        
        for i in range(4):  # 4个其他车辆
            base_idx = start_idx + i * 6
            if base_idx + 5 < len(obs):
                # 反归一化每个特征
                relative_pos_x = (obs[base_idx] * 2 - 1) * perceive_distance      # 前后距离 (米)
                relative_pos_y = (obs[base_idx + 1] * 2 - 1) * perceive_distance  # 左右距离 (米)
                relative_vel_x = (obs[base_idx + 2] * 2 - 1) * max_speed          # 前后相对速度 (km/h)
                relative_vel_y = (obs[base_idx + 3] * 2 - 1) * max_speed          # 左右相对速度 (km/h)
                vehicle_length = obs[base_idx + 4] * max_length                   # 车辆长度 (米)
                vehicle_width = obs[base_idx + 5] * max_width                     # 车辆宽度 (米)
                
                # 计算实际距离和速度
                distance = np.sqrt(relative_pos_x**2 + relative_pos_y**2)
                relative_speed = np.sqrt(relative_vel_x**2 + relative_vel_y**2)
                
                vehicle_info = {
                    'relative_pos_x': relative_pos_x,       # 前后相对位置 (米)
                    'relative_pos_y': relative_pos_y,       # 左右相对位置 (米) 
                    'relative_vel_x': relative_vel_x,       # 前后相对速度 (km/h)
                    'relative_vel_y': relative_vel_y,       # 左右相对速度 (km/h)
                    'length': vehicle_length,               # 车辆长度 (米)
                    'width': vehicle_width,                 # 车辆宽度 (米)
                    'distance': distance,                   # 总距离 (米)
                    'relative_speed': relative_speed,       # 相对速度大小 (km/h)
                    'is_valid': distance > 0.1             # 是否有效（距离>0.1米认为是真实车辆）
                }
                other_vehicles_info.append(vehicle_info)
        
        return other_vehicles_info

    def reward_function(self, vehicle_id: str):
        """
        Override this func to get a new reward function
        :param vehicle_id: id of BaseVehicle
        :return: reward
        """
        vehicle = self.agents[vehicle_id]
        step_info = dict()
        
        # 获取当前观测并反归一化其他车辆信息
        ttc_penalty = 0.0
        epf_penalty = 0.0
        
        # 直接使用observation管理器获取当前观测
        if vehicle_id in self.observations:
            current_obs = self.observations[vehicle_id].observe(vehicle)
            other_vehicles = self.denormalize_other_vehicles_info(current_obs, vehicle)
                
            # 计算碰撞风险 (如果有车辆检测到)
            if len(other_vehicles) > 0:
                # TTC碰撞风险
                # if TTC_AVAILABLE:
                #     ttc_penalty = calculate_ttc_collision_risk(
                #         other_vehicles, penalty_weight=5.0, tau=1.0, max_risk=1.5
                #     )
                #     print(f"TTC Penalty: {ttc_penalty:.3f}")
                
                # EPF椭圆势场风险
                if EPF_AVAILABLE:
                    epf_penalty = calculate_epf_collision_risk(
                        other_vehicles, penalty_weight=5.0, max_risk=1.5
                    )
                    # print(f"EPF Penalty: {epf_penalty:.3f}")
        


        # Reward for moving forward in current lane
        if vehicle.lane in vehicle.navigation.current_ref_lanes:
            current_lane = vehicle.lane
            positive_road = 1
        else:
            current_lane = vehicle.navigation.current_ref_lanes[0]
            current_road = vehicle.navigation.current_road
            positive_road = 1 if not current_road.is_negative_road() else -1
        long_last, _ = current_lane.local_coordinates(vehicle.last_position)
        long_now, lateral_now = current_lane.local_coordinates(vehicle.position)

        # reward for lane keeping, without it vehicle can learn to overtake but fail to keep in lane
        if self.config["use_lateral_reward"]:
            lateral_factor = clip(1 - 2 * abs(lateral_now) / vehicle.navigation.get_current_lane_width(), 0.0, 1.0)
        else:
            lateral_factor = 1.0

        nav = vehicle.navigation

        # 推荐用当前参考车道作为 ref_lane
        # ref_lane = nav.current_ref_lanes[0]   # 或 ref_lane = vehicle.lane

        # # 调用（lanes_id=0 表示当前路段，lanes_id=1 表示下一路段）
        # navi_info, lanes_heading, cp = nav._get_info_for_checkpoint(lanes_id=0, ref_lane=ref_lane, ego_vehicle=vehicle)

        # print("navi_info:", navi_info[0],navi_info[1])
        cp, _ = vehicle.navigation.get_checkpoints()
        dist_m = np.linalg.norm(np.array(cp) - np.array(vehicle.position))
        sigma = 3

        reward = 0.0
        






        #-------------checkpoint reward----------------
        R_ckpt = 0.0
        ckpt_reward = math.exp(- (dist_m /10) ** 2)

        if vehicle.speed_km_h / vehicle.max_speed_km_h>0.1:
            R_ckpt = 2 * ckpt_reward
        else:
            R_ckpt += 0

        current_reference_lane = vehicle.lane

        heading_diff = vehicle.heading_diff(current_reference_lane)
        heading_reward = 0.15 * (1.0 / (abs(0.5 - heading_diff) + 1.0))
        # print('heading_diff:', heading_diff)
        v_t = vehicle.speed_km_h
        v_d = 80
        R_speed = 0.15 * (1.0 / ((abs(v_t - v_d) / v_d) + 1.0))

        #------------smooth reward----------------
        steering_last = clip((vehicle.last_current_action[1][0] + 1) / 2, 0.0, 1.0)
        steering_now = clip((vehicle.steering / vehicle.MAX_STEERING + 1) / 2, 0.0, 1.0)
        delta_steer = abs(steering_now - steering_last)
        R_smooth = 0.05 * (1.0 - delta_steer)
        R_smooth = max(R_smooth, 0.0)
        

        #-------------out of road penalty----------------
        dleft = vehicle.dist_to_left_side        # 左侧到道路边界的距离
        dright = vehicle.dist_to_right_side      # 右侧到道路边界的距离
        W = vehicle.WIDTH                        # 车辆宽度
        Wlane = vehicle.navigation.get_current_lane_width()  # 当前车道宽度
        # print('左侧距离:', dleft, '右侧距离:', dright, '车辆宽度:', W, '车道宽度:', Wlane)
        if dleft < 0.5 * Wlane:
            P_left = 1 / (((dleft - 0.5 * W) / (0.5 * Wlane)) ** 2 + 1.0)
        else:
            P_left = 0.0

        # 计算右侧风险
        if dright < 0.5 * Wlane:
            P_right = 1 / (((dright - 0.5 * W) / (0.5 * Wlane)) ** 2 + 1.0)
        else:
            P_right = 0.0

        # 计算越界惩罚
        R_out_of_road = -3 * (P_left + P_right)

        is_on_path = vehicle.navigation.is_on_recommended_path(vehicle)

        # 静默检测推荐路径状态（不打印）
        if is_on_path and vehicle.speed_km_h / vehicle.max_speed_km_h>0.05:
            out_drivable_area_penalty = 0.5
            # print("? 智能体在推荐路径上")
        else:
            out_drivable_area_penalty = -20
            # print("? 智能体偏离了推荐路径")


        #加入view points奖励
        reward += self.config["driving_reward"] * (long_now - long_last) * lateral_factor * positive_road
        reward += self.config["speed_reward"] * (vehicle.speed_km_h / vehicle.max_speed_km_h) * positive_road
        # reward += R_speed * positive_road
        #reward += R_ckpt
        #reward += out_drivable_area_penalty
        #reward += R_smooth
        #reward += heading_reward
        #reward += R_out_of_road
        
        # 应用碰撞风险惩罚
        # reward -= ttc_penalty  # TTC风险惩罚
        #reward -= epf_penalty  # EPF风险惩罚
        
        step_info["step_reward"] = reward
        # print('step_reward:', reward)
        # print('R_ckpt:', R_ckpt)
        # if R_drivable_area != 0:
        #     print('可行域惩罚:', R_drivable_area)
        # print('step_reward:', reward)
        # print('出界惩罚:', R_out_of_road)
        
        # print(f"离导航点距离: {dist_m}")
        # print('靠近导航点奖励:', R_ckpt)
        if self._is_arrive_destination(vehicle):
            reward = +self.config["success_reward"]
        elif self._is_out_of_road(vehicle):
            reward = -self.config["out_of_road_penalty"]
        elif vehicle.crash_vehicle:
            reward = -self.config["crash_vehicle_penalty"]
        elif vehicle.crash_object:
            reward = -self.config["crash_object_penalty"]
        elif vehicle.crash_sidewalk:
            reward = -self.config["crash_sidewalk_penalty"]
        step_info["route_completion"] = vehicle.navigation.route_completion

        return reward, step_info

    def setup_engine(self):
        super(MetaDriveEnv, self).setup_engine()
        from metadrive.manager.traffic_manager import PGTrafficManager
        from metadrive.manager.pg_map_manager import PGMapManager
        from metadrive.manager.object_manager import TrafficObjectManager
        self.engine.register_manager("map_manager", PGMapManager())
        self.engine.register_manager("traffic_manager", PGTrafficManager())
        if abs(self.config["accident_prob"] - 0) > 1e-2:
            self.engine.register_manager("object_manager", TrafficObjectManager())


if __name__ == '__main__':

    def _act(env, action):
        assert env.action_space.contains(action)
        obs, reward, terminated, truncated, info = env.step(action)
        assert env.observation_space.contains(obs)
        assert np.isscalar(reward)
        assert isinstance(info, dict)

    # 添加激光雷达配置来包含其他车辆信息
    config = {
        "vehicle_config": {
            "lidar": {
                "num_lasers": 120,  # 激光束数量
                "distance": 50,     # 探测距离
                "num_others": 4,    # 检测其他车辆数量
                "add_others_navi": False  # 是否包含其他车辆导航信息
            }
        },
        "traffic_density": 0.3  # 增加交通密度以便观察到其他车辆
    }
    
    env = MetaDriveEnv(config)
    try:
        obs, _ = env.reset()
        # print(f"观测值维度: {obs.shape}, 观测空间: {env.observation_space}")
        # print(f"配置中的激光雷达设置: {env.config['vehicle_config']['lidar']}")
        
        # 测试反归一化函数
        vehicle = env.agent  # 使用单智能体模式的agent属性
        other_vehicles_info = env.denormalize_other_vehicles_info(obs, vehicle)
        # print("\n=== 其他车辆信息 (反归一化后) ===")
        # for i, info in enumerate(other_vehicles_info):
        #     if info['is_valid']:
        #         print(f"车辆 {i+1}: 距离={info['distance']:.1f}m, "
        #               f"位置=({info['relative_pos_x']:.1f}, {info['relative_pos_y']:.1f}), "
        #               f"尺寸={info['length']:.1f}×{info['width']:.1f}m, "
        #               f"相对速度={info['relative_speed']:.1f}km/h")
        #     else:
        #         print(f"车辆 {i+1}: 无效/不存在")
        
        assert env.observation_space.contains(obs)
        _act(env, env.action_space.sample())
        for x in [-1, 0, 1]:
            env.reset()
            for y in [-1, 0, 1]:
                _act(env, [x, y])
    finally:
        env.close()