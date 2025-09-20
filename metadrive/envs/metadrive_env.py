import copy
import math
from metadrive.component.navigation_module.node_network_navigation import NodeNetworkNavigation
from typing import Union

import numpy as np

from metadrive.component.algorithm.blocks_prob_dist import PGBlockDistConfig
from metadrive.component.map.base_map import BaseMap
from metadrive.component.map.pg_map import parse_map_config, MapGenerateMethod
from metadrive.component.pgblock.first_block import FirstPGBlock
from metadrive.constants import DEFAULT_AGENT, TerminationState
from metadrive.envs.base_env import BaseEnv
from metadrive.manager.traffic_manager import TrafficMode
from metadrive.utils import clip, Config

from metadrive.component.sensors.rgb_camera import RGBCamera


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
    success_reward=50.0,
    out_of_road_penalty=20.0,
    crash_vehicle_penalty=20.0,
    crash_object_penalty=20.0,
    crash_sidewalk_penalty=20.0,
    driving_reward=2.0,
    speed_reward=1.0,
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
        self._compute_navi_dist = False

        # scenario setting
        self.start_seed = self.start_index = self.config["start_seed"]
        self.env_num = self.num_scenarios

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

        # =======================（以下为你原来粘贴的逻辑，已失效；保留但整体注释掉以免混淆）=======================
        # vehicle = self.agents[vehicle_id]
        # done = False
        # max_step = self.config["horizon"] is not None and self.episode_lengths[vehicle_id] >= self.config["horizon"]
        # done_info = {...}
        # return done, done_info
        # ======================= 结束 =======================

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
        long, lat = vehicle.navigation.final_lane.local_coordinates(vehicle.position)
        flag = (vehicle.navigation.final_lane.length - 5 < long < vehicle.navigation.final_lane.length + 5) and (
            vehicle.navigation.get_current_lane_width() / 2 >= lat >=
            (0.5 - vehicle.navigation.get_current_lane_num()) * vehicle.navigation.get_current_lane_width()
        )
        return flag

    def _is_out_of_road(self, vehicle):
        ret = not vehicle.on_lane
        if self.config["out_of_route_done"]:
            ret = ret or vehicle.out_of_route
        elif self.config["on_continuous_line_done"]:
            ret = ret or vehicle.on_yellow_continuous_line or vehicle.on_white_continuous_line or vehicle.crash_sidewalk
        if self.config["on_broken_line_done"]:
            ret = ret or vehicle.on_broken_line
        return ret

    # def reward_function(self, vehicle_id: str):
        vehicle = self.agents[vehicle_id]
        step_info = dict()
        reward = 0.0

        if vehicle.lane in vehicle.navigation.current_ref_lanes:
            current_lane = vehicle.lane
            positive_road = 1
        else:
            current_lane = vehicle.navigation.current_ref_lanes[0]
            current_road = vehicle.navigation.current_road
            positive_road = 1 if not current_road.is_negative_road() else -1
        long_last, _ = current_lane.local_coordinates(vehicle.last_position)
        long_now, lateral_now = current_lane.local_coordinates(vehicle.position)

        vehicle.dist_to_left_side = abs(vehicle.dist_to_left_side)
        vehicle.dist_to_right_side = abs(vehicle.dist_to_right_side)
        # ==========out of road 风险
        if vehicle.dist_to_left_side-0.5*vehicle.WIDTH < 0.5:
            out_of_risk_l = 0.1/(((vehicle.dist_to_left_side-0.4*vehicle.WIDTH)/(0.5))**2+1)
        else:
            out_of_risk_l = 0.0
        if vehicle.dist_to_right_side-0.5*vehicle.WIDTH < 0.5:
            out_of_risk_r = 0.1/(((vehicle.dist_to_right_side-0.4*vehicle.WIDTH)/(0.5))**2+1)
        else:
            out_of_risk_r = 0.0
        reward -= 20 * (out_of_risk_l + out_of_risk_r)

        #靠近checkpoint奖励
        ckpt, _ = vehicle.navigation.get_checkpoints()   # 最近的 checkpoint
        dist = np.linalg.norm(vehicle.position - ckpt)   # 与 checkpoint 的欧式距离
        reward += 0.5 * np.exp(-dist)

        current_reference_lane = vehicle.lane
        heading_diff = vehicle.heading_diff(current_reference_lane)
        heading_factor = (1 - math.exp(-10 * (1 - heading_diff)))  # 未使用，但保留你的计算
        lateral_factor = clip(1 - 2 * abs(lateral_now) / vehicle.navigation.get_current_lane_width(), 0.0, 1.0)
        speed = (vehicle.speed_km_h / vehicle.max_speed_km_h)
        progress = long_now - long_last
        reward += self.config["driving_reward"] * progress * positive_road
        reward += self.config["speed_reward"] * speed

        #鼓励车辆朝ckpt




        # 在完成度奖励
        c_now = float(vehicle.navigation.route_completion)
        c_last = float(vehicle.navigation.last_route_completion)
        dc = max(0.0, c_now - c_last)
        reward += dc

        vehicle.navigation.last_route_completion = c_now

        step_info["step_reward"] = reward
        

        if self._is_arrive_destination(vehicle):
            reward = +self.config["success_reward"]
            print("success")
        elif self._is_out_of_road(vehicle):
            reward = -self.config["out_of_road_penalty"]
            print("out of road")
        elif vehicle.crash_vehicle:
            reward = -self.config["crash_vehicle_penalty"]
            print("crash vehicle")
        elif vehicle.crash_object:
            reward = -self.config["crash_object_penalty"]
            print("crash object")
        elif vehicle.crash_sidewalk:
            reward = -self.config["crash_sidewalk_penalty"]
            print("crash sidewalk")
        step_info["route_completion"] = vehicle.navigation.route_completion
        
        print(f"[Step Reward Breakdown] "
          f"out_risk={out_risk_penalty:.3f}, "
          f"ckpt={ckpt_reward:.3f}, "
          f"progress={progress_reward:.3f}, "
          f"speed={speed_reward:.3f}, "
          f"route={route_reward:.3f}, "
          f"TOTAL={reward:.3f}")

        return reward, step_info
    def reward_function(self, vehicle_id: str):
        vehicle = self.agents[vehicle_id]
        step_info = dict()
        reward = 0.0

        # # ===== 1. out-of-road 风险 =====
        # if vehicle.dist_to_left_side-0.5*vehicle.WIDTH < 0.5:
        #     out_of_risk_l = 0.1/(((vehicle.dist_to_left_side-0.4*vehicle.WIDTH)/(0.5))**2+1)
        # else:
        #     out_of_risk_l = 0.0
        # if vehicle.dist_to_right_side-0.5*vehicle.WIDTH < 0.5:
        #     out_of_risk_r = 0.1/(((vehicle.dist_to_right_side-0.4*vehicle.WIDTH)/(0.5))**2+1)
        # else:
        #     out_of_risk_r = 0.0
        # out_risk_penalty = -20 * (out_of_risk_l + out_of_risk_r)
        # reward += out_risk_penalty
        # step_info["out_risk_penalty"] = out_risk_penalty

        # # ===== 2. checkpoint 距离奖励 =====
        # ckpt, _ = vehicle.navigation.get_checkpoints()
        # dist = np.linalg.norm(vehicle.position - ckpt)
        # ckpt_reward = 0.5 * np.exp(-dist)
        # reward += ckpt_reward
        # step_info["ckpt_reward"] = ckpt_reward

        # ===== 3. progress + speed =====
        current_lane = vehicle.lane
        long_last, _ = current_lane.local_coordinates(vehicle.last_position)
        long_now, lateral_now = current_lane.local_coordinates(vehicle.position)
        positive_road = 1
        progress = long_now - long_last
        progress_reward = self.config["driving_reward"] * progress * positive_road
        speed_reward = self.config["speed_reward"] * (vehicle.speed_km_h / vehicle.max_speed_km_h)
        reward += progress_reward + speed_reward
        reward -= 0.01
        step_info["progress_reward"] = progress_reward
        step_info["speed_reward"] = speed_reward
        # print(f"reward{reward}")

        # # ===== 4. route completion =====
        # c_now = float(vehicle.navigation.route_completion)
        # c_last = float(vehicle.navigation.last_route_completion)
        # dc = max(0.0, c_now - c_last)
        # route_reward = dc
        # reward += route_reward
        # step_info["route_reward"] = route_reward
        # vehicle.navigation.last_route_completion = c_now
        # reward -= 0.05  # time penalty

        # ===== 5. 终止条件强奖励/惩罚 =====
        if self._is_arrive_destination(vehicle):
            reward = +self.config["success_reward"]
            step_info["terminal_reason"] = "success"
        elif self._is_out_of_road(vehicle):
            reward = -self.config["out_of_road_penalty"]
            step_info["terminal_reason"] = "out_of_road"
        elif vehicle.crash_vehicle:
            reward = -self.config["crash_vehicle_penalty"]
            step_info["terminal_reason"] = "crash_vehicle"
        elif vehicle.crash_object:
            reward = -self.config["crash_object_penalty"]
            step_info["terminal_reason"] = "crash_object"
        elif vehicle.crash_sidewalk:
            reward = -self.config["crash_sidewalk_penalty"]
            step_info["terminal_reason"] = "crash_sidewalk"

        step_info["total_reward"] = reward

        # === 打印组成部分 ===
        # print(f"[Step Reward Breakdown] "
        #     f"out_risk={out_risk_penalty:.3f}, "
        #     f"ckpt={ckpt_reward:.3f}, "
        #     f"progress={progress_reward:.3f}, "
        #     f"speed={speed_reward:.3f}, "
        #     f"route={route_reward:.3f}, "
        #     f"time_penalty={-0.05:.3f},"
        #     f"TOTAL={reward:.3f}")

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


# =========================
# 离散 vs 多维离散 对比 DEMO
# =========================
def demo(use_multi_discrete: bool):
    """在单离散 vs 多维离散下，对比 action_space 与动作映射结果"""
    sensor_size = (200, 100)
    cfg = dict(
        num_scenarios=1,
        start_seed=1000,
        use_render=False,
        image_observation=False,
        sensors={"rgb_camera": (RGBCamera, *sensor_size)},
        stack_size=4,

        # === 关键：切换离散动作空间类型 ===
        discrete_action=True,
        use_multi_discrete=use_multi_discrete,
        discrete_steering_dim=3,   # 左/中/右 -> 3 档
        discrete_throttle_dim=2,   # 刹车/前进 -> 2 档

        action_check=True,         # 打开动作校验，便于排错
    )

    env = MetaDriveEnv(cfg)
    try:
        obs, _ = env.reset()
        print("\n=== use_multi_discrete =", use_multi_discrete, "===")
        print("Action space:", env.action_space)

        # 演示：采样一个动作，看看两种空间下的差别
        sampled = env.action_space.sample()
        print("Sampled action from action_space:", sampled)

        # 两个语义动作： (左, 刹车) 与 (右, 前进)
        pairs = [(0, 0), (2, 1)]  # (steer_idx, throttle_idx)

        for steer_i, thr_i in pairs:
            if use_multi_discrete:
                # MultiDiscrete([steer_dim, throttle_dim]) -> 向量形式
                action = np.array([steer_i, thr_i], dtype=np.int64)
            else:
                # Discrete(steer_dim * throttle_dim) -> 单索引
                # 组合索引公式：index = steer + throttle * steer_dim
                action = int(steer_i + thr_i * 3)

            print(f"\nRequested (steer_idx={steer_i}, throttle_idx={thr_i})")
            print("-> sending action:", action)
            obs, rew, term, trunc, info = env.step(action)

            # 读取策略映射后的连续控制（范围约为 [-1, 1]）
            pi = env.engine.get_policy(env.agent.id)
            print("Converted continuous action [steer, throttle]:",
                  pi.action_info.get("action"))
    finally:
        env.close()


if __name__ == '__main__':
    # 对比演示：先单离散 Discrete，再多维离散 MultiDiscrete
    demo(use_multi_discrete=False)  # 单离散：Discrete(3*2=6)
    demo(use_multi_discrete=True)   # 多维离散：MultiDiscrete([3, 2])

    # ====== 下面是你原先注释掉的代码，保持不变 ======
    # from metadrive.component.sensors.rgb_camera import RGBCamera
    # from metadrive.envs.metadrive_env import MetaDriveEnv
    # from metadrive.constants import DEFAULT_AGENT
    # import numpy as np
    #
    # sensor_size = (200, 100)
    #
    # cfg = dict(
    #     num_scenarios=1,
    #     start_seed=1000,
    #     random_lane_width=True,
    #     random_lane_num=True,
    #     use_render=False,
    #     traffic_density=0.0,
    #     image_observation=True,
    #     vehicle_config=dict(image_source="rgb_camera"),
    #     sensors={"rgb_camera": (RGBCamera, *sensor_size)},
    #     stack_size=4,
    # )
    #
    # env = MetaDriveEnv(cfg)
    #
    # def _act(env, action):
    #     assert env.action_space.contains(action)
    #     obs, reward, terminated, truncated, info = env.step(action)
    #     assert env.observation_space.contains(obs)
    #     assert np.isscalar(reward)
    #     assert isinstance(info, dict)
    #
    # print("Environment Configuration Keys:")
    # print(env.config.keys())
    #
    # try:
    #     # Reset环境并获取观察数据
    #     obs, _ = env.reset()
    #     print(env.action_space)
    #
    #     # 打印观察数据的类型和形状
    #     if isinstance(obs, dict):
    #         for key, value in obs.items():
    #             print(f"Key: {key}, Type: {type(value)}, Shape: {value.shape if hasattr(value, 'shape') else 'N/A'}")
    #     else:
    #         print(f"Type: {type(obs)}, Shape: {obs.shape if hasattr(obs, 'shape') else 'N/A'}")
    #
    #     # _act(env, env.action_space.sample())
    #     # for x in [-1, 0, 1]:
    #     #     env.reset()
    #     #     for y in [-1, 0, 1]:
    #     #         _act(env, [x, y])
    #
    #     # vehicle = env.agents.get(DEFAULT_AGENT, None)
    #     # if vehicle:
    #     #     print(f"Vehicle Speed (km/h): {vehicle.speed_km_h}")
    #     #     print(dir(vehicle.navigation))
    #     #     print(vehicle.max_speed_km_h)
    #     # env = create_env(True); obs,_ = env.reset()
    #     # print(type(obs), getattr(obs,'keys',lambda:None)(),
    #     #       obs['image'].shape if isinstance(obs,dict) and 'image' in obs else getattr(obs,'shape',None))
    #
    # finally:
    #     env.close()
