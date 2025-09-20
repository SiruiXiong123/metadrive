from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

class MetaDriveMetricsCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_distances = []
        self.episode_collisions = []
        self.episode_success_flags = []
        self.episode_costs = []

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        infos = self.locals.get("infos", [])

        for info in infos:
            if not isinstance(info, dict):
                continue

            ep = info.get("episode")
            if isinstance(ep, dict):
                self.episode_rewards.append(ep.get("r", 0))

            # 成功率（来自 MetaDrive 的 done_info）
            is_success = info.get("is_success", info.get("arrive_dest", 0))
            self.episode_success_flags.append(int(bool(is_success)))

            # 路径完成度 → 距离估算
            route_completion = info.get("route_completion", 0)
            self.episode_distances.append(route_completion * 1000)

            # 成本（MetaDrive cost_function）
            cost = info.get("cost", 0)
            self.episode_costs.append(cost)

            # 碰撞次数估算（总和）
            collisions = sum([
                info.get("crash_vehicle", 0),
                info.get("crash_object", 0),
                info.get("crash_building", 0),
                info.get("crash_human", 0),
                info.get("crash_sidewalk", 0),
            ])
            self.episode_collisions.append(collisions)

        if self.episode_rewards:
            self.logger.record("custom/avg_reward", np.mean(self.episode_rewards))
            self.logger.record("custom/avg_distance_traveled", np.mean(self.episode_distances))
            self.logger.record("custom/avg_collisions", np.mean(self.episode_collisions))
            self.logger.record("custom/success_rate", np.mean(self.episode_success_flags))
            self.logger.record("custom/avg_cost", np.mean(self.episode_costs))

            # 清空缓存
            self.episode_rewards.clear()
            self.episode_distances.clear()
            self.episode_collisions.clear()
            self.episode_success_flags.clear()
            self.episode_costs.clear()
