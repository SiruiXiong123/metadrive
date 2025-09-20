# ppo_cuda_rollout.py
from typing import Dict, Union
import torch as th
from gymnasium.spaces import Box
from stable_baselines3.ppo import PPO
from stable_baselines3.common.vec_env import VecEnv
from cupy_to_torch import cupy_to_torch
from torch_vec_rollout_buffer import TorchVecRolloutBuffer

class PPOTorchVec(PPO):
    def learn(self, *args, **kwargs):
        # 用我们自己的 CUDA 向量化 buffer
        self.rollout_buffer = TorchVecRolloutBuffer(self.n_steps, self.env.num_envs, self.device)
        return super().learn(*args, **kwargs)

    @th.no_grad()
    def collect_rollouts(self, env: VecEnv, callback, rollout_buffer: TorchVecRolloutBuffer, n_rollout_steps: int):
        rollout_buffer.reset()
        n_steps = 0

        if not hasattr(self, "_last_obs"):
            obs, _ = env.reset()
            self._last_obs = obs
            self._last_episode_starts = th.zeros((env.num_envs,), device=self.device, dtype=th.bool)

        while n_steps < n_rollout_steps:
            obs_t = self._obs_to_torch(self._last_obs)  # dict/ndarray -> torch.cuda

            # 前向
            features = self.policy.extract_features(obs_t)
            latent_pi, latent_vf = self.policy.mlp_extractor(features)
            dist = self.policy.get_distribution(latent_pi)
            actions = dist.sample()
            log_prob = dist.log_prob(actions)
            values = self.policy.value_net(latent_vf).flatten()

            # 动作给 env 需要 numpy
            actions_env = actions
            if isinstance(self.action_space, Box) and self.policy.squash_output:
                actions_env = th.tanh(actions_env)
            new_obs, rewards, dones, infos = env.step(actions_env.detach().cpu().numpy())

            # 写 buffer
            reward_t = th.as_tensor(rewards, device=self.device, dtype=th.float32)  # [n_envs]
            done_t   = th.as_tensor(dones,   device=self.device, dtype=th.bool)     # [n_envs]
            rollout_buffer.add(
                obs=obs_t,
                action=actions,
                reward=reward_t,
                done=done_t,
                value=values,
                log_prob=log_prob,
            )

            self._last_obs = new_obs
            self._last_episode_starts = done_t
            n_steps += 1

        # bootstrap
        last_obs_t = self._obs_to_torch(self._last_obs)
        last_feat = self.policy.extract_features(last_obs_t)
        _, last_latent_vf = self.policy.mlp_extractor(last_feat)
        last_values = self.policy.value_net(last_latent_vf).flatten()  # [n_envs]

        rollout_buffer.compute_returns_and_advantage(
            last_value=last_values, gamma=self.gamma, gae_lambda=self.gae_lambda
        )
        return True

    def _obs_to_torch(self, obs: Union[Dict[str, object], object]):
        if isinstance(obs, dict):
            return {k: cupy_to_torch(v, device=self.device, dtype=th.float32) for k, v in obs.items()}
        return cupy_to_torch(obs, device=self.device, dtype=th.float32)
