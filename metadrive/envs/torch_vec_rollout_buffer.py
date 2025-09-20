# torch_vec_rollout_buffer.py
from typing import Dict, Generator, Optional, Tuple, Union
import torch as th
from stable_baselines3.common.buffers import RolloutBufferSamples

class TorchVecRolloutBuffer:
    """
    与 SB3 的 RolloutBuffer 接口对齐，但全部用 CUDA 张量，维度 [n_steps, n_envs, ...]。
    - obs 支持 Dict[str, Tensor] 或 Tensor
    - get(batch_size) 产出 RolloutBufferSamples（展平后并随机打乱）
    """
    def __init__(self, n_steps: int, n_envs: int, device: th.device):
        self.n_steps, self.n_envs, self.device = n_steps, n_envs, device
        self.reset()

    def reset(self) -> None:
        self.pos = 0
        self._obs_is_dict: Optional[bool] = None
        self._obs_list = []  # list of obs per step (each is [n_envs, ...])
        self.actions = None
        self.rewards = th.zeros(self.n_steps, self.n_envs, device=self.device)
        self.dones   = th.zeros(self.n_steps, self.n_envs, dtype=th.bool, device=self.device)
        self.values  = th.zeros(self.n_steps, self.n_envs, device=self.device)
        self.log_probs = th.zeros(self.n_steps, self.n_envs, device=self.device)
        self.advantages = th.zeros(self.n_steps, self.n_envs, device=self.device)
        self.returns    = th.zeros(self.n_steps, self.n_envs, device=self.device)

    def add(
        self,
        obs: Union[Dict[str, th.Tensor], th.Tensor],
        action: th.Tensor,               # [n_envs, ...]
        reward: th.Tensor,               # [n_envs]
        done: th.Tensor,                 # [n_envs] bool
        value: th.Tensor,                # [n_envs]
        log_prob: th.Tensor,             # [n_envs]
    ) -> None:
        if self._obs_is_dict is None:
            self._obs_is_dict = isinstance(obs, dict)
        self._obs_list.append(obs)
        if self.actions is None:
            self.actions = action.new_zeros((self.n_steps, *action.shape))
        self.actions[self.pos]  = action
        self.rewards[self.pos]  = reward
        self.dones[self.pos]    = done
        self.values[self.pos]   = value
        self.log_probs[self.pos]= log_prob
        self.pos += 1

    @th.no_grad()
    def compute_returns_and_advantage(self, last_value: th.Tensor, gamma: float, gae_lambda: float) -> None:
        # last_value: [n_envs]
        last_gae = th.zeros(self.n_envs, device=self.device)
        for t in reversed(range(self.n_steps)):
            next_values = last_value if t == self.n_steps - 1 else self.values[t + 1]
            next_non_terminal = (~self.dones[t]).float()
            delta = self.rewards[t] + gamma * next_values * next_non_terminal - self.values[t]
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            self.advantages[t] = last_gae
        self.returns = self.advantages + self.values

    def _stack_obs(self) -> Union[Dict[str, th.Tensor], th.Tensor]:
        # [list(n_steps)] -> [n_steps, n_envs, ...]
        if self._obs_is_dict:
            keys = self._obs_list[0].keys()
            out: Dict[str, th.Tensor] = {}
            for k in keys:
                out[k] = th.stack([o[k] for o in self._obs_list], dim=0)
            return out
        else:
            return th.stack(self._obs_list, dim=0)

    def get(self, batch_size: int) -> Generator[RolloutBufferSamples, None, None]:
        # 展平为 [n_steps*n_envs, ...] 并随机打乱，按 batch_size 产出
        obs = self._stack_obs()
        if self._obs_is_dict:
            obs = {k: v.reshape(-1, *v.shape[2:]) for k, v in obs.items()}
        else:
            obs = obs.reshape(-1, *obs.shape[2:])
        actions    = self.actions.reshape(-1, *self.actions.shape[2:])
        values     = self.values.reshape(-1)
        log_probs  = self.log_probs.reshape(-1)
        advantages = self.advantages.reshape(-1)
        returns    = self.returns.reshape(-1)

        n = returns.shape[0]
        idx = th.randperm(n, device=self.device)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            mb = idx[start:end]
            mb_obs = {k: v[mb] for k, v in obs.items()} if isinstance(obs, dict) else obs[mb]
            yield RolloutBufferSamples(
                observations=mb_obs,
                actions=actions[mb],
                old_values=values[mb],
                old_log_prob=log_probs[mb],
                advantages=advantages[mb],
                returns=returns[mb],
            )
