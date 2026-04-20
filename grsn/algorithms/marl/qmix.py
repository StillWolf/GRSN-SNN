"""QMIX 训练器。

Q-learning + 单调 mixer 的联合训练：
1. 对 batch 里每条 episode 前向 agent_net → Q_eval（per-agent per-action）
2. 对同一 batch 前向 target_agent_net → Q_target（next obs）
3. 选取 eval 动作 = batch['actions']；target 动作 = argmax over avail_actions
4. mixer 把 per-agent Q 组合成 joint Q_tot
5. TD 目标 y = r + γ(1-done)·target_mixer(Q_target_max, next_state)
6. loss = mean((Q_tot_eval - y.detach())^2 * filled_mask) / filled_mask.sum()
"""
from copy import deepcopy
from typing import Dict

import torch
import torch.nn as nn


class QMIX:
    def __init__(
        self,
        agent_net: nn.Module,
        mixer: nn.Module,
        lr: float = 5e-4,
        gamma: float = 0.99,
        grad_clip: float = 10.0,
        device: torch.device = torch.device("cpu"),
    ):
        self.agent_net = agent_net.to(device)
        self.mixer = mixer.to(device)
        self.target_agent_net = deepcopy(self.agent_net).to(device)
        self.target_mixer = deepcopy(self.mixer).to(device)
        for p in self.target_agent_net.parameters():
            p.requires_grad = False
        for p in self.target_mixer.parameters():
            p.requires_grad = False

        self.gamma = gamma
        self.grad_clip = grad_clip
        self.device = device

        params = list(self.agent_net.parameters()) + list(self.mixer.parameters())
        self.optimizer = torch.optim.Adam(params, lr=lr)

    def _rollout_agent(self, agent_net, obs, batch_size, n_agents):
        """对整条 episode 前向 agent_net。

        obs: (B, T, n_agents, obs_dim) → 折 n_agents 进 batch → (T, B*n_agents, obs_dim)
        返回 Q: (B, T, n_agents, n_actions)
        """
        B, T, N, D = obs.shape
        obs_flat = obs.permute(1, 0, 2, 3).reshape(T, B * N, D)
        init_state = agent_net.init_hidden(B * N, device=self.device, dtype=obs.dtype)
        q_flat, _ = agent_net(obs_flat, init_state)  # (T, B*N, n_actions)
        A = q_flat.shape[-1]
        q = q_flat.reshape(T, B, N, A).permute(1, 0, 2, 3).contiguous()
        return q  # (B, T, n_agents, n_actions)

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        batch = {k: v.to(self.device) for k, v in batch.items()}
        obs = batch["obs"]                   # (B, T, N, obs_dim)
        state = batch["state"]               # (B, T, state_dim)
        actions = batch["actions"]           # (B, T, N)
        avail = batch["avail_actions"]       # (B, T, N, n_actions)
        rewards = batch["rewards"]           # (B, T, 1)
        terminated = batch["terminated"]     # (B, T, 1)
        filled = batch["filled"]             # (B, T, 1)

        B, T, N, _ = obs.shape

        q_eval_all = self._rollout_agent(self.agent_net, obs, B, N)

        with torch.no_grad():
            q_target_all = self._rollout_agent(self.target_agent_net, obs, B, N)

        actions_unsq = actions.unsqueeze(-1)
        q_eval_chosen = q_eval_all.gather(dim=-1, index=actions_unsq).squeeze(-1)

        with torch.no_grad():
            q_target_masked = q_target_all.clone()
            q_target_masked[avail == 0] = -1e9
            q_target_max = q_target_masked.max(dim=-1).values

        q_tot_eval = self.mixer(q_eval_chosen, state).squeeze(-1)  # (B, T)
        with torch.no_grad():
            q_tot_target = self.target_mixer(q_target_max, state).squeeze(-1)

        rewards_s = rewards.squeeze(-1)
        terminated_s = terminated.squeeze(-1)
        filled_s = filled.squeeze(-1)
        y = rewards_s[:, :-1] + self.gamma * (1.0 - terminated_s[:, :-1]) * q_tot_target[:, 1:]
        td = q_tot_eval[:, :-1] - y.detach()

        mask = filled_s[:, :-1]
        num_valid = mask.sum().clamp(min=1.0)
        loss = ((td ** 2) * mask).sum() / num_valid

        self.optimizer.zero_grad()
        loss.backward()
        params = list(self.agent_net.parameters()) + list(self.mixer.parameters())
        grad_norm = torch.nn.utils.clip_grad_norm_(params, self.grad_clip)
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "grad_norm": float(grad_norm),
            "q_tot_mean": q_tot_eval[:, :-1].mul(mask).sum().div(num_valid).item(),
            "y_mean": y.mul(mask).sum().div(num_valid).item(),
        }

    def target_update(self) -> None:
        """硬复制 eval → target。"""
        self.target_agent_net.load_state_dict(self.agent_net.state_dict())
        self.target_mixer.load_state_dict(self.mixer.state_dict())
