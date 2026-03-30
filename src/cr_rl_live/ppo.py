from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


@dataclass
class PPOConfig:
    state_size: int = 13
    action_size: int = 13
    hidden_size: int = 128
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    epochs: int = 4
    batch_size: int = 256
    device: str = "cpu"


class PolicyValueNet(nn.Module):
    def __init__(self, state_size: int, action_size: int, hidden_size: int) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.policy = nn.Linear(hidden_size, action_size)
        self.value = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(x)
        logits = self.policy(h)
        value = self.value(h).squeeze(-1)
        return logits, value

    def act(self, state: np.ndarray, no_op_bias: float = 0.0) -> Dict[str, float]:
        x = torch.from_numpy(state).float().unsqueeze(0)
        logits, value = self.forward(x)
        if no_op_bias != 0.0:
            logits = logits.clone()
            logits[:, 0] += no_op_bias
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return {
            "action": int(action.item()),
            "log_prob": float(log_prob.item()),
            "value": float(value.item()),
        }


def load_or_create_model(checkpoint_path: str, cfg: PPOConfig) -> Tuple[PolicyValueNet, torch.optim.Optimizer]:
    net = PolicyValueNet(cfg.state_size, cfg.action_size, cfg.hidden_size).to(cfg.device)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    path = Path(checkpoint_path)
    if path.exists():
        data = torch.load(path, map_location=cfg.device)
        net.load_state_dict(data["model"])
        opt.load_state_dict(data["optimizer"])
    return net, opt


def save_model(checkpoint_path: str, model: PolicyValueNet, optimizer: torch.optim.Optimizer, step: int) -> None:
    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": int(step),
        },
        path,
    )


def _compute_gae(
    rewards: np.ndarray,
    dones: np.ndarray,
    values: np.ndarray,
    next_values: np.ndarray,
    gamma: float,
    lam: float,
) -> Tuple[np.ndarray, np.ndarray]:
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_adv = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_values[t] * mask - values[t]
        last_adv = delta + gamma * lam * mask * last_adv
        advantages[t] = last_adv
    returns = advantages + values
    return advantages, returns


def train_ppo_epoch(model: PolicyValueNet, optimizer, transitions: List[Dict[str, object]], cfg: PPOConfig) -> Dict[str, float]:
    states = np.stack([t["state"] for t in transitions]).astype(np.float32)
    next_states = np.stack([t["next_state"] for t in transitions]).astype(np.float32)
    actions = np.array([t["action"] for t in transitions], dtype=np.int64)
    old_log_probs = np.array([t["log_prob"] for t in transitions], dtype=np.float32)
    rewards = np.array([t["reward"] for t in transitions], dtype=np.float32)
    dones = np.array([1.0 if t["done"] else 0.0 for t in transitions], dtype=np.float32)

    with torch.no_grad():
        _, values_t = model(torch.from_numpy(states).to(cfg.device))
        _, next_values_t = model(torch.from_numpy(next_states).to(cfg.device))
        values = values_t.cpu().numpy()
        next_values = next_values_t.cpu().numpy()

    advantages, returns = _compute_gae(rewards, dones, values, next_values, cfg.gamma, cfg.gae_lambda)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    states_t = torch.from_numpy(states).to(cfg.device)
    actions_t = torch.from_numpy(actions).to(cfg.device)
    old_log_probs_t = torch.from_numpy(old_log_probs).to(cfg.device)
    advantages_t = torch.from_numpy(advantages).to(cfg.device)
    returns_t = torch.from_numpy(returns).to(cfg.device)

    n = states.shape[0]
    idx = np.arange(n)
    loss_pi_sum = 0.0
    loss_v_sum = 0.0
    entropy_sum = 0.0
    step_count = 0

    for _ in range(cfg.epochs):
        np.random.shuffle(idx)
        for start in range(0, n, cfg.batch_size):
            batch = idx[start:start + cfg.batch_size]
            b_s = states_t[batch]
            b_a = actions_t[batch]
            b_old_lp = old_log_probs_t[batch]
            b_adv = advantages_t[batch]
            b_ret = returns_t[batch]

            logits, values_pred = model(b_s)
            dist = Categorical(logits=logits)
            log_probs = dist.log_prob(b_a)
            ratio = torch.exp(log_probs - b_old_lp)
            s1 = ratio * b_adv
            s2 = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * b_adv
            loss_pi = -torch.min(s1, s2).mean()
            loss_v = F.mse_loss(values_pred, b_ret)
            entropy = dist.entropy().mean()

            loss = loss_pi + cfg.value_coef * loss_v - cfg.entropy_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_pi_sum += float(loss_pi.item())
            loss_v_sum += float(loss_v.item())
            entropy_sum += float(entropy.item())
            step_count += 1

    return {
        "loss_policy": loss_pi_sum / max(1, step_count),
        "loss_value": loss_v_sum / max(1, step_count),
        "entropy": entropy_sum / max(1, step_count),
        "transitions": float(n),
    }

