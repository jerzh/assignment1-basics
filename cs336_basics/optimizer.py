from __future__ import annotations

from typing import Iterable

import math
import torch
import torch.nn as nn
from torch.optim import Optimizer


def cross_entropy(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    inputs = inputs - inputs.amax(-1, keepdim=True)
    log_norm = inputs.exp().sum(dim=-1).log()
    correct_logits = inputs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return (log_norm - correct_logits).mean()


class AdamW(Optimizer):
    def __init__(self, params: Iterable[torch.Tensor], lr: float, betas: tuple[float, float],
                 eps: float, weight_decay: float):
        defaults = {
            "lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

    def step(self, closure=None):
        for group in self.param_groups:
            lr = group["lr"]
            betas = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            p: nn.Parameter
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)
                # main update
                state["step"] += 1
                adj_lr = lr * math.sqrt(1 - betas[1]**state["step"]) / (1 - betas[0]**state["step"])
                p.data.sub_(p.data, alpha=lr * wd)
                state["m"].mul_(betas[0]).add_(grad, alpha=1-betas[0])
                state["v"].mul_(betas[1]).addcmul_(grad, grad, value=1-betas[1])
                p.data.sub_(state["m"] / (torch.sqrt(state["v"]) + eps), alpha=adj_lr)


def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    if it < warmup_iters:
        return it/warmup_iters * max_learning_rate
    elif it < cosine_cycle_iters:
        frac_complete = (it - warmup_iters)/(cosine_cycle_iters - warmup_iters)
        return min_learning_rate + 0.5 * (1 + math.cos(frac_complete * math.pi)) * (max_learning_rate - min_learning_rate)
    else:
        return min_learning_rate


def clip_gradients(parameters: Iterable[nn.Parameter], max_l2_norm: float):
    total_sq = sum((p.grad.detach() ** 2).sum() for p in parameters if p.grad is not None)
    total_norm = total_sq.sqrt()
    if total_norm > max_l2_norm:
        factor = max_l2_norm / (total_norm + 1e-6)
        for p in parameters:
            if p.grad is None:
                continue
            p.grad.mul_(factor)
