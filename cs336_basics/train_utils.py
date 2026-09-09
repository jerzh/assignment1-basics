from __future__ import annotations

import os
import typing

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn


# 5.1 Data loading
def get_batch(
    x: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample `batch_size` random windows of length `context_length` from `x`
    and return (inputs, targets), each of shape (batch_size, context_length),
    on the requested `device`. `targets[b, i] = x[start_b + i + 1]`.
    """
    x_ = torch.from_numpy(x).to(device)
    start_indices = torch.randint(
        high=len(x) - context_length,
        size=(batch_size, 1),
        device=device,
    )
    inputs = x_[start_indices + torch.arange(0, context_length)]
    targets = x_[start_indices + 1 + torch.arange(0, context_length)]
    return (inputs, targets)


# 5.2 Checkpointing
def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
) -> None:
    obj = {
        "model": model.state_dict(),
        "optim": optimizer.state_dict(),
        "iter":  iteration,
    }
    torch.save(obj, out)


def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    obj = torch.load(src)
    model.load_state_dict(obj["model"])
    optimizer.load_state_dict(obj["optim"])
    return obj["iter"]
