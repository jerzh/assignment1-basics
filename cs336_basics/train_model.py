from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path

import numpy as np
import torch
import wandb

from cs336_basics.optimizer import AdamW, cross_entropy, get_lr_cosine_schedule, clip_gradients
from cs336_basics.train_utils import get_batch, save_checkpoint, load_checkpoint
from cs336_basics.transformer import TransformerLM


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a Transformer LM.")

    # ---- data ----
    p.add_argument("--train-data", type=str, required=True,
                   help="Path to tokenized training data (.npy of uint16/uint32 token ids).")
    p.add_argument("--val-data", type=str, required=True,
                   help="Path to tokenized validation data.")

    # ---- model architecture ----
    p.add_argument("--vocab-size", type=int, default=10_000)
    p.add_argument("--context-length", type=int, default=256)
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--num-heads", type=int, default=16)
    p.add_argument("--d-ff", type=int, default=1344)   # ~ (8/3) * d_model, rounded to multiple of 64
    p.add_argument("--rope-theta", type=float, default=10000.0)

    # ---- optimizer (AdamW) ----
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)

    # ---- LR schedule (cosine w/ warmup) ----
    p.add_argument("--lr-min", type=float, default=3e-5)
    p.add_argument("--warmup-iters", type=int, default=200)
    p.add_argument("--cosine-iters", type=int, default=5000,
                   help="Iteration at which LR reaches lr_min; typically = total training iters.")

    # ---- training loop ----
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--total-iters", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", type=str, default="float32")

    # ---- eval / logging / checkpointing ----
    p.add_argument("--eval-interval", type=int, default=250)
    p.add_argument("--eval-iters", type=int, default=50,
                   help="Number of val batches per eval.")
    p.add_argument("--log-interval", type=int, default=10)
    p.add_argument("--checkpoint-interval", type=int, default=1000)
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    p.add_argument("--resume-from", type=str, default=None,
                   help="Path to checkpoint to resume training from.")
    p.add_argument("--wandb-project", type=str, default="cs336-assignment-1")
    p.add_argument("--run-name", type=str, default=None)

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.dtype == "float32":
        dtype = torch.float32
    else:
        raise ValueError("Only f32 supported")

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    torch.manual_seed(args.seed)
    if args.wandb_project is not None:
        wandb.init(project=args.wandb_project, name=args.run_name, config=vars(args))
        run_name = wandb.run.name
    elif args.run_name is None:
        run_name = f"{time.strftime('%Y%m%d-%H%M%S')}"

    train_dataset = np.memmap(args.train_data, dtype=np.uint16, mode="r")
    val_dataset = np.memmap(args.val_data, dtype=np.uint16, mode="r")
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=args.device,
        dtype=dtype,
    )
    optimizer = AdamW(
        params=model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )
    start_iter = 0
    if args.resume_from is not None:
        start_iter = load_checkpoint(args.resume_from, model, optimizer)

    for i in range(start_iter + 1, args.total_iters + 1):
        optimizer.zero_grad()
        inputs, targets = get_batch(train_dataset, args.batch_size, args.context_length, args.device)
        logits = model.forward(inputs)
        loss = cross_entropy(logits, targets)
        loss.backward()
        clip_gradients(model.parameters(), args.grad_clip)
        this_lr = get_lr_cosine_schedule(i, args.lr, args.lr_min, args.warmup_iters, args.cosine_iters)
        for group in optimizer.param_groups:
            group["lr"] = this_lr
        optimizer.step()

        if i % args.log_interval == 0:
            logging.info(f"iter: {i}  train/loss: {loss.item()}")
            if args.wandb_project is not None:
                wandb.log({"train/loss": loss.item(), "lr": this_lr}, step=i)

        if i % args.eval_interval == 0:
            model.eval()
            losses = []
            with torch.no_grad():
                for j in range(args.eval_iters):
                    inputs, targets = get_batch(val_dataset, args.batch_size, args.context_length, args.device)
                    logits = model.forward(inputs)
                    losses.append(cross_entropy(logits, targets).item())
            val_loss = sum(losses) / len(losses)
            logging.info(f"iter: {i}  val/loss: {val_loss}")
            if args.wandb_project is not None:
                wandb.log({"val/loss": val_loss}, step=i)
            model.train()

        if i % args.checkpoint_interval == 0:
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            save_checkpoint(model, optimizer, i, Path(args.checkpoint_dir) / f"{run_name}_iter{i}")

    if args.wandb_project is not None:
        wandb.finish()
