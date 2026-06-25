"""
Training entry point for the refactored CSMT-GNN prototype.

The script supports single-GPU and torchrun DistributedDataParallel runs. It
expects token ids and AST artifacts to be stored as NumPy files sharing a prefix:

    0_tokens.npy
    0_ast_ids.npy
    0_ast_mask.npy
"""

from __future__ import annotations

import argparse
import json
import os
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from csmt_gnn import CSMTConfig, CSMTModel


class CSMTDataset(Dataset):
    def __init__(self, data_dir: Path, ast_dir: Path, max_tokens: int) -> None:
        self.data_dir = data_dir
        self.ast_dir = ast_dir
        self.max_tokens = max_tokens
        self.samples = self._discover()
        if not self.samples:
            raise RuntimeError(f"No *_tokens.npy samples found in {data_dir}")

    def _discover(self) -> List[str]:
        prefixes: List[str] = []
        for token_file in sorted(self.data_dir.glob("*_tokens.npy")):
            prefix = token_file.name[: -len("_tokens.npy")]
            token_shape = np.load(token_file, mmap_mode="r").shape
            long_enough = bool(token_shape) and min(token_shape[0], self.max_tokens) >= 2
            if long_enough and self._ast_ids_path(prefix).exists() and self._ast_mask_path(prefix).exists():
                prefixes.append(prefix)
        return prefixes

    def _ast_ids_path(self, prefix: str) -> Path:
        direct = self.ast_dir / f"{prefix}_ast_ids.npy"
        legacy = self.ast_dir / f"{prefix}_ast_type_ids.npy"
        return direct if direct.exists() else legacy

    def _ast_mask_path(self, prefix: str) -> Path:
        token_level = self.ast_dir / f"{prefix}_token_ast_mask.npy"
        if token_level.exists():
            return token_level
        direct = self.ast_dir / f"{prefix}_ast_mask.npy"
        legacy = self.ast_dir / f"{prefix}_var_def_mask.npy"
        return direct if direct.exists() else legacy

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        prefix = self.samples[index]
        tokens = np.load(self.data_dir / f"{prefix}_tokens.npy", mmap_mode="r")
        ast_ids = np.load(self._ast_ids_path(prefix), mmap_mode="r")
        ast_mask = np.load(self._ast_mask_path(prefix), mmap_mode="r")

        tokens_t = torch.from_numpy(np.array(tokens[: self.max_tokens], dtype=np.int64, copy=True))
        ast_ids_t = torch.from_numpy(np.array(ast_ids, dtype=np.int64, copy=True))
        ast_mask_t = torch.from_numpy(np.array(ast_mask, dtype=np.bool_, copy=True))
        return tokens_t, ast_ids_t, ast_mask_t


def collate_batch(batch):
    max_len = max(tokens.numel() for tokens, _, _ in batch)
    max_blocks = max(ast_ids.size(0) for _, ast_ids, _ in batch)
    max_mask_width = max(ast_mask.numel() for _, _, ast_mask in batch)
    block_size = batch[0][1].size(1)

    tokens_out = torch.zeros(len(batch), max_len, dtype=torch.long)
    lengths = torch.zeros(len(batch), dtype=torch.long)
    ast_ids_out = torch.zeros(len(batch), max_blocks, block_size, dtype=torch.long)
    ast_mask_out = torch.zeros(len(batch), max_mask_width, dtype=torch.bool)

    for idx, (tokens, ast_ids, ast_mask) in enumerate(batch):
        length = tokens.numel()
        num_blocks = ast_ids.size(0)
        tokens_out[idx, :length] = tokens
        lengths[idx] = length
        ast_ids_out[idx, :num_blocks] = ast_ids
        ast_mask_out[idx, : ast_mask.numel()] = ast_mask

    return tokens_out, lengths, ast_ids_out, ast_mask_out


def setup_distributed() -> Tuple[bool, int, int, torch.device]:
    if "RANK" not in os.environ:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return False, 0, 1, device

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    return True, dist.get_rank(), dist.get_world_size(), torch.device("cuda", local_rank)


def cleanup_distributed(enabled: bool) -> None:
    if enabled and dist.is_initialized():
        dist.destroy_process_group()


def unwrap_model(model):
    while hasattr(model, "module"):
        model = model.module
    return getattr(model, "_orig_mod", model)


def load_num_ast_types(ast_dir: Path, default: int) -> int:
    vocab_path = ast_dir / "ast_vocab.json"
    if not vocab_path.exists():
        return default
    data = json.loads(vocab_path.read_text(encoding="utf-8"))
    vocab = data.get("type_vocab", {})
    return max(default, len(vocab))


def build_config(args) -> CSMTConfig:
    if args.max_tokens < 2:
        raise ValueError("--max-tokens must be at least 2 for next-token training.")
    if args.micro_batch_size <= 0:
        raise ValueError("--micro-batch-size must be positive.")
    if args.grad_accum_steps <= 0:
        raise ValueError("--grad-accum-steps must be positive.")
    num_ast_types = args.num_ast_types or load_num_ast_types(args.ast_path, 256)
    return CSMTConfig(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        hidden_size=args.hidden_size,
        block_size=args.block_size,
        max_tokens=args.max_tokens,
        num_heads=args.num_heads,
        num_graph_heads=args.num_graph_heads,
        num_experts=args.num_experts,
        moe_top_k=args.moe_top_k,
        ffn_multiplier=args.ffn_multiplier,
        kv_compression=args.kv_compression,
        num_ast_types=num_ast_types,
        ast_dim=args.ast_dim,
        ast_gate_scale=args.ast_gate_scale,
        boundary_mix=args.boundary_mix,
        boundary_width=args.boundary_width,
        cvd_prob=args.cvd_prob,
        cvd_scope=args.cvd_scope,
        dropout=args.dropout,
        tie_embeddings=args.tie_embeddings,
        use_ast_gate=not args.no_ast_gate,
        use_block_graph=not args.no_block_graph,
        use_cvd=not args.no_cvd,
        use_moe=not args.no_moe,
        use_boundary=not args.no_boundary,
    )


def optimizer_groups(model: CSMTModel, args) -> List[Dict]:
    cvd_params = []
    normal_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "cvd_replacement" in name:
            cvd_params.append(param)
        else:
            normal_params.append(param)
    groups = []
    if normal_params:
        groups.append({"params": normal_params, "lr": args.lr, "weight_decay": args.weight_decay})
    if cvd_params:
        groups.append({"params": cvd_params, "lr": args.lr * args.cvd_lr_multiplier, "weight_decay": 0.0})
    return groups


def save_checkpoint(model, optimizer, step: int, args, rank: int) -> None:
    if rank != 0:
        return
    args.save_dir.mkdir(parents=True, exist_ok=True)
    raw_model = unwrap_model(model)
    torch.save(
        {
            "step": step,
            "model": raw_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": raw_model.config.__dict__,
        },
        args.save_dir / f"checkpoint_{step:08d}.pt",
    )


def train(args) -> None:
    ddp_enabled, rank, world_size, device = setup_distributed()
    torch.backends.cuda.matmul.allow_tf32 = True
    if torch.cuda.is_available():
        torch.backends.cudnn.allow_tf32 = True

    config = build_config(args)
    model = CSMTModel(config).to(device)
    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(model)
    if ddp_enabled:
        model = DistributedDataParallel(model, device_ids=[device.index], find_unused_parameters=True)

    dataset = CSMTDataset(args.data_path, args.ast_path, args.max_tokens)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True) if ddp_enabled else None
    loader = DataLoader(
        dataset,
        batch_size=args.micro_batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        collate_fn=collate_batch,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )

    optimizer = torch.optim.AdamW(optimizer_groups(unwrap_model(model), args), betas=(0.9, 0.95), eps=1e-8)
    use_cuda_amp = torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda", enabled=args.fp16 and use_cuda_amp)
    autocast_dtype = torch.float16 if args.fp16 else torch.bfloat16
    global_step = 0

    def optimizer_step(current_loss: torch.Tensor, actual_accum_steps: int) -> None:
        nonlocal global_step
        if actual_accum_steps != args.grad_accum_steps:
            scale = args.grad_accum_steps / max(1, actual_accum_steps)
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.mul_(scale)
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        global_step += 1

        if rank == 0 and global_step % args.log_interval == 0:
            display_loss = current_loss.item() * actual_accum_steps
            print(
                f"epoch={epoch} step={global_step} "
                f"loss={display_loss:.4f} grad_norm={float(grad_norm):.3f}",
                flush=True,
            )
        if global_step % args.save_interval == 0:
            save_checkpoint(model, optimizer, global_step, args, rank)

    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        optimizer.zero_grad(set_to_none=True)
        pending_accum = 0
        last_loss = None
        for step, (tokens, lengths, ast_ids, ast_mask) in enumerate(loader):
            tokens = tokens.to(device, non_blocking=True)
            lengths = lengths.to(device, non_blocking=True)
            ast_ids = ast_ids.to(device, non_blocking=True)
            ast_mask = ast_mask.to(device, non_blocking=True)
            if int(lengths.max().item()) < 2:
                continue

            should_step = pending_accum + 1 >= args.grad_accum_steps or step + 1 == len(loader)
            sync_context = nullcontext()
            if ddp_enabled and not should_step:
                sync_context = model.no_sync()

            with sync_context:
                with torch.amp.autocast("cuda", enabled=use_cuda_amp, dtype=autocast_dtype):
                    model_input = tokens[:, :-1]
                    labels = tokens[:, 1:]
                    logits = model(
                        model_input,
                        ast_type_ids=ast_ids,
                        var_def_mask=ast_mask,
                        lengths=(lengths - 1).clamp_min(0),
                    )
                    target_len = logits.size(1)
                    labels = labels[:, :target_len]
                    valid = torch.arange(target_len, device=device).view(1, -1) < (lengths - 1).view(-1, 1)
                    token_loss = F.cross_entropy(
                        logits.float().reshape(-1, logits.size(-1)),
                        labels.reshape(-1),
                        reduction="none",
                    ).view_as(labels)
                    loss = (token_loss * valid).sum() / valid.sum().clamp_min(1)
                    raw_model = unwrap_model(model)
                    if args.cvd_reg > 0:
                        loss = loss + args.cvd_reg * raw_model.cvd_regularization()
                    loss = loss + args.moe_aux_loss_weight * raw_model.moe_auxiliary_loss()
                    loss = loss + args.router_z_loss_weight * raw_model.router_z_loss()
                    loss = loss / args.grad_accum_steps

                scaler.scale(loss).backward()
            pending_accum += 1
            last_loss = loss

            if should_step:
                optimizer_step(loss, pending_accum)
                pending_accum = 0

        if pending_accum and last_loss is not None:
            optimizer_step(last_loss, pending_accum)

    save_checkpoint(model, optimizer, global_step, args, rank)
    cleanup_distributed(ddp_enabled)


def parse_args():
    parser = argparse.ArgumentParser(description="Train CSMT-GNN.")
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--ast-path", type=Path, required=True)
    parser.add_argument("--save-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--vocab-size", type=int, default=50000)
    parser.add_argument("--num-ast-types", type=int, default=0)
    parser.add_argument("--num-layers", type=int, default=12)
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--num-heads", type=int, default=12)
    parser.add_argument("--num-graph-heads", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--num-experts", type=int, default=8)
    parser.add_argument("--moe-top-k", type=int, default=2)
    parser.add_argument("--ffn-multiplier", type=float, default=2.0)
    parser.add_argument("--kv-compression", type=float, default=0.25)
    parser.add_argument("--ast-dim", type=int, default=128)
    parser.add_argument("--ast-gate-scale", type=float, default=0.1)
    parser.add_argument("--boundary-mix", type=float, default=0.1)
    parser.add_argument("--boundary-width", type=int, default=1)
    parser.add_argument("--cvd-prob", type=float, default=0.05)
    parser.add_argument("--cvd-scope", choices=["variable", "random"], default="variable")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--tie-embeddings", action="store_true")
    parser.add_argument("--no-ast-gate", action="store_true")
    parser.add_argument("--no-block-graph", action="store_true")
    parser.add_argument("--no-cvd", action="store_true")
    parser.add_argument("--no-moe", action="store_true")
    parser.add_argument("--no-boundary", action="store_true")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--cvd-lr-multiplier", type=float, default=2.0)
    parser.add_argument("--cvd-reg", type=float, default=0.0)
    parser.add_argument("--moe-aux-loss-weight", type=float, default=1e-2)
    parser.add_argument("--router-z-loss-weight", type=float, default=1e-3)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--save-interval", type=int, default=1000)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--compile", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
