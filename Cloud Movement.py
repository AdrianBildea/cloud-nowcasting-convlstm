#!/usr/bin/env python
import os
import time
import json
import math
import random
import argparse
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_device(force_cuda: bool = False):
    if force_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    return device


def startup_report(device: torch.device, cfg):
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM_GB: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f}")
        print(f"CUDA: {torch.version.cuda}")
    print(f"AMP: {cfg.amp and device.type == 'cuda'}")
    print(
        f"Config: filters={cfg.filters}, layers={cfg.num_layers}, seq_len={cfg.seq_len}, "
        f"batch_size={cfg.batch_size}, epochs={cfg.epochs}"
    )


@dataclass
class Config:
    data_dir: str = "./BWimages"
    results_dir: str = "./Results"
    img_h: int = 128
    img_w: int = 128
    img_c: int = 3
    seq_len: int = 8
    stride: int = 1
    filters: int = 96
    ksize: int = 3
    dropout: float = 0.10
    num_layers: int = 2
    decoder_blocks: int = 3
    epochs: int = 40
    batch_size: int = 8
    lr: float = 3e-4
    min_lr: float = 1e-6
    weight_decay: float = 1e-4
    patience: int = 8
    num_workers: int = 0
    grad_clip: float = 1.0
    seed: int = 42
    amp: bool = True
    force_cuda: bool = False
    resume: str = ""


def valid_image_files(data_dir: str):
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"Data directory not found: {root.resolve()}")
    files = sorted([p for p in root.iterdir() if p.is_file() and p.suffix.lower() in exts])
    if not files:
        raise ValueError(f"No image files found in {root.resolve()}")
    return files


def load_frames(cfg: Config):
    files = valid_image_files(cfg.data_dir)
    frames = []
    bad_files = []
    for path in files:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            bad_files.append(path.name)
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (cfg.img_w, cfg.img_h), interpolation=cv2.INTER_AREA)
        frames.append(img.astype(np.float32) / 255.0)
    if not frames:
        raise ValueError("No readable images were loaded.")
    frames = np.stack(frames).astype(np.float32)
    return frames, bad_files


def build_pairs(frames: np.ndarray, seq_len: int, stride: int):
    if len(frames) < seq_len + 1:
        raise ValueError(f"Need at least {seq_len + 1} frames, got {len(frames)}.")
    X, Y = [], []
    for i in range(0, len(frames) - seq_len, stride):
        X.append(frames[i:i + seq_len])
        Y.append(frames[i + 1:i + seq_len + 1])
    return np.stack(X), np.stack(Y)


def split_pairs(X, Y, train_ratio=0.7, val_ratio=0.15):
    n = len(X)
    if n < 6:
        raise ValueError(f"Too few sequence pairs: {n}")
    n_train = max(2, int(n * train_ratio))
    n_val = max(2, int(n * val_ratio))
    n_test = n - n_train - n_val
    if n_test < 2:
        n_test = 2
        n_train = n - n_val - n_test
    i1 = n_train
    i2 = n_train + n_val
    return {
        "train": (X[:i1], Y[:i1]),
        "val": (X[i1:i2], Y[i1:i2]),
        "test": (X[i2:], Y[i2:]),
    }


class SequenceDataset(Dataset):
    def __init__(self, X, Y, augment=False):
        self.X = torch.from_numpy(X.transpose(0, 1, 4, 2, 3)).float()
        self.Y = torch.from_numpy(Y.transpose(0, 1, 4, 2, 3)).float()
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.Y[idx]
        if self.augment:
            if torch.rand(1).item() > 0.5:
                x, y = x.flip(-1), y.flip(-1)
            if torch.rand(1).item() > 0.5:
                x, y = x.flip(-2), y.flip(-2)
        return x, y


class ConvLSTMCell(nn.Module):
    def __init__(self, in_ch, hid_ch, ksize=3):
        super().__init__()
        self.hid_ch = hid_ch
        self.conv = nn.Conv2d(in_ch + hid_ch, 4 * hid_ch, ksize, padding=ksize // 2)

    def forward(self, x, h, c):
        gates = self.conv(torch.cat([x, h], dim=1))
        i, f, g, o = gates.chunk(4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        c = f * c + i * g
        h = o * torch.tanh(c)
        return h, c

    def init_hidden(self, batch_size, height, width, device, dtype):
        h = torch.zeros(batch_size, self.hid_ch, height, width, device=device, dtype=dtype)
        c = torch.zeros(batch_size, self.hid_ch, height, width, device=device, dtype=dtype)
        return h, c


class StackedConvLSTM(nn.Module):
    def __init__(self, in_ch, hid_ch, n_layers, ksize=3, drop=0.1):
        super().__init__()
        self.cells = nn.ModuleList([
            ConvLSTMCell(in_ch if i == 0 else hid_ch, hid_ch, ksize)
            for i in range(n_layers)
        ])
        groups = 8 if hid_ch % 8 == 0 else 1
        self.norms = nn.ModuleList([nn.GroupNorm(groups, hid_ch) for _ in range(n_layers)])
        self.dropout = nn.Dropout2d(drop)

    def forward(self, x):
        B, T, _, H, W = x.shape
        states = [cell.init_hidden(B, H, W, x.device, x.dtype) for cell in self.cells]
        outputs = []
        for t in range(T):
            inp = x[:, t]
            for layer_idx, (cell, norm) in enumerate(zip(self.cells, self.norms)):
                h, c = cell(inp, *states[layer_idx])
                h = norm(h)
                states[layer_idx] = (h, c)
                inp = self.dropout(h) if layer_idx < len(self.cells) - 1 else h
            outputs.append(h.unsqueeze(1))
        return torch.cat(outputs, dim=1)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        groups = 8 if channels % 8 == 0 else 1
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(groups, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(groups, channels),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.block(x))


class ConvLSTMPredictorV2(nn.Module):
    def __init__(self, in_ch, hid_ch, n_layers, ksize=3, drop=0.1, decoder_blocks=3):
        super().__init__()
        self.encoder = StackedConvLSTM(in_ch, hid_ch, n_layers, ksize, drop)
        self.refine = nn.Sequential(*[ResidualBlock(hid_ch) for _ in range(decoder_blocks)])
        self.head = nn.Sequential(
            nn.Conv2d(hid_ch, hid_ch, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hid_ch, in_ch, 1),
        )

    def forward(self, x):
        feats = self.encoder(x)
        B, T, C, H, W = feats.shape
        y = feats.reshape(B * T, C, H, W)
        y = self.refine(y)
        y = self.head(y)
        return torch.sigmoid(y.reshape(B, T, -1, H, W))


def gaussian_kernel(window_size=11, sigma=1.5, channels=3, device="cpu", dtype=torch.float32):
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    kernel_2d = (g[:, None] @ g[None, :]).unsqueeze(0).unsqueeze(0)
    kernel_2d = kernel_2d.expand(channels, 1, window_size, window_size).contiguous()
    return kernel_2d


def ssim_score(pred, target, window_size=11, sigma=1.5):
    B, T, C, H, W = pred.shape
    x = pred.reshape(B * T, C, H, W)
    y = target.reshape(B * T, C, H, W).to(dtype=x.dtype)
    kernel = gaussian_kernel(window_size, sigma, C, x.device, x.dtype)
    padding = window_size // 2

    mu_x = F.conv2d(x, kernel, padding=padding, groups=C)
    mu_y = F.conv2d(y, kernel, padding=padding, groups=C)

    mu_x2 = mu_x.pow(2)
    mu_y2 = mu_y.pow(2)
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(x * x, kernel, padding=padding, groups=C) - mu_x2
    sigma_y2 = F.conv2d(y * y, kernel, padding=padding, groups=C) - mu_y2
    sigma_xy = F.conv2d(x * y, kernel, padding=padding, groups=C) - mu_xy

    c1 = x.new_tensor(0.01 ** 2)
    c2 = x.new_tensor(0.03 ** 2)

    ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
        (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2) + x.new_tensor(1e-8)
    )
    return ssim_map.mean().float()


class V2Loss(nn.Module):
    def __init__(self, alpha=0.55, beta=0.20, gamma=0.25):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.pixel = nn.SmoothL1Loss(beta=0.05)

    def gradient_loss(self, pred, target):
        dy_p = pred[:, :, :, 1:, :] - pred[:, :, :, :-1, :]
        dx_p = pred[:, :, :, :, 1:] - pred[:, :, :, :, :-1]
        dy_t = target[:, :, :, 1:, :] - target[:, :, :, :-1, :]
        dx_t = target[:, :, :, :, 1:] - target[:, :, :, :, :-1]
        return (dy_p - dy_t).abs().mean() + (dx_p - dx_t).abs().mean()

    def forward(self, pred, target):
        pixel = self.pixel(pred, target)
        edge = self.gradient_loss(pred, target)
        ssim_term = 1.0 - ssim_score(pred, target)
        total = self.alpha * pixel + self.beta * edge + self.gamma * ssim_term
        stats = {
            "pixel": float(pixel.detach().item()),
            "edge": float(edge.detach().item()),
            "ssim_loss": float(ssim_term.detach().item()),
        }
        return total, stats


def make_loader(ds, batch_size, shuffle, num_workers, pin_memory):
    kwargs = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(ds, **kwargs)


def psnr(pred, target):
    mse = torch.mean((pred - target) ** 2).item()
    if mse <= 1e-12:
        return 99.0
    return float(20.0 * math.log10(1.0 / math.sqrt(mse)))


def evaluate(model, loader, criterion, device, use_amp):
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    preds_all = []
    targets_all = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            with autocast(device_type=device.type, enabled=use_amp):
                pred = model(xb)
                loss, _ = criterion(pred, yb)
            total_loss += loss.item()
            total_mae += (pred - yb).abs().mean().item()
            total_psnr += psnr(pred, yb)
            total_ssim += float(ssim_score(pred, yb).item())
            preds_all.append(pred.float().cpu())
            targets_all.append(yb.float().cpu())

    return {
        "loss": total_loss / max(1, len(loader)),
        "mae": total_mae / max(1, len(loader)),
        "psnr": total_psnr / max(1, len(loader)),
        "ssim": total_ssim / max(1, len(loader)),
        "preds": torch.cat(preds_all, dim=0),
        "targets": torch.cat(targets_all, dim=0),
    }


def save_curves(history, out_dir: Path):
    steps = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    axes[0].plot(steps, history["train_loss"], label="Train", linewidth=2)
    axes[0].plot(steps, history["val_loss"], label="Val", linewidth=2)
    axes[0].set_title("Loss")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(steps, history["train_mae"], label="Train", linewidth=2)
    axes[1].plot(steps, history["val_mae"], label="Val", linewidth=2)
    axes[1].set_title("MAE")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    axes[2].plot(steps, history["val_psnr"], label="Val PSNR", linewidth=2)
    axes[2].plot(steps, history["val_ssim"], label="Val SSIM", linewidth=2)
    axes[2].set_title("Validation Quality")
    axes[2].grid(alpha=0.3)
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(out_dir / "training_curves.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def save_preview(inputs, preds, targets, out_dir: Path):
    t_steps = min(4, inputs.shape[1])
    fig, axes = plt.subplots(3, t_steps, figsize=(4 * t_steps, 8))
    if t_steps == 1:
        axes = np.array(axes).reshape(3, 1)

    for t in range(t_steps):
        axes[0, t].imshow(inputs[0, t].permute(1, 2, 0).cpu().numpy().clip(0, 1))
        axes[0, t].set_title(f"Input {t}")
        axes[1, t].imshow(preds[0, t].permute(1, 2, 0).cpu().numpy().clip(0, 1))
        axes[1, t].set_title(f"Pred {t}")
        axes[2, t].imshow(targets[0, t].permute(1, 2, 0).cpu().numpy().clip(0, 1))
        axes[2, t].set_title(f"Target {t}")
        for r in range(3):
            axes[r, t].axis("off")

    plt.tight_layout()
    plt.savefig(out_dir / "predictions_preview.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def maybe_resume(model, optimizer, scheduler, scaler, resume_path, device):
    start_epoch = 1
    best_val = float("inf")
    if resume_path:
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_val = float(ckpt.get("best_val", best_val))
        print(f"Resumed from {resume_path} at epoch {start_epoch - 1}")
    return start_epoch, best_val


def main(cfg: Config):
    seed_everything(cfg.seed)
    device = setup_device(cfg.force_cuda)
    startup_report(device, cfg)
    use_amp = bool(cfg.amp and device.type == "cuda")

    out_dir = Path(cfg.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "best_model.pt"

    frames, bad_files = load_frames(cfg)
    X, Y = build_pairs(frames, cfg.seq_len, cfg.stride)
    splits = split_pairs(X, Y)

    train_ds = SequenceDataset(*splits["train"], augment=True)
    val_ds = SequenceDataset(*splits["val"], augment=False)
    test_ds = SequenceDataset(*splits["test"], augment=False)

    print(f"Frames: {len(frames)} | Pairs: {len(X)}")
    print(f"Splits: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    train_loader = make_loader(train_ds, cfg.batch_size, True, cfg.num_workers, device.type == "cuda")
    val_loader = make_loader(val_ds, cfg.batch_size, False, cfg.num_workers, device.type == "cuda")
    test_loader = make_loader(test_ds, cfg.batch_size, False, cfg.num_workers, device.type == "cuda")

    model = ConvLSTMPredictorV2(
        cfg.img_c,
        cfg.filters,
        cfg.num_layers,
        cfg.ksize,
        cfg.dropout,
        cfg.decoder_blocks,
    ).to(device)

    criterion = V2Loss(alpha=0.70, beta=0.20, gamma=0.10)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.epochs,
        eta_min=cfg.min_lr,
    )
    scaler = GradScaler(device=device.type, enabled=use_amp)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_mae": [],
        "val_mae": [],
        "val_psnr": [],
        "val_ssim": [],
    }

    start_epoch, best_val = maybe_resume(model, optimizer, scheduler, scaler, cfg.resume, device)
    bad_epochs = 0
    sample_x = None
    start = time.time()

    for epoch in range(start_epoch, cfg.epochs + 1):
        model.train()
        running_loss = 0.0
        running_mae = 0.0
        running_pixel = 0.0
        running_edge = 0.0
        running_ssim_loss = 0.0

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, enabled=use_amp):
                pred = model(xb)
                loss, loss_stats = criterion(pred, yb)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            running_mae += (pred.detach() - yb).abs().mean().item()
            running_pixel += loss_stats["pixel"]
            running_edge += loss_stats["edge"]
            running_ssim_loss += loss_stats["ssim_loss"]

            if sample_x is None:
                sample_x = xb[:1].detach().float().cpu()

        train_loss = running_loss / max(1, len(train_loader))
        train_mae = running_mae / max(1, len(train_loader))
        train_pixel = running_pixel / max(1, len(train_loader))
        train_edge = running_edge / max(1, len(train_loader))
        train_ssim_loss = running_ssim_loss / max(1, len(train_loader))

        val_stats = evaluate(model, val_loader, criterion, device, use_amp)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_stats["loss"])
        history["train_mae"].append(train_mae)
        history["val_mae"].append(val_stats["mae"])
        history["val_psnr"].append(val_stats["psnr"])
        history["val_ssim"].append(val_stats["ssim"])

        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:03d}/{cfg.epochs:03d} | train_loss={train_loss:.4f} | "
            f"val_loss={val_stats['loss']:.4f} | train_mae={train_mae:.4f} | "
            f"val_mae={val_stats['mae']:.4f} | val_psnr={val_stats['psnr']:.2f} | "
            f"val_ssim={val_stats['ssim']:.4f} | pixel={train_pixel:.4f} | "
            f"edge={train_edge:.4f} | ssim_loss={train_ssim_loss:.4f} | lr={lr_now:.2e}"
        )

        if val_stats["loss"] < best_val:
            best_val = val_stats["loss"]
            bad_epochs = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "best_val": best_val,
                    "config": asdict(cfg),
                },
                ckpt_path,
            )
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                print(f"Early stopping at epoch {epoch}.")
                break

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    test_stats = evaluate(model, test_loader, criterion, device, use_amp)

    save_curves(history, out_dir)
    if sample_x is not None:
        save_preview(sample_x, test_stats["preds"][:1], test_stats["targets"][:1], out_dir)

    metrics = {
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "frames": int(len(frames)),
        "pairs": int(len(X)),
        "splits": {"train": len(train_ds), "val": len(val_ds), "test": len(test_ds)},
        "bad_files": bad_files,
        "best_val_loss": float(best_val),
        "test_loss": float(test_stats["loss"]),
        "test_mae": float(test_stats["mae"]),
        "test_psnr": float(test_stats["psnr"]),
        "test_ssim": float(test_stats["ssim"]),
        "seconds": float(time.time() - start),
        "config": asdict(cfg),
    }

    with open(out_dir / "metrics_v2.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(out_dir / "run_info_v2.txt", "w", encoding="utf-8") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")

    print(
        json.dumps(
            {
                "best_val_loss": best_val,
                "test_loss": test_stats["loss"],
                "test_mae": test_stats["mae"],
                "test_psnr": test_stats["psnr"],
                "test_ssim": test_stats["ssim"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cloud movement predictor.")
    parser.add_argument("--data-dir", default="./BWimages")
    parser.add_argument("--results-dir", default="./Results")
    parser.add_argument("--img-h", type=int, default=128)
    parser.add_argument("--img-w", type=int, default=128)
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--filters", type=int, default=96)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--decoder-blocks", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", default="")
    parser.add_argument("--force-cuda", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()

    cfg = Config(
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        img_h=args.img_h,
        img_w=args.img_w,
        seq_len=args.seq_len,
        stride=args.stride,
        filters=args.filters,
        num_layers=args.num_layers,
        decoder_blocks=args.decoder_blocks,
        dropout=args.dropout,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        min_lr=args.min_lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        num_workers=args.num_workers,
        grad_clip=args.grad_clip,
        seed=args.seed,
        amp=not args.no_amp,
        force_cuda=args.force_cuda,
        resume=args.resume,
    )
    main(cfg)