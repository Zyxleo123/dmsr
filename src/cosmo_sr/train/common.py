"""Shared training utilities: device, logging, checkpoints, slice previews."""
from __future__ import annotations

import csv
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from ..utils.config import collect_run_metadata, save_config, write_run_metadata


def grad_global_norm(parameters) -> float:
    """Global L2 norm of gradients (call after ``backward()``, before ``zero_grad``)."""
    total = 0.0
    for p in parameters:
        if p.grad is not None:
            total += float(p.grad.detach().pow(2).sum().item())
    return total ** 0.5


def gpu_mem_mb(device: Optional[torch.device] = None) -> Optional[float]:
    """Peak allocated CUDA memory in MB (``None`` on CPU)."""
    try:
        if torch.cuda.is_available() and (device is None or device.type == "cuda"):
            return torch.cuda.max_memory_allocated() / 1e6
    except Exception:
        pass
    return None


def system_metrics(device: torch.device, step_time_s: Optional[float] = None) -> Dict[str, float]:
    """Cheap, universally useful training telemetry (throughput, memory)."""
    out: Dict[str, float] = {}
    if step_time_s is not None and step_time_s > 0:
        out["step_time_ms"] = step_time_s * 1000.0
        out["steps_per_sec"] = 1.0 / step_time_s
    mem = gpu_mem_mb(device)
    if mem is not None:
        out["gpu_mem_mb"] = mem
    return out


def amp_components(enabled: bool, device: torch.device):
    """Mixed-precision helpers: ``(autocast_factory, scaler)``.

    ``autocast_factory()`` returns a context manager; ``scaler`` is a
    :class:`torch.amp.GradScaler`. Both are no-ops (fp32, disabled scaler) when
    ``enabled=False`` or ``device`` is not CUDA, so call sites can use them
    unconditionally::

        autocast, scaler = amp_components(cfg.get("amp", False), device)
        with autocast():
            loss = ...
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)   # before grad-norm / clipping
        scaler.step(optimizer)
        scaler.update()
    """
    use = bool(enabled) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use)

    def autocast():
        return torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=use)

    return autocast, scaler


def select_device(cfg_device: Optional[str] = None) -> torch.device:
    if cfg_device:
        return torch.device(cfg_device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def maybe_init_wandb(
    cfg: Dict[str, Any], run_dir: str | os.PathLike, job_type: str = "train"
) -> bool:
    """Initialise a Weights & Biases run if configured/available.

    Enabled unless ``cfg['wandb']['mode'] == 'disabled'``. Reads project/entity
    from ``cfg['wandb']`` (falling back to env ``WANDB_PROJECT``/``WANDB_ENTITY``
    and sensible defaults). Requires ``WANDB_API_KEY`` (from ``~/.bashrc``) for
    online logging; otherwise silently no-ops. Returns ``True`` on success.
    """
    wcfg = dict(cfg.get("wandb", {}) or {})
    mode = str(wcfg.get("mode", os.environ.get("WANDB_MODE", "online")))
    if mode == "disabled" or str(wcfg.get("enabled", True)).lower() == "false":
        return False
    if not os.environ.get("WANDB_API_KEY") and mode == "online":
        # no key -> fall back to offline so runs are still captured locally
        mode = "offline"
    try:
        import wandb
    except Exception:
        return False
    run_dir = Path(run_dir)
    meta = collect_run_metadata()
    try:
        wandb.init(
            project=wcfg.get("project", os.environ.get("WANDB_PROJECT", "cosmo_sr")),
            entity=wcfg.get("entity", os.environ.get("WANDB_ENTITY")),
            name=wcfg.get("name", run_dir.name),
            group=wcfg.get("group"),
            job_type=job_type,
            dir=str(run_dir),
            config={**cfg, "run_metadata": meta},
            mode=mode,
            reinit=True,
        )
    except Exception as e:  # pragma: no cover - network/setup issues
        print(f"[wandb] init failed ({e}); continuing without wandb")
        return False
    return True


def finish_wandb() -> None:
    try:
        import wandb

        if wandb.run is not None:
            wandb.finish()
    except Exception:
        pass


class CSVLogger:
    """Minimal CSV logger; mirrors to TensorBoard and Weights & Biases if active."""

    def __init__(
        self,
        run_dir: str | os.PathLike,
        filename: str = "metrics.csv",
        use_wandb: bool = False,
    ):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.run_dir / filename
        self._fieldnames: List[str] = ["step"]
        self._rows: List[Dict[str, float]] = []
        self._tb = None
        self._wandb = None
        try:
            from torch.utils.tensorboard import SummaryWriter

            self._tb = SummaryWriter(log_dir=str(self.run_dir / "tb"))
        except Exception:
            self._tb = None
        if use_wandb:
            try:
                import wandb

                if wandb.run is not None:
                    self._wandb = wandb
            except Exception:
                self._wandb = None

    def log(self, step: int, metrics: Dict[str, float]) -> None:
        row: Dict[str, float] = {"step": int(step)}
        for k, v in metrics.items():
            row[k] = float(v)
        new_cols = [k for k in row if k not in self._fieldnames]
        self._fieldnames.extend(new_cols)
        self._rows.append(row)
        # New columns can appear mid-run (e.g. periodic eval metrics); rewrite the
        # whole file on a schema change, otherwise append the single row.
        if new_cols or not self.path.exists():
            with open(self.path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=self._fieldnames, extrasaction="ignore")
                w.writeheader()
                for r in self._rows:
                    w.writerow({k: r.get(k, "") for k in self._fieldnames})
        else:
            with open(self.path, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=self._fieldnames, extrasaction="ignore")
                w.writerow({k: row.get(k, "") for k in self._fieldnames})
        if self._tb is not None:
            for k, v in metrics.items():
                self._tb.add_scalar(k, float(v), int(step))
        if self._wandb is not None and self._wandb.run is not None:
            self._wandb.log({k: float(v) for k, v in metrics.items()}, step=int(step))

    def close(self) -> None:
        if self._tb is not None:
            self._tb.close()


def save_checkpoint(path: str | os.PathLike, model: torch.nn.Module, optimizer=None,
                    step: int = 0, extra: Optional[Dict[str, Any]] = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state: Dict[str, Any] = {"model": model.state_dict(), "step": int(step)}
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    if extra:
        state["extra"] = extra
    torch.save(state, str(path))


def load_checkpoint(path: str | os.PathLike, model: torch.nn.Module, optimizer=None,
                    map_location=None) -> Dict[str, Any]:
    state = torch.load(str(path), map_location=map_location)
    model.load_state_dict(state["model"])
    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    return state


def save_slice_png(path: str | os.PathLike, field: np.ndarray, channel: int = 0,
                   title: Optional[str] = None) -> None:
    """Save a 2D central slice of a ``(C, N, N, N)`` field as a PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    N = field.shape[1]
    sl = field[channel, N // 2]
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(sl, origin="lower")
    if title:
        ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def init_run_dir(run_dir: str | os.PathLike, cfg: Dict[str, Any]) -> Path:
    """Create the run directory, saving the config and environment metadata."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, run_dir / "config.yaml")
    write_run_metadata(run_dir, extra={"run_dir": str(run_dir)})
    return run_dir


def to_device_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def build_lr_schedule(optimizer, base_lr: float, steps: int, warmup: int = 0,
                      min_lr_frac: float = 1.0):
    """Linear warmup then cosine decay to ``min_lr_frac * base_lr``.

    Returns ``None`` when there is nothing to do (no warmup and no decay), so
    callers can keep a constant LR without paying for a scheduler.
    """
    warmup = max(0, int(warmup))
    if warmup == 0 and min_lr_frac >= 1.0:
        return None

    def lr_lambda(step: int) -> float:
        if warmup and step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, steps - warmup)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_frac + (1.0 - min_lr_frac) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def estimate_residual_std(
    paired_iter, ops, device: torch.device, n_batches: int = 32
) -> torch.Tensor:
    """Per-channel std of the degradation residual ``r = x_R - A_R(x_2R)``.

    Two consumers need this and both break badly without it:

    * the channel-weighted MSE -- the residual std differs by ~9x between the
      displacement channels (~0.066) and the velocity channels (~0.58), so an
      unweighted MSE puts ~99% of its weight on velocity and the model is
      effectively never trained on displacement;
    * the stochastic degrader -- conditional flow matching interpolates the
      target against a unit Gaussian, so a target whose channels span an order
      of magnitude in scale is badly conditioned. We whiten by these stds and
      un-whiten at sample time.

    Estimated from the training stream so it always matches the actual data,
    and stored in the checkpoint so sampling can reproduce it.
    """
    sq_sum = None
    n = 0
    for _ in range(max(1, int(n_batches))):
        b = to_device_batch(next(paired_iter), device)
        r = b["lr"] - ops.A(b["hr"])           # (B, C, N, N, N)
        s = r.pow(2).sum(dim=(0, 2, 3, 4))     # (C,)
        sq_sum = s if sq_sum is None else sq_sum + s
        n += r.shape[0] * r.shape[2] * r.shape[3] * r.shape[4]
    std = (sq_sum / max(n, 1)).sqrt()
    return std.clamp_min(1e-6).float()
