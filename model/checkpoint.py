from pathlib import Path
from typing import Optional

import torch
from loguru import logger


class CheckpointManager:
    def __init__(self, checkpoint_dir: str = "outputs/checkpoints", keep_best_n: int = 3):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep_best_n = keep_best_n

    def save(
        self,
        model_state: dict,
        optimizer_state: dict,
        config: dict,
        epoch: int,
        val_loss: float,
        symbol: str = "model",
        fold: int = 0,
    ) -> Path:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.checkpoint_dir / f"{symbol}_fold{fold}_{timestamp}.pt"

        torch.save(
            {
                "model_state": model_state,
                "optimizer_state": optimizer_state,
                "config": config,
                "epoch": epoch,
                "val_loss": val_loss,
            },
            path,
        )
        logger.debug(f"Checkpoint saved: {path}")
        self._cleanup(symbol, fold)
        return path

    def load(
        self, model: torch.nn.Module, path: str, load_optimizer: bool = False
    ) -> Optional[dict]:
        path = Path(path)
        if not path.exists():
            # Try to find best checkpoint
            candidates = list(self.checkpoint_dir.glob(f"*_best.pt"))
            if not candidates:
                logger.warning(f"No checkpoint found at {path}")
                return None
            path = candidates[-1]

        ckpt = torch.load(path, map_location="cpu")
        model.load_state_dict(ckpt["model_state"])
        logger.info(f"Loaded checkpoint: {path}, epoch={ckpt['epoch']}, val_loss={ckpt['val_loss']:.6f}")
        return ckpt

    def _cleanup(self, symbol: str, fold: int):
        pattern = f"{symbol}_fold{fold}_*.pt"
        checkpoints = sorted(
            self.checkpoint_dir.glob(pattern),
            key=lambda p: p.stat().st_mtime,
        )
        # Keep best file and latest N
        best_files = [c for c in checkpoints if "best" in c.name]
        regular_files = [c for c in checkpoints if "best" not in c.name]

        while len(regular_files) > self.keep_best_n:
            oldest = regular_files.pop(0)
            oldest.unlink()
