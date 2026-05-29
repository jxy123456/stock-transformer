"""训练循环。"""

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader


class Trainer:
    def __init__(self, model: nn.Module, config: dict, device: str = None):
        self.model = model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        tcfg = config.get("training", {})
        self.data_parallel = (
            self.device.startswith("cuda")
            and tcfg.get("data_parallel", True)
            and torch.cuda.device_count() > 1
        )
        if self.data_parallel:
            device_ids = tcfg.get("device_ids")
            self.model = nn.DataParallel(self.model, device_ids=device_ids)

        self.mixed_precision = self.device.startswith("cuda") and tcfg.get("mixed_precision", True)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.mixed_precision)
        self.epochs = tcfg.get("epochs", 100)
        self.patience = tcfg.get("early_stopping_patience", 15)
        self.min_delta = tcfg.get("early_stopping_min_delta", 0.0)
        self.clip_norm = tcfg.get("gradient_clip", 1.0)
        self.warmup_epochs = tcfg.get("warmup_epochs", 5)
        self.lr = tcfg.get("lr", 1e-4)
        wd = tcfg.get("weight_decay", 1e-4)
        self.loss_weights = tcfg.get("loss_weights", {"5d": 0.6, "20d": 0.4})

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = AdamW(self.model.parameters(), lr=self.lr, weight_decay=wd)
        self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=10, T_mult=2)

        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.history = {"train_loss": [], "val_loss": []}

    def _state_dict(self):
        model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        return model.state_dict()

    def train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total = 0.0
        for x, y in loader:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            y5, y20 = y[:, 0], y[:, 1]

            with torch.amp.autocast("cuda", enabled=self.mixed_precision):
                out = self.model(x)
                loss = (
                    self.loss_weights["5d"] * self.criterion(out["logits_5d"], y5) +
                    self.loss_weights["20d"] * self.criterion(out["logits_20d"], y20)
                )

            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            total += loss.item()
        return total / max(len(loader), 1)

    @torch.no_grad()
    def validate(self, loader: DataLoader) -> float:
        self.model.eval()
        total = 0.0
        for x, y in loader:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            y5, y20 = y[:, 0], y[:, 1]
            with torch.amp.autocast("cuda", enabled=self.mixed_precision):
                out = self.model(x)
                loss = (
                    self.loss_weights["5d"] * self.criterion(out["logits_5d"], y5) +
                    self.loss_weights["20d"] * self.criterion(out["logits_20d"], y20)
                )
            total += loss.item()
        return total / max(len(loader), 1)

    def train(self, train_loader, val_loader, checkpoint_dir="outputs/checkpoints",
              name="model", fold=0):
        ckpt_dir = Path(checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        n_gpus = torch.cuda.device_count() if self.device.startswith("cuda") else 0
        logger.info(
            f"Training on {self.device}, gpus={n_gpus}, "
            f"data_parallel={self.data_parallel}, amp={self.mixed_precision}, "
            f"{self.epochs} epochs max"
        )

        for epoch in range(self.epochs):
            if epoch < self.warmup_epochs:
                factor = (epoch + 1) / max(self.warmup_epochs, 1)
                for pg in self.optimizer.param_groups:
                    pg["lr"] = self.lr * factor
            else:
                self.scheduler.step()

            t0 = time.time()
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            elapsed = time.time() - t0

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)

            if (epoch + 1) % 10 == 0 or epoch == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                logger.info(
                    f"Epoch {epoch+1}/{self.epochs} | "
                    f"train={train_loss:.6f} val={val_loss:.6f} | lr={lr:.2e} | {elapsed:.1f}s"
                )

            if val_loss < self.best_val_loss - self.min_delta:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                path = ckpt_dir / f"{name}_fold{fold}_best.pt"
                torch.save({
                    "model_state": self._state_dict(),
                    "optimizer_state": self.optimizer.state_dict(),
                    "epoch": epoch, "val_loss": val_loss,
                }, path)
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        return self.history

    @torch.no_grad()
    def predict(self, loader: DataLoader):
        """返回 logits_5d, logits_20d 的 numpy 数组。"""
        self.model.eval()
        out_5d, out_20d = [], []
        for x, _ in loader:
            x = x.to(self.device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=self.mixed_precision):
                o = self.model(x)
            out_5d.append(o["logits_5d"].cpu().numpy())
            out_20d.append(o["logits_20d"].cpu().numpy())
        return (
            np.concatenate(out_5d, axis=0),
            np.concatenate(out_20d, axis=0),
        )
