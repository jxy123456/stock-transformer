import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader

from loguru import logger


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        config: dict,
        device: str = None,
    ):
        self.model = model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        training_cfg = config.get("training", {})
        self.lr = training_cfg.get("lr", 5e-4)
        self.weight_decay = training_cfg.get("weight_decay", 0.01)
        self.epochs = training_cfg.get("epochs", 100)
        self.patience = training_cfg.get("early_stopping_patience", 15)
        self.clip_norm = training_cfg.get("gradient_clip_norm", 1.0)
        self.warmup_epochs = training_cfg.get("warmup_epochs", 5)

        loss_name = training_cfg.get("loss", "huber")
        if loss_name == "huber":
            delta = training_cfg.get("huber_delta", 1.0)
            self.criterion = nn.HuberLoss(delta=delta)
        elif loss_name == "mse":
            self.criterion = nn.MSELoss()
        elif loss_name == "mae":
            self.criterion = nn.L1Loss()
        else:
            self.criterion = nn.HuberLoss(delta=1.0)

        self.optimizer = AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2
        )

        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.history = {"train_loss": [], "val_loss": []}

    def train_epoch(self, train_loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for x, y in train_loader:
            x = x.to(self.device)
            y = y.to(self.device)

            pred = self.model(x)
            loss = self.criterion(pred, y)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_norm)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def validate(self, val_loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                pred = self.model(x)
                loss = self.criterion(pred, y)
                total_loss += loss.item()
                n_batches += 1

        return total_loss / max(n_batches, 1)

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        checkpoint_dir: str = "outputs/checkpoints",
        symbol: str = "model",
        fold: int = 0,
    ) -> dict:
        checkpoint_path = Path(checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Training on {self.device}, {self.epochs} epochs max")

        for epoch in range(self.epochs):
            # Warmup LR
            if epoch < self.warmup_epochs:
                warmup_factor = (epoch + 1) / self.warmup_epochs
                for pg in self.optimizer.param_groups:
                    pg["lr"] = self.lr * warmup_factor

            t0 = time.time()
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)

            if epoch >= self.warmup_epochs:
                self.scheduler.step()

            elapsed = time.time() - t0
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)

            if (epoch + 1) % 5 == 0 or epoch == 0:
                logger.info(
                    f"Epoch {epoch+1}/{self.epochs} | "
                    f"train_loss={train_loss:.6f} val_loss={val_loss:.6f} | "
                    f"lr={self.optimizer.param_list[0]['lr']:.2e} | "
                    f"{elapsed:.1f}s"
                )

            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                best_path = checkpoint_path / f"{symbol}_fold{fold}_best.pt"
                torch.save(
                    {
                        "model_state": self.model.state_dict(),
                        "optimizer_state": self.optimizer.state_dict(),
                        "epoch": epoch,
                        "val_loss": val_loss,
                    },
                    best_path,
                )
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break

        return self.history

    def predict(self, data_loader: DataLoader) -> torch.Tensor:
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for x, _ in data_loader:
                x = x.to(self.device)
                pred = self.model(x)
                predictions.append(pred.cpu())
        return torch.cat(predictions, dim=0)
