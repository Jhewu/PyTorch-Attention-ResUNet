# External
import gc

import torch
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.optim import Optimizer
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm

# Internal
import os
import time
import copy
import shutil 
import contextlib
from itertools import cycle
from typing import Any, Tuple, List
from abc import ABC, abstractmethod

OPTIMIZER_REGISTRY = {
    "adam":     torch.optim.Adam,
    "adamw":    torch.optim.AdamW,
    "sgd":      torch.optim.SGD,
    "rmsprop":  torch.optim.RMSprop,
}

SCHEDULER_REGISTRY = {
    "step":       torch.optim.lr_scheduler.StepLR,
    "cosine":     torch.optim.lr_scheduler.CosineAnnealingLR,
    "exponential":torch.optim.lr_scheduler.ExponentialLR,
    "plateau":    torch.optim.lr_scheduler.ReduceLROnPlateau,
}

# ─────────────────────────────────────────────
# 1. BASE CONTRACTS  (Abstract interfaces)
# ─────────────────────────────────────────────

class BaseModel(nn.Module, ABC):
    """All models must implement forward() and predict()."""

    @abstractmethod
    def forward(self, batch: Any) -> torch.Tensor:
        ...

    @abstractmethod
    def predict(self, batch: Any) -> torch.Tensor:
        """Post-processed output (e.g. argmax, sigmoid threshold)."""
        ...


class BaseLoss(ABC):
    """Wraps any loss function behind a unified interface."""

    @abstractmethod
    def compute(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ...

    def __call__(self, predictions, targets):
        return self.compute(predictions, targets)


class BaseMetrics(ABC):
    """Computes and accumulates metrics over a full epoch."""

    @abstractmethod
    def update(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        """Accumulate batch-level stats."""
        ...

    @abstractmethod
    def compute(self) -> dict[str, float]:
        """Return final aggregated metrics as a named dict."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Clear accumulated state between epochs."""
        ...

class BaseDataset(Dataset, ABC):
    """Owns dataset logic."""
    def __getitem__(self, index: int) -> Tuple:
        ...

class BaseDataLoader(DataLoader, ABC):
    """Owns ready-to-use DataLoaders."""

    @abstractmethod
    def get_dataloader(self, split: str) -> Tuple[DataLoader]:
        """Default returns two DataLoaders: train and val/test. Override if you want more splits."""
        ...

class BaseTrainer:
    """
    Generic trainer driven by a YAML config dict, e.g.:

        with open(PARAM_DIR, "r") as f:
            params = yaml.safe_load(f)

    trainer = SegmentationTrainer(model=model, 
        loss_fn=loss, 
        metrics=metrics, 
        dataloader=dataloader, 
        params=params['trainer'], 
        param_dir=PARAM_DIR)
    """
    def __init__(
        self,
        model:    BaseModel,
        loss_fn:  BaseLoss,
        metrics:  BaseMetrics,
        dataloader:  BaseDataLoader,
        params:   dict,          # raw yaml.safe_load() output
        param_dir: str, 
    ):
        # resolve device
        self.device  = params.get("device", "cuda" if torch.cuda.is_available() else "cpu")

        # ── model & components ────────────────
        self.model   = model.to(self.device)
        self.loss_fn = loss_fn
        self.metrics = metrics
        self.dataloader = dataloader

        # ── sub-dicts with safe fallbacks ─────
        t_cfg  = params.get("training",  {})
        o_cfg  = params.get("optimizer", {})
        s_cfg  = params.get("scheduler", {})
        e_cfg  = params.get("early_stopping", {})

        # ── training hyperparams ──────────────
        self.epochs          =     t_cfg.get("epochs",          10)
        self.mixed_precision =     t_cfg.get("use_mixed_precision", True)
        self.is_load_and_train =   t_cfg.get("use_load_and_train", False)
        self.load_and_train_path = t_cfg.get("load_and_train_path", None)
        self.gradient_clip   =     t_cfg.get("gradient_clip",   1.0)
        self.verbose         =     t_cfg.get("verbose",         True)

        # ── optimizer (looked up from registry) ──
        opt_name  = o_cfg.get("name", "adamW").lower()
        opt_cls   = OPTIMIZER_REGISTRY.get(opt_name)
        if opt_cls is None:
            raise ValueError(f"Unknown optimizer '{opt_name}'. "
                             f"Choose from: {list(OPTIMIZER_REGISTRY)}")
        self.optimizer: Optimizer = opt_cls(
            self.model.parameters(),
            lr=o_cfg.get("learning_rate", 1e-4),
            weight_decay=o_cfg.get("weight_decay", 0.0),
        )

        # ── scheduler (optional) ──────────────
        self.scheduler = None
        if s_cfg:
            sch_name = s_cfg.get("name", "").lower()
            sch_cls  = SCHEDULER_REGISTRY.get(sch_name)
            if sch_cls is None:
                raise ValueError(f"Unknown scheduler '{sch_name}'. "
                                 f"Choose from: {list(SCHEDULER_REGISTRY)}")
            self.scheduler = sch_cls(self.optimizer, **s_cfg.get("kwargs") or {})
        
        # ── early stopping ────────────────────
        self.early_stopping_enabled     = e_cfg.get("enabled", True)
        self.early_stopping_start_epoch = e_cfg.get("start_epoch", 0)
        self.early_stopping_patience    = e_cfg.get("patience", 10)
        self.early_stopping_metric      = e_cfg.get("metric", "val_loss")
        self.early_stopping_mode        = e_cfg.get("mode", "min")
        self.early_stopping_delta       = e_cfg.get("delta", 1e-3)

        # ── internal state ─────────────────────
        self._history: dict[str, list] = {"train_loss": [], "val_loss": [], "val_metrics": []}
        self._best_weights = None

        self._early_stopping_counter = 0
        self._best_es_metric = float("inf") if self.early_stopping_mode == "min" else float("-inf")

        self._scaler = (
            torch.amp.GradScaler()
            if self.mixed_precision and "cuda" in self.device
            else None
        )

        self.checkpoint_dir = self.build_checkpoint_path()
        self._create_dir(self.checkpoint_dir)
        shutil.copy(param_dir,
             os.path.join(self.checkpoint_dir, os.path.basename(param_dir)))
    
        if self.is_load_and_train and self.load_and_train_path is not None: 
            checkpoint = torch.load(self.load_and_train_path, weights_only=True)
            self.model.load_state_dict(checkpoint)

    # ── private utils ───────────────────────────

    def _get_current_time(self) -> str:
        """
        Get current time in YMD | HMS format
        Used for creating non-conflicting result dirs
        Returns
            (str) Time in Ymd | HMs format
        """
        current_time = time.localtime()
        return time.strftime("%Y_%m_%d_%H_%M_%S", current_time)

    def _create_dir(self, directory: str):
        """
        Creates the given directory if it does not exists
        Args:
            directory (str): directory to be created
        """
        if not os.path.exists(directory):
            os.makedirs(directory)

    # ── core loops ──────────────────────────── 

    def _move(self, batch):
        """Recursively move tensors in batch to device."""
        if isinstance(batch, torch.Tensor):
            return batch.to(self.device)
        if isinstance(batch, (list, tuple)):
            return type(batch)(self._move(x) for x in batch)
        if isinstance(batch, dict):
            return {k: self._move(v) for k, v in batch.items()}
        return batch   # non-tensor (e.g. str labels) left as-is

    def _unpack(self, batch):
        """
        Assumes batch is (inputs, targets) or a dict with keys 'inputs'/'targets'.
        Override this in a subclass if your DataLoader returns something else.
        """
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            return batch[0], batch[1]
        if isinstance(batch, dict):
            return batch["inputs"], batch["targets"]
        raise ValueError("Override _unpack() to handle your DataLoader format.")

    def _train_epoch(self, loader: DataLoader) -> float:
        self.set_model_to_train()
        total_loss, n = 0.0, 0

        autocast_ctx = (
                torch.amp.autocast(device_type=self.device)
                if self.mixed_precision
                else contextlib.nullcontext()
        )

        for raw_batch in tqdm(loader):
            batch          = self._move(raw_batch)
            inputs, targets = self._unpack(batch)

            self.optimizer.zero_grad()
            with autocast_ctx: # mixed precision  
                preds, loss = self.train_step(inputs, targets)

            if torch.isnan(loss):
                print("NaN loss detected!")
                print("Pred min/max:", preds.min().item(), preds.max().item())
                print("Targets min/max:", targets.min().item(), targets.max().item())
                
                raise RuntimeError("NaN loss detected — training aborted.")

            if self.mixed_precision and "cuda" in self.device: 
                self._scaler.scale(loss).backward()
                self._scaler.unscale_(self.optimizer)

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)

                self._scaler.step(self.optimizer)
                self._scaler.update()
            else: 
                loss.backward()

                nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)

                self.optimizer.step()

            total_loss += loss.item() * (targets.size(0) if hasattr(targets, "size") else 1)
            n          += (targets.size(0) if hasattr(targets, "size") else 1)

        return total_loss / max(n, 1)

    @torch.no_grad()
    def _eval_epoch(self, loader: DataLoader) -> tuple[float, dict]:
        self.model.eval()
        self.metrics.reset()
        total_loss, n = 0.0, 0

        autocast_ctx = (
                torch.amp.autocast(device_type=self.device)
                if self.mixed_precision
                else contextlib.nullcontext()
        )

        for raw_batch in tqdm(loader):
            batch           = self._move(raw_batch)
            inputs, targets = self._unpack(batch)

            with autocast_ctx:
                preds, loss = self.eval_step(inputs, targets)

            self.metrics.update(preds, targets)

            total_loss += loss.item() * (targets.size(0) if hasattr(targets, "size") else 1)
            n          += (targets.size(0) if hasattr(targets, "size") else 1)

        return total_loss / max(n, 1), self.metrics.compute()

    # ── public utils ──────────────────────────

    def build_checkpoint_path(self): 
        return os.path.join("checkpoints", self._get_current_time())

    def set_model_to_train(self): 
        """
        For overriding in case of models with different train/eval modes for submodules (e.g. BatchNorm, Dropout).
        """
        self.model.train()
    
    def _flatten_history(self, history: dict) -> dict:
        history_flat = {
            "train_loss": list(history["train_loss"]),
            "val_loss": list(history["val_loss"]), 
            # Create copy to avoid mutation of original history if subclassing
        }

        for metrics in history["val_metrics"]:
            for k, v in metrics.items():
                if k not in history_flat:
                    history_flat[k] = []
                history_flat[k].append(v)
        return history_flat
    
    def save_csv(self, history: dict, save_path: str) -> None: 
        history_flat: dict = self._flatten_history(history)
        pd.DataFrame(history_flat).to_csv(
        os.path.join(save_path), index=False)

    def save_history_plot(self, save_path: str, history: dict,
                         filename: str = "plot.png") -> None:
        """
        Plot every metric stored in 'self._history'.
        Uses dual y-axes to handle vastly different scales (e.g., HD95).
        """
        os.makedirs(save_path, exist_ok=True)

        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax2 = ax1.twinx()

        colour_cycle = cycle(
            ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
             "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
             "#bcbd22", "#17becf"]
        )

        # Plot loss metrics on left axis, HD95 on right
        for key in sorted(history.keys()):
            values = history[key]
            label = key.replace("_", " ").title()
            color = next(colour_cycle)
            
            if "hd95" in key.lower():
                ax2.plot(values, label=label, color=color, linestyle="--")
            else:
                ax1.plot(values, label=label, color=color)

        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss / Metrics")
        ax2.set_ylabel("HD95")
        ax1.grid(True)
        
        # Combine legends from both axes
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

        plt.title("Training History")
        out_file = os.path.join(save_path, filename)
        plt.savefig(out_file)
        plt.close(fig)
        # plt.show()

    # ── public API ────────────────────────────
    def eval_step(self, inputs, targets):    
        preds = self.model(inputs)
        loss = self.loss_fn(preds, targets)
        return preds, loss

    def train_step(self, inputs, targets) -> torch.Tensor:
        """Returns loss for one batch. Override for custom forward logic."""
        preds = self.model(inputs)
        loss = self.loss_fn(preds, targets)
        return preds, loss

    def train(self):
        train_loader, val_loader = self.dataloader.get_dataloader("train")

        for epoch in range(1, self.epochs + 1):
            train_loss            = self._train_epoch(train_loader)
            val_loss, val_metrics = self._eval_epoch(val_loader)

            # ── resolve tracked metric (shared by checkpointing + early stopping) ──
            if self.early_stopping_metric == "val_loss":
                es_value = val_loss
            else:
                es_value = val_metrics.get(self.early_stopping_metric)
                if es_value is None:
                    raise ValueError(f"Metric '{self.early_stopping_metric}' not found in val_metrics")

            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(es_value)
                else:
                    self.scheduler.step()

            improved = (
                es_value < self._best_es_metric - self.early_stopping_delta
                if self.early_stopping_mode == "min"
                else es_value > self._best_es_metric + self.early_stopping_delta
            )

            # ── checkpoint best model ─────────────
            if improved:
                self._best_es_metric = es_value
                self._best_weights = {k: v.detach().cpu() for k, v in self.model.state_dict().items()}
                if self.checkpoint_dir:
                    torch.save(self._best_weights, self.checkpoint_dir + "/best.pth")

            # ── update history ─────────────────────
            self._history["train_loss"].append(train_loss)
            self._history["val_loss"].append(val_loss)
            self._history["val_metrics"].append(val_metrics)

            #── print training progress ────────────
            if self.verbose:
                metrics_str = "  ".join(f"{k}: {v:.4f}" for k, v in val_metrics.items())
                print(f"[{epoch:03d}/{self.epochs}]  "
                      f"train_loss: {train_loss:.4f}  "
                      f"val_loss: {val_loss:.4f}  {metrics_str}")

            # ── early stopping counter ────────────
            if self.early_stopping_enabled and epoch >= self.early_stopping_start_epoch:
                if improved:
                    self._early_stopping_counter = 0
                else:
                    self._early_stopping_counter += 1
                    if self._early_stopping_counter >= self.early_stopping_patience:
                        if self.verbose:
                            print(f"Early stopping triggered at epoch {epoch} "
                                  f"(no improvement for {self.early_stopping_patience} epochs)")
                        break

            # ── save history to csv ───────────────
            self.save_csv(history = self._history, save_path=f"{self.checkpoint_dir}/history.csv")

            snapshot = torch.cuda.memory_stats()
            print(f"[Epoch {epoch}] Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB  "
            f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB  "
            f"Active allocs: {snapshot['active.all.current']}")

            ### --- TROUBLESHOOTING --- ###
            # gc.collect()
            # torch.cuda.empty_cache()
            ### --- TROUBLESHOOTING --- ###

        # ── save last model weights ───────────────
        if self.checkpoint_dir:
            torch.save(self.model.state_dict(), self.checkpoint_dir + "/last.pth")
        
        # ── save history as plot ──────────────────
        self.save_history_plot(save_path=self.checkpoint_dir,
            history=self._flatten_history(self._history), filename="plot_all.png")

    def evaluate(self, split: str = "test") -> tuple[float, dict]:
        train_loader, val_loader = self.dataloader.get_dataloader(split)
        loader = val_loader if split == "test" else train_loader
        loss, metrics = self._eval_epoch(loader)
        print(f"\n── {split.upper()} RESULTS ──")
        print(f"  loss: {loss:.4f}")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
        return loss, metrics

# ─────────────────────────────────────────────
# 4. EXAMPLE CONCRETE IMPLEMENTATIONS
# ─────────────────────────────────────────────

# ── Model ─────────────────────────────────────
class SimpleClassifier(BaseModel):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x).argmax(dim=-1)


# ── Loss ──────────────────────────────────────
class CrossEntropyLoss(BaseLoss):
    def __init__(self):
        self._fn = nn.CrossEntropyLoss()

    def compute(self, preds, targets):
        return self._fn(preds, targets)


# ── Metrics ───────────────────────────────────
class AccuracyMetrics(BaseMetrics):
    def __init__(self):
        self.correct = 0
        self.total   = 0

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        self.correct += (preds == targets).sum().item()
        self.total   += targets.size(0)

    def compute(self) -> dict[str, float]:
        return {"accuracy": self.correct / max(self.total, 1)}

    def reset(self):
        self.correct = 0
        self.total   = 0


# ── Dataset ───────────────────────────────────
from torch.utils.data import TensorDataset

class SimpleTabularDataset(BaseDataset):
    def __init__(self, X: torch.Tensor, y: torch.Tensor,
                 val_frac=0.1, test_frac=0.1, batch_size=32):
        n     = len(X)
        n_val = int(n * val_frac)
        n_tst = int(n * test_frac)
        self._splits = {
            "train": TensorDataset(X[:n - n_val - n_tst], y[:n - n_val - n_tst]),
            "val":   TensorDataset(X[n - n_val - n_tst: n - n_tst], y[n - n_val - n_tst: n - n_tst]),
            "test":  TensorDataset(X[-n_tst:], y[-n_tst:]),
        }
        self.batch_size = batch_size

    def get_dataloader(self, split: str) -> DataLoader:
        return DataLoader(
            self._splits[split],
            batch_size=self.batch_size,
            shuffle=(split == "train"),
        )


# ─────────────────────────────────────────────
# 5. USAGE
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # TODO: Create a working example
    pass
    # torch.manual_seed(42)

    # # fake data: 500 samples, 20 features, 3 classes
    # X = torch.randn(500, 20)
    # y = torch.randint(0, 3, (500,))

    # trainer = BaseTrainer(
    #     model   = SimpleClassifier(input_dim=20, num_classes=3),
    #     loss_fn = CrossEntropyLoss(),
    #     metrics = AccuracyMetrics(),
    #     dataset = SimpleTabularDataset(X, y, batch_size=32),
    #     config  = TrainerConfig(epochs=5, learning_rate=1e-3),
    # )

    # trainer.train()
    # trainer.evaluate("test")