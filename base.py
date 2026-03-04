# External
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm

# Internal
import os
import copy
import contextlib
from typing import Any, Tuple
from abc import ABC, abstractmethod

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

class BaseTrainer:
    """
    Generic trainer driven by a YAML config dict, e.g.:

        with open(PARAM_DIR, "r") as f:
            params = yaml.safe_load(f)

        trainer = Trainer(model, loss_fn, metrics, dataset, params)

    Expected YAML structure (all keys optional — defaults shown):
    ┌─────────────────────────────────────────┐
    │ training:                               │
    │   epochs: 10                            │
    │   gradient_clip: 0.0                    │
    │   checkpoint_path: null                 │
    │   verbose: true                         │
    │                                         │
    │ optimizer:                              │
    │   name: adam          # see registry    │
    │   learning_rate: 1e-3                   │
    │   weight_decay: 0.0                     │
    │                                         │
    │ scheduler:            # optional block  │
    │   name: cosine        # see registry    │
    │   kwargs:             # passed as-is    │
    │     T_max: 10                           │
    │                                         │
    └─────────────────────────────────────────┘
    """

    def __init__(
        self,
        model:    BaseModel,
        loss_fn:  BaseLoss,
        metrics:  BaseMetrics,
        dataloader:  BaseDataLoader,
        params:   dict,          # raw yaml.safe_load() output
    ):
        # ── resolve device ────────────────────
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
        self.epochs          = t_cfg.get("epochs",          10)
        self.gradient_clip   = t_cfg.get("gradient_clip",   0.0)
        self.checkpoint_path = t_cfg.get("checkpoint_path", None)
        self.verbose         = t_cfg.get("verbose",         True)
        self.mixed_precision = t_cfg.get("mixed_precision", True)

        # ── optimizer (looked up from registry) ──
        opt_name  = o_cfg.get("name", "adam").lower()
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
        self.early_stopping_enabled = e_cfg.get("enabled", True)
        self.early_stopping_patience = e_cfg.get("patience", 10)
        self.early_stopping_delta = e_cfg.get("delta", 1e-3)
        self.early_stopping_metric = e_cfg.get("metric", "val_loss")
        self.early_stopping_mode = e_cfg.get("mode", "min")
        self.early_stopping_start_epoch = e_cfg.get("start_epoch", 0)

        self.history: dict[str, list] = {"train_loss": [], "val_loss": [], "val_metrics": []}
        self._best_weights = None

        self._early_stopping_counter = 0
        self._best_es_metric = float("inf") if self.early_stopping_mode == "min" else float("-inf")

        self._autocast_ctx = (
                torch.amp.autocast(device_type=self.device)
                if self.mixed_precision
                else contextlib.nullcontext()
        )

        self._scaler = (
            torch.amp.GradScaler()
            if self.mixed_precision and "cuda" in self.device
            else None
        )

    # ── core loops ──────────────────────────── 

    def create_dir(self, directory: str):
        """
        Creates the given directory if it does not exists
        Args:
            directory (str): directory to be created
        """
        if not os.path.exists(directory):
            os.makedirs(directory)
    
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

    def _train_step(self, inputs, targets) -> torch.Tensor:
        """Returns loss for one batch. Override for custom forward logic."""
        preds = self.model(inputs)
        loss = self.loss_fn(preds, targets)
        return preds, loss

    def _set_model_to_train(self): 
        """
        For overriding in case of models with different train/eval modes for submodules (e.g. BatchNorm, Dropout).
        """
        self.model.train()

    def _train_epoch(self, loader: DataLoader) -> float:
        self._set_model_to_train()
        total_loss, n = 0.0, 0
        for raw_batch in tqdm(loader):
            batch          = self._move(raw_batch)
            inputs, targets = self._unpack(batch)

            self.optimizer.zero_grad()
            with self._autocast_ctx:        
                preds, loss = self._train_step(inputs, targets)

            if torch.isnan(loss):
                print("NaN loss detected!")
                print("Pred min/max:", preds.min().item(), preds.max().item())
                print("Targets min/max:", targets.min().item(), targets.max().item())
                
                raise RuntimeError("NaN loss detected — training aborted.")

            if self.mixed_precision and "cuda" in self.device: 
                self._scaler.scale(loss).backward()
                self._scaler.unscale_(self.optimizer)

                if self.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)

                self._scaler.step(self.optimizer)
                self._scaler.update()
            else: 
                loss.backward()

                if self.gradient_clip > 0:
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
        for raw_batch in tqdm(loader):
            batch           = self._move(raw_batch)
            inputs, targets = self._unpack(batch)

            with self._autocast_ctx:
                preds, loss = self._train_step(inputs, targets)

            self.metrics.update(preds, targets)

            total_loss += loss.item() * (targets.size(0) if hasattr(targets, "size") else 1)
            n          += (targets.size(0) if hasattr(targets, "size") else 1)

        return total_loss / max(n, 1), self.metrics.compute()

    # ── public API ────────────────────────────

    def train(self):
        train_loader, val_loader = self.dataloader.get_dataloader("train")

        for epoch in range(1, self.epochs + 1):
            train_loss            = self._train_epoch(train_loader)
            val_loss, val_metrics = self._eval_epoch(val_loader)

            if self.scheduler:
                self.scheduler.step()

            # ── resolve tracked metric (shared by checkpointing + early stopping) ──
            if self.early_stopping_metric == "val_loss":
                es_value = val_loss
            else:
                es_value = val_metrics.get(self.early_stopping_metric)
                if es_value is None:
                    raise ValueError(f"Metric '{self.early_stopping_metric}' not found in val_metrics")

            improved = (
                es_value < self._best_es_metric - self.early_stopping_delta
                if self.early_stopping_mode == "min"
                else es_value > self._best_es_metric + self.early_stopping_delta
            )

            # ── checkpoint best model ──────────
            if improved:
                self._best_es_metric = es_value
                self._best_weights   = copy.deepcopy(self.model.state_dict())
                if self.checkpoint_path:
                    torch.save(self._best_weights, self.checkpoint_path)

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_metrics"].append(val_metrics)

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

        # restore best weights at end of training
        if self._best_weights is not None:
            self.model.load_state_dict(self._best_weights)

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