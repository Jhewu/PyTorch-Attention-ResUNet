# Local
from base import BaseTrainer

# External 
import torch

class SegmentationTrainer(BaseTrainer):
    """
    Subclass of BaseTrainer
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def eval_step(self, inputs, targets):    
        preds = self.model(inputs)
        pred_sigmoid = torch.nn.functional.sigmoid(preds)
        preds_binary = (pred_sigmoid > 0.5).float()  
        loss = self.loss_fn(preds, targets)
        return preds_binary, loss

    def evaluate(self, split: str = "test") -> tuple[float, dict]:
        train_loader, val_loader = self.dataloader.get_dataloader(split)
        loader = val_loader if split == "test" else train_loader
        loss, metrics = self._eval_epoch(loader)
        print(f"\n── {split.upper()} RESULTS ──")
        print(f"  loss: {loss:.4f}")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
        return loss, metrics