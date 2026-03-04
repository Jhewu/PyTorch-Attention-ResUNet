# Local
from base import BaseLoss

# External 
import torch
from monai.losses import DiceCELoss

class SegmentationLoss(BaseLoss):
    def __init__(self):
        self.loss = DiceCELoss(
            include_background=False,  # Single class
            to_onehot_y=False,
            sigmoid=True,
            reduction="mean",
            batch=True,)
    
    def compute(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.loss(predictions, targets)

    def __call__(self, predictions, targets):
        return self.compute(predictions, targets)




