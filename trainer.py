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