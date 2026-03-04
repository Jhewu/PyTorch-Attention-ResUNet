# External
from monai.metrics import DiceMetric, HausdorffDistanceMetric
import torch

# Local 
from base import BaseMetrics

class SegmentationMetrics(BaseMetrics):
    """
    Implements the MetricsCalculator protocol for binary segmentation.

    Encapsulates all MONAI-specific setup and preprocessing so that Trainer
    stays generic. The trainer only calls reset() / update() / compute().

    compute() returns:
        {
            "dice":      float,
            "hd95":      float,
            "precision": float,
            "recall":    float,
        }
    """
    def __init__(self):
        self._dice = DiceMetric(
            include_background=False,
            reduction="mean_batch",
            get_not_nans=False,
            ignore_empty=False,
            num_classes=2,       # [background, foreground] — single-class task
            return_with_label=False,
        )
        self._hd95 = HausdorffDistanceMetric(
            include_background=False,
            percentile=95,
            reduction="none",
            get_not_nans=True,
        )
        self._precision_sum = 0.0
        self._recall_sum = 0.0
        self._n_batches = 0

    def update(self, pred_binary: torch.Tensor, target: torch.Tensor) -> None:
        """
        Args:
            pred:   binarized logits from the model  (B, 1, H, W)
            target: binary ground-truth masks  (B, 1, H, W)
        """
        # Dice
        self._dice(pred_binary, target)

        # Precision / Recall
        # TODO: Separate Precision / Recall Metrics from the SegmentationMetrics Class
        TP = (pred_binary * target).sum().float()
        FP = (pred_binary * (1 - target)).sum().float()
        FN = ((1 - pred_binary) * target).sum().float()
        self._precision_sum += (TP / (TP + FP + 1e-6)).item()
        self._recall_sum += (TP / (TP + FN + 1e-6)).item()

        # HD95 — expects one-hot (B, C, H, W) with C=2
        pred_onehot = torch.cat([1 - pred_binary, pred_binary], dim=1)
        mask_onehot = torch.cat([1 - target, target], dim=1)
        self._hd95(pred_onehot, mask_onehot)

        self._n_batches += 1

    def compute(self) -> dict:
        dice = self._dice.aggregate().item()

        hd95_vals, not_nan_counts = self._hd95.aggregate()
        valid = not_nan_counts.bool()
        hd95 = torch.mean(hd95_vals[valid]).item() if valid.any() else float("nan")

        n = max(self._n_batches, 1)
        precision = self._precision_sum / n
        recall = self._recall_sum / n

        return dict(dice=dice, hd95=hd95, precision=precision, recall=recall)

    def reset(self) -> None:
        self._dice.reset()
        self._hd95.reset()
        self._precision_sum = 0.0
        self._recall_sum = 0.0
        self._n_batches = 0
