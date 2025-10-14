from modules.unet import UNet
from dataset import CustomDataset

import os
import time
from typing import Tuple, List
from itertools import cycle

from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete

import torch
from torch.amp import GradScaler
from torch import optim
from torch.utils.data import DataLoader

from torchinfo import summary
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

class Trainer: 
    def __init__(self,
                model: torch.nn.Module,
                data_path: str, 
                model_path: str = None,
                device: str = "cuda",
                early_stopping_start: int = 50,
                image_size: int = 160, 
                batch_size: int = 128, 
                lr: float = 1e-4,
                epochs: int = 100, 
                patience: int = 25,
                load_and_train: bool = False,
                early_stopping: bool = True,
                mixed_precision: bool = True,
                ):
        """
        Initialize a generic training loop for a torch.nn.Module
        
        This class handles the complete training loop, writes to CSV
        
        Args:
            model (YOLOU): The YOLOU model instance to be trained
            data_path (str): Path to the dataset directory containing training data
            model_path (str, optional): Path to pre-trained YOLOv12-Seg model weights to load. Defaults to None.
            device (str, optional): Device to run training on ('cuda' or 'cpu'). Defaults to "cuda".
            early_stopping_start (int, optional): Epoch number to start early stopping monitoring. Defaults to 50.
            image_size (int, optional): Input image size for model training. Defaults to 160.
            batch_size (int, optional): Batch size for training. Defaults to 128.
            lr (float, optional): Learning rate for optimizer. Defaults to 1e-4.
            epochs (int, optional): Maximum number of training epochs. Defaults to 100.
            patience (int, optional): Number of epochs to wait before early stopping. Defaults to 25.
            load_and_train (bool, optional): Whether to load existing model and continue training. Defaults to False.
            early_stopping (bool, optional): Whether to enable early stopping mechanism. Defaults to True.
            mixed_precision (bool, optional): Whether to use mixed precision training. Defaults to True.
         
        Attributes:
            model (YOLOU): The YOLOU model being trained
            device (str): Training device ('cuda' or 'cpu')
            data_path (str): Path to training dataset
            model_path (str): Path to pre-trained model weights
            loss (YOLOULoss): Loss function for training
            image_size (int): Input image dimensions
            batch_size (int): Training batch size
            lr (float): Learning rate
            epochs (int): Maximum training epochs
            early_stopping_start (int): Early stopping initiation epoch
            patience (int): Early stopping patience parameter
            load_and_train (bool): Flag for loading and continuing training
            early_stopping (bool): Flag for enabling early stopping
            mixed_precision (bool): Flag for mixed precision training
            history (None): Training history storage (initialized as None)
            
        Methods:
            train: Execute the training process.
            create_dataloader: Get train and validation datasets as dataloaders.
            plot_loss_curves: Visualize training and validation loss/metric curves.
            save_model: Save model training checkpoints.
            get_current_time: Generate a timestamp for logging/output directories.
            create_dir: Create output directories for logs, models, or results.

        """
    
        self.model = model
        self.device = device
        self.data_path = data_path
        self.model_path = model_path
  
        self.loss = DiceLoss(
            include_background = False, # might change this later
            sigmoid=True, 
            reduction="mean")

        self.post_pred = AsDiscrete(threshold=0.5)
        
        self.metric = DiceMetric(
            include_background = False, 
            reduction="mean"
        )

        self.image_size = image_size
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.early_stopping_start = early_stopping_start
        self.patience = patience

        # bool
        self.load_and_train = load_and_train
        self.early_stopping = early_stopping
        self.mixed_precision = mixed_precision

        ### non-parameters
        self.history = None

    def get_current_time(self) -> str: 
        """
        Get current time in YMD | HMS format
        Used for creating non-conflicting result dirs
        """
        current_time = time.localtime()
        return time.strftime("%Y_%m_%d_%H_%M_%S", current_time)

    def create_dir(self, folder_name: str):
        """
        Creates the given directory if it does not exists
        """
        if not os.path.exists(folder_name):
            os.makedirs(folder_name) 

    def plot_loss_curves(self, save_path: str, filename: str = "plot.png") -> None:
        """
        Plot every metric stored in ``self.history``.
        The method automatically discovers keys, assigns a colour, and
        draws a legend entry for each.

        Parameters
        ----------
        save_path : str
            Directory to which the plot PNG will be written.
        filename : str, default "plot.png"
            File name for the saved image.
        """
        if not hasattr(self, "history") or not isinstance(self.history, dict):
            raise ValueError("`self.history` must be a dict of metric lists")

        # Create output dir if it does not exist
        os.makedirs(save_path, exist_ok=True)

        plt.figure(figsize=(10, 6))

        # Pick a colour palette – reuse if more metrics than colours
        colour_cycle = cycle(
            ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
             "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
             "#bcbd22", "#17becf"]
        )

        # Sort keys to keep a deterministic order
        for key in sorted(self.history.keys()):
            values: List[float] = self.history[key]
            # Use the key itself as the label (nice formatting optional)
            label = key.replace("_", " ").title()
            plt.plot(values, label=label, color=next(colour_cycle))

        plt.title("Training History")
        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)

        out_file = os.path.join(save_path, filename)
        plt.savefig(out_file)
        plt.show()
            
    def create_dataloader(self, data_path: str) -> Tuple[DataLoader, DataLoader]: 
        """
        Create dataloader from custom SegmentationDataset
        Depends on SegmentationDataset

        Args:
            data_path (str): root directory of dataset

        Returns:
            (Tuple[Dataloader]): train_dataloader and val_dataloader
        """
        train_dataset = CustomDataset(
                            root_path = data_path, 
                            image_path = "images/train",  
                            mask_path = "masks/train",
                            image_size = self.image_size)
                            
        val_dataset = CustomDataset(
                            root_path = data_path, 
                            image_path = "images/val", 
                            mask_path = "masks/val", 
                            image_size = self.image_size)

        train_dataloader = DataLoader(dataset=train_dataset,
                                    batch_size=self.batch_size,
                                    shuffle=True)
                                    
        val_dataloader = DataLoader(dataset=val_dataset,
                                    batch_size=self.batch_size,
                                    shuffle=False) # <- do not shuffle

        return train_dataloader, val_dataloader
    
    def train(self) -> None: 
        """
        Trains the model
        """
        train_dataloader, val_dataloader = self.create_dataloader(data_path=self.data_path)

        # Model training config
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        scaler = GradScaler(self.device) # --> mixed precision

        # Show summary
        summary(self.model, input_size=(self.batch_size, 4, self.image_size, self.image_size))

        # Initialize variables for callbacks
        self.history = dict(train_loss=[], val_loss=[], train_dice_metric=[], val_dice_metric=[])
        best_val_loss = float("inf")

        # Create result directory
        dest_dir = f"runs/{self.get_current_time()}" 
        model_dir = os.path.join(dest_dir, "weights")
        self.create_dir(model_dir)

        # Add seed for reproducibility
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)

        patience = 0 # --> local patience for early stopping

        self.metric.reset() # 1. Reset: Clears previous runs/epochs if resuming/starting
        for epoch in tqdm(range(self.epochs)):
            self.model.train()
            self.metric.reset()

            start_time = time.time()
            train_running_loss = 0
            train_running_dice_metric = 0

            if self.mixed_precision:
                for idx, img_mask in enumerate(tqdm(train_dataloader)):
                    img = img_mask[0].float().to(self.device)
                    mask = img_mask[1].float().to(self.device)

                    optimizer.zero_grad()
                    with torch.amp.autocast(device_type=self.device): 
                        pred = self.model(img) # logits
                        loss = self.loss(pred, mask)

                    if torch.isnan(loss):
                        print("NaN loss detected!")
                        print("Pred min/max:", pred.min().item(), pred.max().item())
                        print("Mask min/max:", mask.min().item(), mask.max().item())
                        break

                    scaler.scale(loss).backward()

                    # Unscales the gradients of optimizer's assigned params in-place
                    scaler.unscale_(optimizer)

                    # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                    # Optimizer's gradients are already unscaled, so scaler.step does not unscale them,
                    #   although it still skips optimizer.step() if the gradients contain infs or NaNs.
                    scaler.step(optimizer)

                    # Updates the scale for next iteration.
                    scaler.update()

                    train_running_loss += loss.item()

                    # Update metrics
                    pred_sigmoid = torch.nn.functional.sigmoid(pred)
                    pred_binary = self.post_pred(pred_sigmoid)
                    self.metric(pred_binary, mask)

            else:
                for idx, img_mask in enumerate(tqdm(train_dataloader)):
                    img = img_mask[0].float().to(self.device)
                    mask = img_mask[1].float().to(self.device)

                    optimizer.zero_grad()
                    pred = self.model(img) # logits
                    loss = self.loss(pred, mask)

                    train_running_loss += loss.item()
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()

                    # Update metrics
                    pred_sigmoid = torch.nn.functional.sigmoid(pred)
                    pred_binary = self.post_pred(pred_simoid)
                    self.metric(pred_binary, mask)
                    
            end_time = time.time()
            train_loss = train_running_loss / (idx + 1)# len(train_dataloader)
            train_dice_metric = self.metric.aggregate().item()
            self.metric.reset() # <- Reset again
            
            self.model.eval()
            val_running_loss = 0

            with torch.no_grad():
                for idx, img_mask in enumerate(tqdm(val_dataloader)):
                    img = img_mask[0].float().to(self.device)
                    mask = img_mask[1].float().to(self.device)

                    pred = self.model(img) # logits
                    loss = self.loss(pred, mask)

                    val_running_loss += loss.item()

                    pred_sigmoid = torch.nn.functional.sigmoid(pred)
                    pred_binary = self.post_pred(pred_sigmoid) 
                    self.metric(pred_binary, mask)

                val_loss = val_running_loss / (idx + 1) # len(val_dataloader)
                val_dice_metric = self.metric.aggregate().item()
                self.metric.reset() # <- reset
            
            # update the scheduler
            scheduler.step()

            # update the history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_dice_metric"].append(val_dice_metric)
            self.history["train_dice_metric"].append(train_dice_metric)

            if val_loss < best_val_loss: 
                if (best_val_loss - val_loss) > 1e-3:
                    print(f"Validation loss improved from {best_val_loss:.4f} to {val_loss:.4f}. Saving model...")
                    best_val_loss = val_loss
                    torch.save(self.model.state_dict(), os.path.join(os.path.join(model_dir, "best.pth")))
                    patience = 0
                else: 
                    print(f"Validation loss improved slightly from {best_val_loss:.4f} to {val_loss:.4f}, but not significantly enough to save the model.")
                    if epoch+1 >= self.early_stopping_start: 
                        patience+=1
            else:
                if epoch+1 >= self.early_stopping_start: 
                    patience+=1
            
            history_df = pd.DataFrame(self.history)
            history_df.to_csv(os.path.join(dest_dir, "history.csv"), index=False)

            print("-"*30)
            print(f"This is Patience {patience}")
            print(f"Training Speed per EPOCH (in seconds): {end_time - start_time:.4f}")
            print(f"Maximum Gigabytes of VRAM Used: {torch.cuda.max_memory_reserved(self.device) * 1e-9:.4f}")
            print(f"Train Loss EPOCH {epoch+1}: {train_loss:.4f}")
            print(f"Valid Loss EPOCH {epoch+1}: {val_loss:.4f}")
            print(f"Train DICE Score EPOCH {epoch+1}: {train_dice_metric:.4f}")
            print(f"Valid DICE Score EPOCH {epoch+1}: {val_dice_metric:.4f}")
            print("-"*30)

            if patience >= self.patience: 
                print(f"\nEARLY STOPPING: Valid Loss did not improve since epoch {epoch+1-patience}, terminating training...")
                break

        torch.save(self.model.state_dict(), os.path.join(os.path.join(model_dir, "last.pth")))
        self.plot_loss_curves(save_path=dest_dir)

if __name__ == "__main__": 
    unet = UNet(in_channels = 4, 
                widths = [64, 128, 256, 512], 
                num_classes = 1)
  
    trainer = Trainer(
                    model=unet,         
                    data_path="data/stacked_segmentation", 
                    model_path=None, 
                    load_and_train=False,
                    mixed_precision = True,
                    
                    epochs=50,
                    image_size = 160,
                    batch_size = 64,
                    lr = 1e-4,

                    early_stopping = True,
                    early_stopping_start = 10,
                    patience = 10, 
                    device = "cuda"
                    )
                            
    trainer.train()
