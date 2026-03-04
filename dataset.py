# Internal
import os
import random
from typing import Tuple

# External
import cv2

import torch
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

# Local 
from base import BaseDataLoader, BaseDataset

class SegmentationDataLoader(BaseDataLoader):
    def __init__(
        self,
        # Dataset Parameters
        root_path:   str,
        image_dir:   str,
        mask_dir:    str,
        image_size:  int,
        augmentation: bool = True,
        subsample:   float = 1.0,

        # DataLoader Parameters
        batch_size:  int   = 128,
        num_workers: int   = 10,
        shuffle:     bool = False,
        persistent_workers: bool = True, 
        pin_memory: bool = True, 
    ):
        
        self.train_dataset = SegmentationDataset(
                            root_path = root_path, 
                            image_path = f"{image_dir}/train",  
                            mask_path = f"{mask_dir}/train",
                            image_size = image_size, 
                            augmentation = augmentation, 
                            subsample = subsample)
                            
        self.val_dataset = SegmentationDataset(
                            root_path = root_path, 
                            image_path = f"{image_dir}/test", 
                            mask_path = f"{mask_dir}/test", 
                            image_size = image_size, 
                            augmentation = augmentation, 
                            subsample = subsample)

        self._batch_size  = batch_size
        self._num_workers = num_workers
        self._shuffle = shuffle
        self._persistent_workers = persistent_workers
        self._pin_memory = pin_memory

    def get_dataloader(self, split: str) -> DataLoader:
        train_dataloader = DataLoader(dataset=self.train_dataset,
                                    batch_size=self._batch_size,
                                    shuffle=self._shuffle, 
                                    num_workers=self._num_workers, 
                                    persistent_workers=self._persistent_workers,
                                    pin_memory=self._pin_memory)
                                    
        val_dataloader = DataLoader(dataset=self.val_dataset,
                                    batch_size=self._batch_size,
                                    shuffle=self._shuffle, 
                                    num_workers=self._num_workers, 
                                    persistent_workers=self._persistent_workers,
                                    pin_memory=self._pin_memory) # <- do not shuffle

        return train_dataloader, val_dataloader

class SegmentationDataset(BaseDataset):
    def __init__(self,
                 root_path: str,
                 image_path: str,
                 mask_path: str,
                 image_size: int = 160,
                 augmentation: bool = True,
                 subsample: float = 1.0):

        # Paths
        self.root_path = root_path
        self.image_dir = root_path + f"/{image_path}/"
        self.mask_dir = root_path + f"/{mask_path}/"

        image_filenames = sorted(os.listdir(self.image_dir))
        self.basenames = [os.path.splitext(f)[0] for f in image_filenames]
        self.basenames = self.basenames[:int(len(self.basenames) * subsample)]

        # Basic validation check (for mask/image pairs)
        for basename in self.basenames:
            if not os.path.exists(self.mask_dir + basename + ".png"):
                raise FileNotFoundError(f"Mask file not found for {basename}")

        # Loading
        self.image_size = image_size
        self.to_tensor = transforms.ToTensor()

        # Augmentation
        self.do_augmentation = augmentation
        self.augmentation = transforms.Compose([
            transforms.RandomAffine(
                degrees=10,
                translate=(5/self.image_size, 5 /
                           self.image_size),  # <- ~5 pixels
                # <- Mild zoom in/out
                scale=(0.8, 1.2)),
            transforms.RandomHorizontalFlip(p=0.5)
        ])

    def __getitem__(self, index) -> Tuple[torch.tensor, torch.tensor]:
        basename = self.basenames[index]

        # Construct Imag/Masks Paths
        img_path = self.image_dir + basename + ".png"
        mask_path = self.mask_dir + basename + ".png"

        # Load Image (RGBA -> BGRA in cv2.IMREAD_UNCHANGED)
        img_rgb_a = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

        # Load Mask (L/Grayscale)
        mask_np = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Resize Image & Masks
        img_resized = cv2.resize(
            img_rgb_a, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        mask_resized = cv2.resize(
            mask_np, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)

        # Convert to PyTorch Tensor (HWC -> CWH, /255)
        img = self.to_tensor(img_resized)
        mask = self.to_tensor(mask_resized)

        # Data Augmentation
        if self.do_augmentation:

            # Horizontal Flip (p = 0.5)
            if random.random() < 0.5:
                img = TF.hflip(img)
                mask = TF.hflip(mask)

            # Rotation (+10 degree, p = 0.5)
            if random.random() < 0.5:
                angle = random.uniform(-10, 10)
                img = TF.rotate(
                    img, angle, interpolation=InterpolationMode.BILINEAR)
                mask = TF.rotate(mask, angle,
                                 interpolation=InterpolationMode.NEAREST)

            # Translation 3% (p = 0.5)
            if random.random() < 0.5:
                max_t = int(0.03 * self.image_size)
                tx = random.randint(-max_t, max_t)
                ty = random.randint(-max_t, max_t)

                img = TF.affine(
                    img, angle=0, translate=(tx, ty),
                    scale=1.0, shear=0,
                    interpolation=InterpolationMode.BILINEAR
                )
                mask = TF.affine(
                    mask, angle=0, translate=(tx, ty),
                    scale=1.0, shear=0,
                    interpolation=InterpolationMode.NEAREST
                )

            # Scaling 5% (p = 0.5)
            if random.random() < 0.5:
                scale = random.uniform(0.9, 1.1)

                img = TF.affine(
                    img, angle=0, translate=(0, 0),
                    scale=scale, shear=0,
                    interpolation=InterpolationMode.BILINEAR
                )
                mask = TF.affine(
                    mask, angle=0, translate=(0, 0),
                    scale=scale, shear=0,
                    interpolation=InterpolationMode.NEAREST
                )

        return img, mask

    def __len__(self) -> int:
        return len(self.basenames)