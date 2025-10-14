import os
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data.dataset import Dataset

class CustomDataset(Dataset):
    def __init__(self, 
                    root_path: str, 
                    image_path: str, 
                    mask_path: str,
                    image_size: int,
                    subsample: int = 1.0):
        self.root_path = root_path
        self.images = sorted([root_path+f"/{image_path}/"+i for i in os.listdir(root_path+f"/{image_path}/")])
        self.masks = sorted([root_path+f"/{mask_path}/"+i for i in os.listdir(root_path+f"/{mask_path}/")])
        
        if len(self.images) != len(self.masks): 
            raise ValueError("Length of images, masks, and heatmaps are not the same")
        
        # Subsample
        self.images = self.images[:int(len(self.images)*subsample)]
        self.masks = self.masks[:int(len(self.masks)*subsample)]
        
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])
        self.image_size = image_size
        
    def __getitem__(self, index) -> torch.tensor:
        img = Image.open(self.images[index]).convert("RGBA")   
        mask = Image.open(self.masks[index]).convert("L")
        
        return self.transform(img), self.transform(mask)

    def __len__(self) -> int:
        return len(self.images)
