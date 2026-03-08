# Local
from models.unet import UNet
from models.tiny_unet import TinyUNet
from models.unext import UNext
from models.attention_unet import AttU_Net

from trainer import SegmentationTrainer
from metrics import SegmentationMetrics
from loss import SegmentationLoss
from dataset import SegmentationDataLoader # <- import later

from tools.count_parameters import print_trainable_parameters

# External
import yaml 
import torch
import numpy as np

# Internal 
import random
import argparse

def set_seed(seed: int = 42): 
    # Set Seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        # Sets seed for all available GPUs
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

if __name__ == "__main__": 
    # -------------------------------------------------------------
    des="""
    Run evaluation with the specified parameters in parameters.yaml
    """
    # -------------------------------------------------------------

    parser = argparse.ArgumentParser(description=des.lstrip(" "), formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-p", "--param_dir", type=str, help='directory of YAML training parameter configuration file\t[parameters.yaml]')
    parser.add_argument("-d", "--dataset_dir", type=str, help='root directory of dataset\t[]')
    parser.add_argument("-w", "--weight_dir", type=str, help='directory of pretrained weights\t[runs/segmantic_0/best.pt]')
    args = parser.parse_args()

    ## --- Parse ----------------- ##
    if args.param_dir is not None:
        PARAM_DIR = args.param_dir
    else: 
        PARAM_DIR = "parameters.yaml"

    if args.weight_dir is not None: 
        WEIGHT_DIR = args.weight_dir
    else: 
        WEIGHT_DIR = "runs/segmantic_0/best.pt"

    ## --- Load Parameters ----------------- ##
    with open(f"{PARAM_DIR}", "r") as f:
        params = yaml.safe_load(f)
    
    # Set configuration to load pretrained model weights
    params['trainer']['training']['use_load_and_train'] = True
    params['trainer']['training']['load_and_train_path'] = WEIGHT_DIR
    
    if args.dataset_dir is not None: 
        # TODO: Make declaration consistent
            params['dataloader']['root_path'] = args.dataset_dir

    model_cfg = params['model']
    MODEL = model_cfg['name'].lower()
    if MODEL == "unet":
        model = UNet(
            in_channels = model_cfg['in_channels'],
            num_classes = model_cfg['out_channels'],
            widths = [64, 128, 256, 512], 
        ).to("cuda")
        
    elif MODEL == "tiny_unet":
        model = TinyUNet(
            in_channels = model_cfg['in_channels'], 
            num_classes = model_cfg['out_channels']).to("cuda")   
        
    elif MODEL == "unext": 
        model = UNext(
            input_channels = model_cfg['in_channels'],
            num_classes = model_cfg['out_channels']).to("cuda")
    
    elif MODEL == "attention_unet": 
        model = AttU_Net(
            img_ch=model_cfg['in_channels'], 
            output_ch=model_cfg['out_channels']).to('cuda')
    else: 
        raise ValueError(f"Model {MODEL} not recognized. Please choose from ['unet', 'tiny_unet', 'unext', 'attention_unet']")
    
    metrics = SegmentationMetrics()
    loss = SegmentationLoss()
    print_trainable_parameters(model)

    d_cfg = params['dataloader']
    dataloader = SegmentationDataLoader(
        root_path= d_cfg['root_path'],
        image_dir=d_cfg['image_dir'],
        mask_dir=d_cfg['mask_dir'],
        image_size=d_cfg['image_size'],
        augmentation=False,
        subsample=d_cfg['subsample'],
        batch_size=d_cfg['batch_size'],
        num_workers=d_cfg['num_workers'],
        shuffle=False,
        persistent_workers=d_cfg['use_persistent_workers'],
        pin_memory=d_cfg['use_pin_memory'],
    )

    set_seed()

    trainer = SegmentationTrainer(model=model, 
            loss_fn=loss, 
            metrics=metrics, 
            dataloader=dataloader, 
            params=params['trainer'], 
            param_dir=PARAM_DIR
            )

    trainer.evaluate(
        split = "test")