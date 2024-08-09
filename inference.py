import argparse
import shutil
import os
import yaml
import timm
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_lightning import Trainer
import numpy as np
from PIL import Image
import wandb
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from dataset import ArtportalenDataModule, unnormalize

def get_args():
    parser = argparse.ArgumentParser(description='Inference without GradCAM visualization.')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the config.')
    parser.add_argument('--gpu', type=bool, default=False, help='Gpu true?.')
    
    return parser.parse_args()

def main():
    
    args = get_args()

    shutil.copyfile(args.config, "config.yaml")
    with open("config.yaml", 'r') as file:
        config = yaml.safe_load(file)
    from model import SimpleModel

    data = ArtportalenDataModule(data_dir=config['dataset'], batch_size=config['batch_size'], size=config['img_size'])
    data.prepare_testing_data(config['dataset'])
    dataloader = data.test_dataloader()

    model = SimpleModel(model_name=config['model_name'], pretrained=False, num_classes=config['num_classes'], outdir=config['outdir'])
    if args.gpu:
        checkpoint = torch.load(config['checkpoint'])
    else:
        checkpoint = torch.load(config['checkpoint'], map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(torch.device('cpu'))
    

    trainer = Trainer(accelerator="cpu")
    # trainer.fit(model, data)
    trainer.test(model, dataloaders=dataloader, ckpt_path=config['checkpoint'])

if __name__ == '__main__':
    main()
    # python inference.py --checkpoint checkpoints/model-j75sihxi-best-epoch049/model.ckpt --dataset testing/images --num-classes 5 --gpu False

