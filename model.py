# https://github.com/karasawatakumi/pytorch-image-classification/blob/main/train.py

import os
import numpy as np
import yaml

import wandb
import timm
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from torchmetrics import Accuracy

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from dataset import ArtportalenDataModule, unnormalize

# Load configuration from YAML file
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

class SimpleModel(LightningModule):
    def __init__(
        self,
        model_name: str = 'resnet18',
        pretrained: bool = False,
        num_classes: int | None = None,
        outdir: str = 'results',
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = timm.create_model(
            model_name=model_name, pretrained=pretrained, num_classes=num_classes
        )
        self.train_loss = nn.CrossEntropyLoss()
        self.train_acc = Accuracy(task='multiclass', num_classes=num_classes)
        self.val_loss = nn.CrossEntropyLoss()
        self.val_acc = Accuracy(task='multiclass', num_classes=num_classes)
        self.gradient = None
        self.outdir = outdir
    
    def activations_hook(self, grad):
        self.gradient = grad

    def get_gradient(self):
        return self.gradient

    def get_activations(self, x):
        for name, module in self.model.named_modules():
            if name == 'layer4':
                x = module(x)
                x.register_hook(self.activations_hook)
                return x
        return None

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, target = batch

        out = self(x)
        _, pred = out.max(1)

        loss = self.train_loss(out, target)
        acc = self.train_acc(pred, target)
        self.log_dict({'train/loss': loss, 'train/acc': acc}, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        with torch.enable_grad():
            x, target = batch
            out = self(x)
            _, pred = out.max(1)

            loss = self.val_loss(out, target)
            acc = self.val_acc(pred, target)
            self.log_dict({'val/loss': loss, 'val/acc': acc})

            # self.model.eval() # handled by pytorch lightning

            unnormalized_x = unnormalize(x[0].cpu(), config['transforms']['mean'], config['transforms']['std']).permute(1, 2, 0).numpy()
            unnormalized_x = np.clip(unnormalized_x, 0, 1)  # Ensure the values are within [0, 1]


            cam = GradCAM(model=self.model, target_layers=[self.model.layer4[-1]])
            targets = [ClassifierOutputTarget(class_idx) for class_idx in target]
            grayscale_cam = cam(input_tensor=x, targets=targets)
            grayscale_cam = grayscale_cam[0, :]
            visualization = show_cam_on_image(unnormalized_x, grayscale_cam, use_rgb=True)
            img = Image.fromarray((visualization * 255).astype(np.uint8))

             # Log image to Wandb
            wandb_img = wandb.Image(visualization, caption=f"GradCAM Batch {batch_idx} Image 0")
            self.logger.experiment.log({"GradCAM Images": wandb_img})

            
            # save locally
            os.makedirs(self.outdir, exist_ok=True)
            img.save(os.path.join(self.outdir, f'cam_image_val_batch{batch_idx}_img0.png'))
            
            # To save all images in batch:
            # for i in range(len(x)):
            #     grayscale_cam_img = 
            # grayscale_cam[i]
            #     visualization = show_cam_on_image(x[i].cpu().numpy().transpose(1, 2, 0), grayscale_cam_img, use_rgb=True)
            #     img = Image.fromarray((visualization * 255).astype(np.uint8))
            #     os.makedirs(self.hparams.outdir, exist_ok=True)
            #     img.save(os.path.join(self.hparams.outdir, f'cam_image_val_batch{batch_idx}_img{i}.png'))
            
            # self.model.train()


    def configure_optimizers(self):
        optimizer = get_optimizer(self.parameters())
        lr_scheduler_config = get_lr_scheduler_config(optimizer)
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}
    

def get_optimizer(parameters) -> torch.optim.Optimizer:
    if config['solver']['OPT'] == 'adam':
        optimizer = torch.optim.Adam(parameters, lr=config['solver']['BASE_LR'], weight_decay=config['solver']['WEIGHT_DECAY'])
    elif config['solver']['OPT'] == 'sgd':
        optimizer = torch.optim.SGD(
            parameters, lr=config['solver']['BASE_LR'], weight_decay=config['solver']['WEIGHT_DECAY'], momentum=config['solver']['MOMENTUM']
        )
    else:
        raise NotImplementedError()

    return optimizer


def get_lr_scheduler_config(optimizer: torch.optim.Optimizer) -> dict:
    if config['solver']['LR_SCHEDULER'] == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=config['solver']['LR_STEP_SIZE'], gamma=config['solver']['LR_DECAY_RATE']
        )
        lr_scheduler_config = {
            'scheduler': scheduler,
            'interval': 'epoch',
            'frequency': 1,
        }
    elif config['solver']['LR_SCHEDULER'] == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=config['solver']['LR_STEP_MILESTONES'], gamma=config['solver']['LR_DECAY_RATE']
        )
        lr_scheduler_config = {
            'scheduler': scheduler,
            'interval': 'epoch',
            'frequency': 1,
        }
    elif config['solver']['LR_SCHEDULER'] == 'reduce_on_plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.1, patience=10, threshold=0.0001
        )
        lr_scheduler_config = {
            'scheduler': scheduler,
            'monitor': 'val/loss',
            'interval': 'epoch',
            'frequency': 1,
        }
    else:
        raise NotImplementedError

    return lr_scheduler_config
