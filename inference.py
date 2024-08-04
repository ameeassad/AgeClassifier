import argparse
import os
import timm
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import numpy as np
from PIL import Image
import wandb
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from model import SimpleModel
from dataset import ArtportalenDataModule, unnormalize

def save_gradcam(model, dataloader, outdir):
    os.makedirs(outdir, exist_ok=True)
    for batch_idx, (images, targets) in enumerate(dataloader):
        images.requires_grad = True
        cam = GradCAM(model=model, target_layers=[model.model.layer4[-1]])
        targets = [ClassifierOutputTarget(target.item()) for target in targets]
        grayscale_cam = cam(input_tensor=images, targets=targets)
        
        for i in range(len(images)):
            unnormalized_img = unnormalize(images[i], model.mean, model.std)
            visualization = show_cam_on_image(unnormalized_img, grayscale_cam[i], use_rgb=True)
            img = Image.fromarray((visualization * 255).astype(np.uint8))
            img.save(os.path.join(outdir, f'cam_image_inference_batch{batch_idx}_img{i}.png'))
            
            # Optionally log to Wandb
            if wandb.run:
                wandb_img = wandb.Image(visualization, caption=f"GradCAM Batch {batch_idx} Image {i}")
                wandb.log({"GradCAM Images": wandb_img})

def get_args():
    parser = argparse.ArgumentParser(description='Inference with GradCAM visualization.')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the model checkpoint.')
    parser.add_argument('--dataset', type=str, required=True, help='Path to the dataset.')
    parser.add_argument('--model-name', type=str, default='resnet152', help='Model name (timm).')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size.')
    parser.add_argument('--num-classes', type=int, required=True, help='Number of classes.')
    parser.add_argument('--img-size', type=int, default=224, help='Input size of image.')
    parser.add_argument('--outdir', type=str, default='gradcam_outputs', help='Directory to save GradCAM images.')
    parser.add_argument('--use-wandb', action='store_true', help='Flag to use Wandb for logging.')
    parser.add_argument('--wandb-run-id', type=str, help='Wandb run ID to load the checkpoint from Wandb.')
    parser.add_argument('--annot-dir', type=str, default='annotations', help='Root directory where COCO annotations are')
    return parser.parse_args()

def main():
    
    args = get_args()

    if args.use_wandb:
        # wandb.login()
        wandb.init(project='age-classifier', resume='allow', id=args.wandb_run_id)

    if args.wandb_run_id:
        artifact = wandb.use_artifact(f'age-classifier:{args.wandb_run_id}')
        artifact_dir = artifact.download()
        checkpoint_path = os.path.join(artifact_dir, 'checkpoints', os.path.basename(args.checkpoint))
    else:
        checkpoint_path = args.checkpoint

    data = ArtportalenDataModule(data_dir=args.dataset, batch_size=args.batch_size, size=args.img_size)
    data.setup_from_coco(args.annot_dir + '/modified_val_annotations.json', args.annot_dir + '/modified_val_annotations.json')
    dataloader = data.val_dataloader()

    model = SimpleModel(
        model_name=args.model_name, pretrained=True, num_classes=data.num_classes, outdir=args.outdir
    )
    model.load_state_dict(torch.load(checkpoint_path)["state_dict"])
    model.eval()


    save_gradcam(model, dataloader, args.outdir)

    print(f"GradCAM visualizations saved to {args.outdir}")

if __name__ == '__main__':
    main()
