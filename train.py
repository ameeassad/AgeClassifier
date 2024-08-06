# https://github.com/karasawatakumi/pytorch-image-classification/blob/main/train.py

import argparse
from pprint import pprint
import yaml

import torch
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import seed_everything

from dataset import ArtportalenDataModule
from model import SimpleModel

# Load configuration from YAML file
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train classifier.')
    parser.add_argument(
        '--dataset', '-d', type=str, required=True, help='Root directory of dataset'
    )
    parser.add_argument(
        '--outdir', '-o', type=str, default='results', help='Output directory'
    )
    parser.add_argument(
        '--model-name', '-m', type=str, default='resnet18', help='Model name (timm)'
    )
    parser.add_argument(
        '--img-size', '-i', type=int, default=112, help='Input size of image'
    )
    parser.add_argument(
        '--epochs', '-e', type=int, default=100, help='Number of training epochs'
    )
    parser.add_argument(
        '--save-interval', '-s', type=int, default=10, help='Save interval (epoch)'
    )
    parser.add_argument('--batch-size', '-b', type=int, default=8, help='Batch size')
    parser.add_argument(
        '--num-workers', '-w', type=int, default=12, help='Number of workers'
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        '--gpu-ids', type=int, default=None, nargs='+', help='GPU IDs to use'
    )
    group.add_argument('--n-gpu', type=int, default=None, help='Number of GPUs')
    parser.add_argument('--seed', type=int, default=42, help='Seed')
    parser.add_argument(
        '--annot-dir', type=str, default='annotations', help='Root directory where COCO annotations are'
    )
    parser.add_argument(
        '--checkpoint', type=str, help='Path to checkpoint if loading model'
    )
    args = parser.parse_args()
    return args


def get_basic_callbacks(checkpoint_interval: int = 1) -> list:
    lr_callback = LearningRateMonitor(logging_interval='epoch')
    ckpt_callback = ModelCheckpoint(
        dirpath='checkpoints',
        filename='epoch{epoch:03d}',
        auto_insert_metric_name=False,
        save_top_k=-1,
        every_n_epochs=checkpoint_interval,
        save_last=True,
    )
    early_stop_callback = EarlyStopping(
        monitor=config['early_stopping']['monitor'],  # Monitored metric
        min_delta=config['early_stopping']['min_delta'],      # Minimum change to qualify as an improvement
        patience=config['early_stopping']['patience'],         # Number of epochs with no improvement after which training will be stopped
        verbose=config['early_stopping']['verbose'],
        mode=config['early_stopping']['mode']           # Mode for the monitored metric ('min' or 'max')
    )
    return [ckpt_callback, lr_callback, early_stop_callback]


def get_gpu_settings(
    gpu_ids: list[int], n_gpu: int
) -> tuple[str, int | list[int] | None, str | None]:
    """Get gpu settings for pytorch-lightning trainer:
    https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#trainer-flags

    Args:
        gpu_ids (list[int])
        n_gpu (int)

    Returns:
        tuple[str, int, str]: accelerator, devices, strategy
    """
    if not torch.cuda.is_available():
        return "cpu", None, None

    if gpu_ids is not None:
        devices = gpu_ids
        strategy = "ddp" if len(gpu_ids) > 1 else None
    elif n_gpu is not None:
        # int
        devices = n_gpu
        strategy = "ddp" if n_gpu > 1 else None
    else:
        devices = 1
        strategy = None

    return "gpu", devices, strategy


def get_trainer(args: argparse.Namespace) -> Trainer:
    callbacks = get_basic_callbacks(checkpoint_interval=args.save_interval)
    accelerator, devices, strategy = get_gpu_settings(args.gpu_ids, args.n_gpu)

    if config['use_wandb']:
        wandb_logger = WandbLogger(project=config['project_name'], log_model=True)
    else:
        wandb_logger = None

    trainer_args = {
        'max_epochs': args.epochs,
        'callbacks': callbacks,
        'default_root_dir': args.outdir,
        'accelerator': accelerator,
        'devices': devices,
        'logger': wandb_logger,
        'deterministic': True,
    }

    if strategy is not None:
        trainer_args['strategy'] = strategy

    trainer = Trainer(**trainer_args)
    return trainer


if __name__ == '__main__':
    args = get_args()
    seed_everything(args.seed, workers=True)

    data = ArtportalenDataModule(data_dir=args.dataset, batch_size=args.batch_size, size=args.img_size)
    data.setup_from_coco(args.annot_dir + '/modified_val_annotations.json', args.annot_dir + '/modified_val_annotations.json')

    
    if args.checkpoint:
        model = SimpleModel(model_name=args.model_name, pretrained=False, num_classes=data.num_classes, outdir=args.outdir)
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model = SimpleModel(model_name=args.model_name, pretrained=True, num_classes=data.num_classes, outdir=args.outdir)


    trainer = get_trainer(args)

    print('Args:')
    pprint(args.__dict__)

    
    trainer.fit(model, data)