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


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train classifier.')
    parser.add_argument(
        '--config', type=str, required=True, default="./config.yaml", help='Path to config yaml file'
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


def get_trainer(config) -> Trainer:
    callbacks = get_basic_callbacks(checkpoint_interval=int(config['save_interval']))
    accelerator, devices, strategy = get_gpu_settings(config['gpu_ids'], config['n_gpu'])

    if config['use_wandb']:
        wandb_logger = WandbLogger(project=config['project_name'], log_model=True)
    else:
        wandb_logger = None

    trainer_args = {
        'max_epochs': config['epochs'],
        'callbacks': callbacks,
        'default_root_dir': config['outdir'],
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
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    seed_everything(config['seed'], workers=True)

    data = ArtportalenDataModule(data_dir=config['dataset'], batch_size=config['batch_size'], size=config['img_size'])
    data.setup_from_coco(config['annot_dir'] + '/modified_val_annotations.json', config['annot_dir'] + '/modified_val_annotations.json')

    
    if config['checkpoint']:
        model = SimpleModel(model_name=config['model_name'], pretrained=False, num_classes=data.num_classes, outdir=config['outdir'])
        checkpoint = torch.load(config['checkpoint'])
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model = SimpleModel(model_name=config['model_name'], pretrained=True, num_classes=data.num_classes, outdir=config['outdir'])


    trainer = get_trainer(args)

    print('Args:')
    pprint(args.__dict__)
    print('configuration:')
    pprint(config)

    
    trainer.fit(model, data)