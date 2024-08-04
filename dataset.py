from pycocotools.coco import COCO
import numpy as np
import os
import math
import pandas as pd
import json
from IPython.display import display
from PIL import Image
import cv2

from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision import transforms
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
    Pad
)
from torchvision.transforms.functional import resize, pad
import pytorch_lightning as pl


class ArtportalenDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=8, size=256, mean=0.5, std=0.5):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.size = size
        self.mean = mean
        self.std = std

        # transformations
        self.train_transforms = Compose([
            # Resize(self.size),
            # Pad((self.size - 1, self.size - 1), padding_mode='constant'),
            RandomHorizontalFlip(),
            ToTensor(),
            Normalize((mean, mean, mean), (std, std, std)),
        ])

        self.val_transforms = Compose([
            # Resize(self.size),
            # Pad((self.size - 1, self.size - 1), padding_mode='constant'),
            ToTensor(),
            Normalize((mean, mean, mean), (std, std, std)),
        ])

    def prepare_data(self):
        # download, split, etc.
        pass

    def setup_for_training(self, image_dir):
        pass


    def setup_from_coco(self, train_annot, val_annot, stage=None):
        # Load COCO annotations
        with open(train_annot, 'r') as f:
            train_data = json.load(f)
        with open(val_annot, 'r') as f:
            val_data = json.load(f)

        # Initialize COCO objects
        train_coco = COCO(train_annot)
        val_coco = COCO(val_annot)

        # Convert annotations to DataFrame
        train_df = self.coco_to_dataframe(train_coco)
        val_df = self.coco_to_dataframe(val_coco)

        # Printing sizes
        print(f"Train: {len(train_df)} Val: {len(val_df)}")

        self.train_dataset = EagleDataset(train_df, self.data_dir, self.train_transforms)
        self.val_dataset = EagleDataset(val_df, self.data_dir, self.val_transforms)

        # Check number of classes
        unique_classes = train_df['category_id'].unique()
        print(f"Unique classes in dataset: {unique_classes}")
        self.num_classes = len(unique_classes)
        print(f"Number of classes: {self.num_classes}")

    def coco_to_dataframe(self, coco):
        # Create a DataFrame from COCO annotations
        data = []
        for ann in coco.anns.values():
            img_info = coco.loadImgs(ann['image_id'])[0]
            data.append({
                'image_id': ann['image_id'],
                'file_name': img_info['file_name'] + '.jpg',
                'height': img_info['height'],
                'width': img_info['width'],
                'category_id': ann['category_id'],
                'bbox': ann['bbox'],
                'area': ann['area'],
                'iscrowd': ann['iscrowd'],
                'segmentation': ann['segmentation'],

            })
        return pd.DataFrame(data)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2)


class EagleDataset(Dataset):
    def __init__(self, dataframe, data_dir, transform=None, size=256):
        self.dataframe = dataframe
        self.data_dir = data_dir
        self.transform = transform
        self.size = size

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_info = self.dataframe.iloc[idx]
        img_path = os.path.join(self.data_dir, img_info['file_name'])
        label = img_info['category_id'] - 1 

        image = Image.open(img_path).convert("RGB")

        # Extract bounding box and crop the image
        bbox = img_info['bbox']
        x_min = math.floor(bbox[0])
        y_min = math.floor(bbox[1])
        w = math.ceil(bbox[2])
        h = math.ceil(bbox[3])
        bbox = [x_min, y_min, w, h]

        segmentation = img_info['segmentation']
        mask = self.create_mask(image.size, segmentation)

        # masked_image = np.array(cropped_image) * np.expand_dims(cropped_mask, axis=2)
        masked_image = np.array(image) * np.expand_dims(mask, axis=2)
        masked_image = Image.fromarray(masked_image.astype('uint8'))

        # Crop the image and the mask to the bounding box
        masked_image = masked_image.crop((x_min, y_min, x_min + w, y_min + h))

        # Resize and pad the image
        masked_image = self.resize_and_pad(masked_image, self.size)

        if self.transform:
            masked_image = self.transform(masked_image)

        return masked_image, label

    def create_mask(self, image_size, segmentation):
        mask = np.zeros(image_size[::-1], dtype=np.uint8)
        for seg in segmentation:
            poly = np.array(seg).reshape((len(seg) // 2, 2)).astype(np.int32)
            cv2.fillPoly(mask, [poly], 1)
        return mask

    def resize_and_pad(self, image, size):
        # Resize maintaining aspect ratio
        original_width, original_height = image.size
        aspect_ratio = original_width / original_height

        if original_width > original_height:
            new_width = size
            new_height = math.ceil(new_width / aspect_ratio)
        else:
            new_height = size
            new_width = math.ceil(new_height * aspect_ratio)

        resized_image = resize(image, (new_height, new_width))

        # Calculate padding
        pad_width = size - new_width
        pad_height = size - new_height
        padding = (pad_width // 2, pad_height // 2, pad_width - (pad_width // 2), pad_height - (pad_height // 2))

        # Pad the image to make it square
        padded_image = pad(resized_image, padding, fill=0, padding_mode='constant')

        return padded_image
    

def unnormalize(x, mean, std):
    if x.dim() == 3:  # Ensure the tensor has 3 dimensions
        unnormalized_x = x.clone()
        for t, m, s in zip(unnormalized_x, mean, std):
            t.mul_(s).add_(m)
        return unnormalized_x
    else:
        raise ValueError(f"Expected input tensor to have 3 dimensions, but got {x.dim()} dimensions.")
