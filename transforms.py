import random
import cv2
from PIL import Image
import torch
from torchvision import transforms
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
    Pad,
    RandomRotation,
    ColorJitter,
)
import torchvision.transforms.functional as F
from torchvision.transforms.functional import resize, pad


mean=(0.5, 0.5, 0.5)
std=(0.5, 0.5, 0.5)

common_transforms = Compose([
    RandomHorizontalFlip(),
    RandomRotation(degrees=15)
])
rgb_transforms = Compose([
    ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    ToTensor(),
    Normalize(mean=mean, std=std)
])
val_transforms = Compose([
    # Resize(self.size),
    # Pad((self.size - 1, self.size - 1), padding_mode='constant'),
    ToTensor(),
    Normalize(mean=mean, std=std)
])
train_transforms = Compose([
    # Resize(self.size),
    # Pad((self.size - 1, self.size - 1), padding_mode='constant'),
    common_transforms,
    rgb_transforms
])

class ValTransforms:
    def __init__(self, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), skeleton=False):
        self.skeleton = skeleton
        self.rgb_transforms = Compose([
            ToTensor(),
            Normalize(mean=mean, std=std)
        ])

    def __call__(self, image, skeleton_channel=None):
        # Apply basic transformations to the image (RGB)
        image = self.rgb_transforms(image)

        # Apply to tensor to the skeleton channel
        skeleton_channel = torch.tensor(skeleton_channel, dtype=torch.float32)
        
        skeleton_channel = skeleton_channel.unsqueeze(0)
        concatenated = torch.cat((image, skeleton_channel), dim=0)
     
        return concatenated

class SynchTransforms:
    def __init__(self, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), degrees=15):
        self.mean = mean
        self.std = std
        self.degrees = degrees
        self.rgb_transforms = Compose([
            ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            ToTensor(),
            Normalize(mean=mean, std=std),
        ])
    
    def __call__(self, rgb_img, skeleton_channel=None):
        # Apply the same random horizontal flip
        if random.random() < 0.5:
            rgb_img = rgb_img.transpose(Image.FLIP_LEFT_RIGHT)  # Flip RGB
            skeleton_channel = cv2.flip(skeleton_channel, 1)  # Flip skeleton

        # Apply the same random rotation
        angle = random.uniform(-self.degrees, self.degrees)
        rgb_img = rgb_img.rotate(angle)
        skeleton_channel = self.rotate_image(skeleton_channel, angle)

        # Apply additional transforms to the RGB image
        rgb_img = self.rgb_transforms(rgb_img)

        # Convert the skeleton channel to tensor after transformations
        # skeleton_channel = torch.tensor(skeleton_channel, dtype=torch.float32).unsqueeze(0)  # Add a channel dimension [1, H, W]
        skeleton_channel = torch.tensor(skeleton_channel, dtype=torch.float32)
        
        skeleton_channel = skeleton_channel.unsqueeze(0)
        concatenated = torch.cat((rgb_img, skeleton_channel), dim=0)

        return concatenated

    def rotate_image(self, image, angle):
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height), flags=cv2.INTER_LINEAR)
        return rotated_image


class RGBTransforms:
    def __init__(self, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), degrees=15):
        self.transforms = Compose([
            RandomHorizontalFlip(),
            RandomRotation(degrees=degrees),
            ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            ToTensor(),
            Normalize(mean=mean, std=std),
        ])
    def __call__(self, img):
        return self.transforms(img)
    

class SkelTransforms:
    def __init__(self, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), degrees=15):
        self.mean = mean
        self.std = std
        self.degrees = degrees
        
    def __call__(self, img):
        img = self.random_horizontal_flip(img, p=0.5)
        img = self.random_rotation(img, degrees=self.degrees)
        tensor = torch.tensor(img, dtype=torch.float32)

        return tensor
    
    def random_horizontal_flip(self, image, p=0.5):
        if random.random() < p:
            return cv2.flip(image, 1)  # Flip horizontally
        return image
    
    def random_rotation(self, image, degrees=15):
        height, width = image.shape[:2]
        rotation_angle = random.uniform(-degrees, degrees)  # Random angle between -degrees and +degrees
        center = (width // 2, height // 2)  # Center of the image

        # Get the rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)

        # Perform the rotation
        rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height), flags=cv2.INTER_LINEAR)

        return rotated_image

class RGBSkelTransforms:
    """
    Custom transform class for applying transformations to both the RGB image and the skeleton channel.

    Args:
        rgb_transforms (callable): Transformations specific to the RGB channels.
        common_transforms (callable, optional): Spatial transformations applied to both RGB and skeleton.
    """
    def __init__(self, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
        # Color transforms only for RGB
        rgb_transforms = []
        rgb_transforms.append(ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05))
        rgb_transforms.append(ToTensor())
        rgb_transforms.append(Normalize(mean=mean, std=std))

        self.rgb_transforms = Compose(rgb_transforms)

        # Spatial transforms for both RGB and skeleton
        self.common_transforms = Compose([
            RandomHorizontalFlip(),
            RandomRotation(degrees=15)
        ])

    def __call__(self, img):
        """
        Applies the transformations to both the RGB and skeleton channels.

        Args:
            img (torch.Tensor): Input image tensor with 4 channels (RGB + skeleton).

        Returns:
            torch.Tensor: Transformed image tensor.
        """
        # img is expected to have 4 channels: RGB (3 channels) + Skeleton (1 channel)
        rgb_img = img[:3, :, :]  # First 3 channels (RGB)
        skeleton_img = img[3:, :, :]  # 4th channel (Skeleton)

        # Apply spatial transforms (flip and rotate) to both
        rgb_img = self.common_transforms(rgb_img)
        skeleton_img = self.common_transforms(skeleton_img)

        # Apply RGB-specific transforms (color jitter, normalize) only to the RGB part
        rgb_img = self.rgb_transforms(rgb_img)

        # Concatenate the RGB and skeleton channels back together
        img = torch.cat((rgb_img, skeleton_img), dim=0)
        
        return img

        # Split the RGB (first 3 channels) and skeleton (4th channel)
        rgb_img = img.convert("RGB")  # Convert PIL Image to RGB
        
        # Apply common transforms (flip, rotate) on both RGB and skeleton
        if self.common_transforms:
            rgb_img = self.common_transforms(rgb_img)
        
        # Apply RGB-specific transforms (like ColorJitter)
        if self.rgb_transforms:
            rgb_img = self.rgb_transforms(rgb_img)

        # Convert RGB image to tensor
        rgb_tensor = self.to_tensor(rgb_img)

        # Check if skeleton channel exists (assuming RGBA, where 4th channel is skeleton)
        skeleton_tensor = None
        if img.mode == 'RGBA':
            skeleton_img = img.split()[3]  # Extract alpha channel as skeleton
            skeleton_tensor = self.to_tensor(skeleton_img)

        # Apply normalization to RGB tensor
        rgb_tensor = self.normalize(rgb_tensor)

        # If skeleton exists, concatenate it as the 4th channel
        if skeleton_tensor is not None:
            img_tensor = torch.cat((rgb_tensor, skeleton_tensor.unsqueeze(0)), dim=0)  # Add channel dim for skeleton
        else:
            img_tensor = rgb_tensor
        
        return img_tensor
        
        # # Split the RGB and skeleton channels
        # rgb_img = img[:3, :, :]  # First 3 channels (RGB)
        # skeleton_img = img[3:, :, :]  # 4th channel (skeleton)

        # # Apply spatial transforms (e.g., flip, rotate) to both
        # if self.common_transforms:
        #     rgb_img = self.common_transforms(rgb_img)
        #     skeleton_img = self.common_transforms(skeleton_img)

        # # Apply RGB-specific transforms (e.g., ColorJitter, Normalize)
        # rgb_img = self.rgb_transforms(rgb_img)
        
        # # Concatenate the RGB and skeleton channels back
        # img = torch.cat((rgb_img, skeleton_img), dim=0)
        
        # return img
