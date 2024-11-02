from torchvision import datasets, transforms
from base import BaseDataLoader
from torch.utils.data import Dataset
import os
import cv2
import numpy as np
from utils import generate_rotated_image
import torch.nn.functional as F
import torch



class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class MITIndoorDataset(Dataset):
    def __init__(self, data_dir, transform=None, input_shape=(224, 224), crop_center=True, crop_largest_rect=True):
        self.data_dir = data_dir
        self.transform = transform
        # load all image files, sorting them to
        self.imgs = os.listdir(self.data_dir)
        self.imgs = [os.path.join(self.data_dir, img) for img in self.imgs]
        self.images = sorted(self.imgs)
        self.input_shape = input_shape
        self.crop_center = crop_center
        self.crop_largest_rect = crop_largest_rect

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        image = cv2.imread(self.images[idx], cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rotation_angle = np.random.randint(360)
        rotated_image = generate_rotated_image(
                image,
                rotation_angle,
                size=self.input_shape[:2],
                crop_center=self.crop_center,
                crop_largest_rect=self.crop_largest_rect
            )
        if rotated_image.ndim == 2:
            rotated_image = np.expand_dims(rotated_image, axis=2)
        if self.transform:
            rotated_image = self.transform(rotated_image)
        return rotated_image, rotation_angle 

def normalize_transform(pretrained):
    if pretrained: # Normalization for pre-trained weights.
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
            )
    else: # Normalization when training from scratch.
        normalize = transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    return normalize

class MITIndoorDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, pretrained=True, shuffle=True, validation_split=0.0, num_workers=1, input_shape=(224, 224)):
        trsfm = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(input_shape),
            transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
            transforms.ToTensor(),
            normalize_transform(pretrained),
        ])
        self.data_dir = data_dir
        self.dataset = MITIndoorDataset(self.data_dir, transform=trsfm, input_shape=input_shape)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)