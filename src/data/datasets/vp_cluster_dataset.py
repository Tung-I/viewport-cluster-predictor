import csv
import glob
import torch
import numpy as np
import os
from PIL import Image 

from src.data.datasets.base_dataset import BaseDataset
from src.data.transforms import compose

from torchvision import transforms

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

normalize = transforms.Normalize(MEAN, STD)


class VPClusterDataset(BaseDataset):
    def __init__(self, train_preprocessings, valid_preprocessings, transforms, **kwargs):
        super().__init__(**kwargs)
        self.csv_dir = self.data_dir / 'ClusterData'
        self.frame_dir = self.data_dir / 'Frames'
        self.train_preprocessings = compose(train_preprocessings)
        self.valid_preprocessings = compose(valid_preprocessings)
        self.transforms = compose(transforms)

        self.train_data = []
        self.val_data = []

        for i in range(1, 21):
            formatted_number = str(i).zfill(2)
            csv_path = self.csv_dir / f'Video{formatted_number}.csv'
            with open(csv_path, "r") as f:
                rows = csv.reader(f)
                for c, row in enumerate(rows):
                    im_path = self.frame_dir / f'Video{formatted_number}' / f'{c}.jpg'
                    self.train_data.append((im_path, row))


        for i in range(21, 25):
            formatted_number = str(i).zfill(2)
            csv_path = self.csv_dir / f'Video{formatted_number}.csv'
            with open(csv_path, "r") as f:
                rows = csv.reader(f)
                for c, row in enumerate(rows):
                    label = [float(x) for x in row]
                    im_path = self.frame_dir / f'Video{formatted_number}' / f'{c}.jpg'
                    self.val_data.append((im_path, label))

        
def __len__(self):
    return len(self.data_paths)

def __getitem__(self, index):
    image_path, row = self.data_paths[index]

    # Load the image
    image = Image.open(image_path)
    image = np.array(image).astype(np.float32)

    # Check if image is grayscale and convert it to RGB
    if len(image.shape) == 2:
        image = np.stack((image,)*3, axis=-1)

    # Normalize the image
    image = transforms.functional.to_tensor(image)
    image = normalize(image)

    # Process the label
    label = np.array(row)

    # Normalize pitch [-90, 90] to [0, 1] and yaw [-180, 180] to [0, 1]
    label[:8:2] = (label[:8:2] + 90) / 180  # Normalize pitch
    label[1:8:2] = (label[1:8:2] + 180) / 360  # Normalize yaw

    return {"image": image, "label": torch.from_numpy(label)}
