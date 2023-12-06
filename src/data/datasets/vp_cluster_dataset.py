import csv
import glob
import torch
import numpy as np
import os
from PIL import Image 
import cv2
import random

from src.data.datasets.base_dataset import BaseDataset
# from src.data.transforms import compose

from torchvision import transforms

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

normalize = transforms.Normalize(MEAN, STD)


class VPClusterDataset(BaseDataset):
    def __init__(self, train_preprocessings, valid_preprocessings, transforms, **kwargs):
        super().__init__(**kwargs)
        self.csv_dir = self.data_dir / 'ClusterData'
        self.frame_dir = self.data_dir / 'Frames'
        # self.train_preprocessings = compose(train_preprocessings)
        # self.valid_preprocessings = compose(valid_preprocessings)
        # self.transforms = compose(transforms)

        self.train_data = []
        self.val_data = []

        for i in range(1, 21):
            formatted_number = str(i).zfill(2)
            csv_path = self.csv_dir / f'Video{formatted_number}.csv'
            with open(csv_path, "r") as f:
                rows = csv.reader(f)
                for c, row in enumerate(rows):
                    row = [float(x) for x in row]
      
                    # extract the pitch and yaw into [(pitch, yaw), (pitch, yaw), ...]
                    pairs = [(row[0], row[1]), (row[2], row[3]), (row[4], row[5]), (row[6], row[7])]
                    probs = [row[8], row[9], row[10], row[11]]
                    # sort the rows by the probability
                    pairs = [x for _, x in sorted(zip(probs, pairs), reverse=True)]
                    # sort the probs
                    probs = sorted(probs, reverse=True)
                    # flatten the list
                    row = []
                    for pair in pairs:
                        row.extend(pair)
                    row = row + probs
            
             
                    im_path = self.frame_dir / f'Video{formatted_number}' / f'{c}.jpg'
                    self.train_data.append((im_path, row))


        for i in range(21, 25):
            formatted_number = str(i).zfill(2)
            csv_path = self.csv_dir / f'Video{formatted_number}.csv'
            with open(csv_path, "r") as f:
                rows = csv.reader(f)
                for c, row in enumerate(rows):
                    row = [float(x) for x in row]

                    # extract the pitch and yaw into [(pitch, yaw), (pitch, yaw), ...]
                    pairs = [(row[0], row[1]), (row[2], row[3]), (row[4], row[5]), (row[6], row[7])]
                    probs = [row[8], row[9], row[10], row[11]]
                    # sort the rows by the probability
                    pairs = [x for _, x in sorted(zip(probs, pairs), reverse=True)]
                    # sort the probs
                    probs = sorted(probs, reverse=True)
                    # flatten the list
                    row = []
                    for pair in pairs:
                        row.extend(pair)
                    row = row + probs

                    im_path = self.frame_dir / f'Video{formatted_number}' / f'{c}.jpg'
                    self.val_data.append((im_path, row))

        self.data_paths = self.train_data if self.type == 'train' else self.val_data

        # random sample 10% of the data
        sample_rate = 0.5
        self.data_paths = random.sample(self.data_paths, int(len(self.data_paths) * 0.5))

        
    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, index):
        image_path, row = self.data_paths[index]
        # print()
        # print(image_path)
        # print(row)
        # print()

        # Load the image
        image = Image.open(image_path)
        image = np.array(image).astype(np.float32)

        # resize the image to 640x320
        image = cv2.resize(image, (1024, 512))

        # Check if image is grayscale and convert it to RGB
        if len(image.shape) == 2:
            image = np.stack((image,)*3, axis=-1)

        # Normalize the image
        image = transforms.functional.to_tensor(image)
        image = normalize(image)

        # Process the label
        label = np.array(row)

        # # Normalize pitch [-90, 90] to [0, 1] and yaw [-180, 180] to [0, 1]
        # label[:8:2] = (label[:8:2] + 90) / 180  # Normalize pitch
        # label[1:8:2] = (label[1:8:2] + 180) / 360  # Normalize yaw

        # print(f'label in dataset: {label}')

        return {"image": image, "label": torch.from_numpy(label)}
