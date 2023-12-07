import torch
import random
import numpy as np
from torchvision.utils import make_grid
from .base_logger import BaseLogger
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
H = 512
W = 1024

class VPClusterLogger(BaseLogger):
    def _process_batch(self, images, outputs, labels):
        """
        Process a batch to create images with viewport centers on a black background.
        """
        processed_pred_images = []
        processed_gt_images = []

        for output, label in zip(outputs, labels):
            # Create black background images
            pred_img = Image.new('RGB', (W, H), color='black')
            gt_img = Image.new('RGB', (W, H), color='black')

            # Make the boundary of the image white
            draw_pred = ImageDraw.Draw(pred_img)
            draw_gt = ImageDraw.Draw(gt_img)
            draw_pred.rectangle((0, 0, W-1, H-1), outline=(255, 255, 255))
            draw_gt.rectangle((0, 0, W-1, H-1), outline=(255, 255, 255))


            # Draw predicted and ground truth viewport centers
            self._draw_viewport_centers(draw_pred, output[:8], (255, 255, 255))  # White for prediction
            self._draw_viewport_centers(draw_gt, label[:8], (255, 255, 255))    # White for ground truth

            # Convert back to tensor and add to list
            processed_pred_img = torch.tensor(np.array(pred_img).transpose(2, 0, 1))
            processed_gt_img = torch.tensor(np.array(gt_img).transpose(2, 0, 1))
            
            processed_pred_images.append(processed_pred_img)
            processed_gt_images.append(processed_gt_img)

            # put int gpu
        return torch.stack(processed_pred_images).cuda(), torch.stack(processed_gt_images).cuda()

    def _add_images(self, epoch, train_batch, train_output, valid_batch, valid_output):
        """Plot the visualization results.
        Args:
            epoch (int): The number of trained epochs.
            train_batch (dict): The training batch.
            train_output (torch.Tensor): The training output.
            valid_batch (dict): The validation batch.
            valid_output (torch.Tensor): The validation output.
        """
        # Get original images
        train_img = make_grid(train_batch['image'], nrow=1, normalize=True, scale_each=True, pad_value=1)
        valid_img = make_grid(valid_batch['image'], nrow=1, normalize=True, scale_each=True, pad_value=1)

        # Process batches for predicted and ground truth viewport centers
        train_pred_images, train_gt_images = self._process_batch(train_batch['image'], train_output, train_batch['label'])
        valid_pred_images, valid_gt_images = self._process_batch(valid_batch['image'], valid_output, valid_batch['label'])

        # Combine images into grids
        # images must all be in the same device

        train_grid = torch.cat((train_img, make_grid(train_pred_images, nrow=1), make_grid(train_gt_images, nrow=1)), dim=-1)
        valid_grid = torch.cat((valid_img, make_grid(valid_pred_images, nrow=1), make_grid(valid_gt_images, nrow=1)), dim=-1)

        # Add images to TensorBoard
        self.writer.add_image('train', train_grid, epoch)
        self.writer.add_image('valid', valid_grid, epoch)

    # def _add_images(self, epoch, train_batch, train_output, valid_batch, valid_output):
    #     """Plot the visualization results.
    #     Args:
    #         epoch (int): The number of trained epochs.
    #         train_batch (dict): The training batch.
    #         train_output (torch.Tensor): The training output.
    #         valid_batch (dict): The validation batch.
    #         valid_output (torch.Tensor): The validation output.
    #     """


    #     num_classes = train_output.size(1)
    #     train_img = make_grid(train_batch['image'], nrow=1, normalize=True, scale_each=True, pad_value=1)
    #     valid_img = make_grid(valid_batch['image'], nrow=1, normalize=True, scale_each=True, pad_value=1)

    #     train_draw_images = self._process_batch(train_batch['image'], train_output, train_batch['label'])
    #     valid_draw_images = self._process_batch(valid_batch['image'], valid_output, valid_batch['label'])
    #     train_grid = torch.cat((train_img, train_label, train_pred), dim=-1)
    #     valid_grid = torch.cat((valid_img, valid_label, valid_pred), dim=-1)
    #     train_grid = train_img
    #     valid_grid = valid_img
    #     self.writer.add_image('train', train_grid, epoch)
    #     self.writer.add_image('valid', valid_grid, epoch)

    def _draw_viewport_centers(self, draw, viewport_data, color):
        """
        Draw viewport centers on the image.
        viewport_data: (8,) array of four (pitch, yaw) pairs
        """
        radius = 25
        for i in range(0, len(viewport_data), 2):
            pitch, yaw = viewport_data[i], viewport_data[i + 1]
            # print(f'pitch: {pitch}, yaw: {yaw}')
            # scale pitch and yaw to image size, where pitch ranges from -90 to 90 and yaw ranges from -180 to 180
            # pitch = pitch * 180 - 90
            # yaw = yaw * 360 - 180 
            
            # Normalize pitch [-90, 90] to [0, 1] and yaw [-180, 180] to [0, 1]
            pitch = (pitch + 90) / 180
            yaw = (yaw + 180) / 360

            x = int(yaw * W)
            y = int(pitch * H)

            draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=color, outline=color)
            radius -= 5


    # def _draw_viewport_centers(self, draw, viewport_data, color):
    #     """
    #     Draw viewport centers on the image.
    #     """
    #     H, W = 360, 640  # Assuming image dimensions are 360x640
    #     for i in range(0, len(viewport_data), 2):
    #         pitch, yaw = viewport_data[i], viewport_data[i + 1]
    #         x = int(((yaw * 0.5 + 0.5) * W))  # Scale yaw to image width
    #         y = int(((pitch * 0.5 + 0.5) * H))  # Scale pitch to image height
    #         draw.ellipse((x-5, y-5, x+5, y+5), fill=color, outline=color)