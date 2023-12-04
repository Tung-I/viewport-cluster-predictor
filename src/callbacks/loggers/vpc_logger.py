import torch
import random
import numpy as np
from torchvision.utils import make_grid
from .base_logger import BaseLogger
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

class VPClusterLogger(BaseLogger):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _add_images(self, epoch, train_batch, train_output, valid_batch, valid_output):
        """
        Plot the visualization results with viewport centers.
        """
        # Process a single batch (you can modify this to process more)
        train_images = self._process_batch(train_batch['image'], train_output, train_batch['label'])
        valid_images = self._process_batch(valid_batch['image'], valid_output, valid_batch['label'])

        # Convert to grid and add to TensorBoard
        self.writer.add_image('train', make_grid(train_images, nrow=1), epoch)
        self.writer.add_image('valid', make_grid(valid_images, nrow=1), epoch)

    def _process_batch(self, images, outputs, labels):
        """
        Process a batch to overlay viewport centers on images.
        """
        processed_images = []
        for img, output, label in zip(images, outputs, labels):
            # Convert to PIL Image for drawing
            pil_img = Image.fromarray(img.numpy().transpose(1, 2, 0).astype(np.uint8))
            draw = ImageDraw.Draw(pil_img)

            # Draw predicted and ground truth viewport centers
            self._draw_viewport_centers(draw, output[:8], (255, 0, 0))  # Red for prediction
            self._draw_viewport_centers(draw, label[:8], (0, 255, 0))   # Green for ground truth

            # Convert back to tensor and add to list
            processed_img = torch.tensor(np.array(pil_img).transpose(2, 0, 1))
            processed_images.append(processed_img)

        return torch.stack(processed_images)

    def _draw_viewport_centers(self, draw, viewport_data, color):
        """
        Draw viewport centers on the image.
        """
        H, W = 360, 640  # Assuming image dimensions are 360x640
        for i in range(0, len(viewport_data), 2):
            pitch, yaw = viewport_data[i], viewport_data[i + 1]
            x = int(((yaw * 0.5 + 0.5) * W))  # Scale yaw to image width
            y = int(((pitch * 0.5 + 0.5) * H))  # Scale pitch to image height
            draw.ellipse((x-5, y-5, x+5, y+5), fill=color, outline=color)
