import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F
from typing import Callable, Sequence, Union, List, Dict
import math


class WeightedMSELoss(nn.Module):
    """Weighted Mean Squared Error Loss.
    """
    def __init__(self):
        super().__init__()
        # Define weights for each entry in the 12-dimensional vector
        self.weights = torch.tensor([0.4/4, 0.4/4, 0.4/4, 0.4/4, 0.4/4, 0.4/4, 0.4/4, 0.4/4, 0.2/4, 0.2/4, 0.2/4, 0.2/4])

    def forward(self, output, target):
        """
        Args:
            output (torch.Tensor) (N, 12): The model output.
            target (torch.Tensor) (N, 12): The target values.
        Returns:
            loss (torch.Tensor) (0): The weighted MSE loss.
        """
        if self.weights.device != output.device:
            self.weights = self.weights.to(output.device)

        # Calculate the weighted MSE loss
        loss = self.weights * (output - target) ** 2
        return loss.mean()
