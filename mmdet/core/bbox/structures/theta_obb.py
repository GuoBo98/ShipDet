import numpy as np
import torch

from .base_obb import BaseInstanceOBBs


class ThetaOBBs(BaseInstanceOBBs):
    @property
    def center(self):
        """torch.Tensor: A tensor with center of each obb."""
        return self.tensor[:, :2]

    @property
    def angle(self):
        """torch.Tensor: a tensor with 8 corners of each obb."""
        return self.tensor[:, -1]

    def rotate(self, angles, axis=0):
        """Calculate whether the points are in any of the boxes.

        Args:
            angles (float): Rotation angles.
            axis (int): The axis to rotate the boxes.
        """
        pass

    def flip(self, bev_direction='horizontal'):
        """Flip the boxes in BEV along given BEV direction."""
        pass

    def scale(self, scale_factor):
        """Scale the box with horizontal and vertical scaling factors.

        Args:
            scale_factors (float): Scale factors to scale the boxes.
        """
        pass

    def convert_to(self, dst):
        """Convert self to ``dst`` mode.

        Args:
            dst (:obj:`OBBMode`): The target OBB mode.

        Returns:
            :obj:`BaseInstanceOBBs`: The converted box of the same type \
                in the `dst` mode.
        """
        pass

