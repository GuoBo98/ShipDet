from mmdet.core.mask.structures import BitmapMasks


class MaskOBBs(BitmapMasks):
    @property
    def center(self):
        """torch.Tensor: A tensor with center of each obb."""
        pass

    @property
    def angle(self):
        """torch.Tensor: a tensor with 8 corners of each obb."""
        pass

    def rotate(self, angles, axis=0):
        """Calculate whether the points are in any of the boxes.

        Args:
            angles (float): Rotation angles.
            axis (int): The axis to rotate the boxes.
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