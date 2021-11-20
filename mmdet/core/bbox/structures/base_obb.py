import numpy as np
import torch
from abc import abstractmethod


class BaseInstanceOBBs(object):
    def __init__(self, tensor, box_dim=5):
        if isinstance(tensor, torch.Tensor):
            device = tensor.device
        else:
            device = torch.device('cpu')
        tensor = torch.as_tensor(tensor, dtype=torch.float32, device=device)
        if tensor.numel() == 0:
            # Use reshape, so we don't end up creating a new tensor that
            # does not depend on the inputs (and consequently confuses jit)
            tensor = tensor.reshape((0, box_dim)).to(
                dtype=torch.float32, device=device)
        assert tensor.dim() == 2 and tensor.size(-1) == box_dim, tensor.size()
        
        self.box_dim = box_dim
        self.tensor = tensor

    @property
    def center(self):
        """torch.Tensor: A tensor with center of each obb."""
        return self.tensor[:, :2]

    @property
    def angle(self):
        """torch.Tensor: a tensor with 8 corners of each obb."""
        return self.tensor[:, -1]

    @abstractmethod
    def rotate(self, angles, axis=0):
        """Calculate whether the points are in any of the boxes.

        Args:
            angles (float): Rotation angles.
            axis (int): The axis to rotate the boxes.
        """
        pass

    @abstractmethod
    def flip(self, bev_direction='horizontal'):
        """Flip the boxes in BEV along given BEV direction."""
        pass

    @abstractmethod
    def scale(self, scale_factor):
        """Scale the box with horizontal and vertical scaling factors.

        Args:
            scale_factors (float): Scale factors to scale the boxes.
        """
        pass

    @abstractmethod
    def convert_to(self, dst, rt_mat=None):
        """Convert self to ``dst`` mode.

        Args:
            dst (:obj:`BoxMode`): The target Box mode.
            rt_mat (np.ndarray | torch.Tensor): The rotation and translation
                matrix between different coordinates. Defaults to None.
                The conversion from `src` coordinates to `dst` coordinates
                usually comes along the change of sensors, e.g., from camera
                to LiDAR. This requires a transformation matrix.

        Returns:
            :obj:`BaseInstance3DBoxes`: The converted box of the same type \
                in the `dst` mode.
        """
        pass

    def nonempty(self, threshold: float = 0.0):
        """Find boxes that are non-empty.

        A box is considered empty,
        if either of its side is no larger than threshold.

        Args:
            threshold (float): The threshold of minimal sizes.

        Returns:
            torch.Tensor: A binary vector which represents whether each \
                box is empty (False) or non-empty (True).
        """
        box = self.tensor
        size_w = box[..., 2]
        size_h = box[..., 3]
        keep = ((size_w > threshold) & (size_h > threshold))
        
        return keep

    def __getitem__(self, item):
        """
        Note:
            The following usage are allowed:
            1. `new_boxes = boxes[3]`:
                return a `Boxes` that contains only one box.
            2. `new_boxes = boxes[2:10]`:
                return a slice of boxes.
            3. `new_boxes = boxes[vector]`:
                where vector is a torch.BoolTensor with `length = len(boxes)`.
                Nonzero elements in the vector will be selected.
            Note that the returned Boxes might share storage with this Boxes,
            subject to Pytorch's indexing semantics.

        Returns:
            :obj:`BaseInstances3DBoxes`: A new object of  \
                :class:`BaseInstances3DBoxes` after indexing.
        """
        original_type = type(self)
        if isinstance(item, int):
            return original_type(
                self.tensor[item].view(1, -1),
                box_dim=self.box_dim)
        b = self.tensor[item]
        assert b.dim() == 2, \
            f'Indexing on Boxes with {item} failed to return a matrix!'
        return original_type(b, box_dim=self.box_dim)

    def __len__(self):
        """int: Number of boxes in the current object."""
        return self.tensor.shape[0]

    def __repr__(self):
        """str: Return a strings that describes the object."""
        return self.__class__.__name__ + '(\n    ' + str(self.tensor) + ')'

    def to(self, device):
        """Convert current boxes to a specific device.

        Args:
            device (str | :obj:`torch.device`): The name of the device.

        Returns:
            :obj:`BaseInstance3DBoxes`: A new boxes object on the \
                specific device.
        """
        original_type = type(self)
        return original_type(
            self.tensor.to(device),
            box_dim=self.box_dim)

    def clone(self):
        """Clone the Boxes.

        Returns:
            :obj:`BaseInstance3DBoxes`: Box object with the same properties \
                as self.
        """
        original_type = type(self)
        return original_type(
            self.tensor.clone(), box_dim=self.box_dim)

    @property
    def device(self):
        """str: The device of the boxes are on."""
        return self.tensor.device

    def __iter__(self):
        """Yield a box as a Tensor of shape (4,) at a time.

        Returns:
            torch.Tensor: A box of shape (4,).
        """
        yield from self.tensor

    @classmethod
    def overlaps(cls, boxes1, boxes2, mode='iou'):
        """Calculate 3D overlaps of two boxes.

        Note:
            This function calculates the overlaps between ``boxes1`` and
            ``boxes2``, ``boxes1`` and ``boxes2`` should be in the same type.

        Args:
            boxes1 (:obj:`BaseInstanceBoxes`): Boxes 1 contain N boxes.
            boxes2 (:obj:`BaseInstanceBoxes`): Boxes 2 contain M boxes.
            mode (str, optional): Mode of iou calculation. Defaults to 'iou'.

        Returns:
            torch.Tensor: Calculated iou of boxes' heights.
        """
        pass

    def new_box(self, data):
        """Create a new box object with data.

        The new box and its tensor has the similar properties \
            as self and self.tensor, respectively.

        Args:
            data (torch.Tensor | numpy.array | list): Data to be copied.

        Returns:
            :obj:`BaseInstance3DBoxes`: A new bbox object with ``data``, \
                the object's other properties are similar to ``self``.
        """
        new_tensor = self.tensor.new_tensor(data) \
            if not isinstance(data, torch.Tensor) else data.to(self.device)
        original_type = type(self)
        return original_type(
            new_tensor, box_dim=self.box_dim)
