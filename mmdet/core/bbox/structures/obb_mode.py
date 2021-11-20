import numpy as np
import torch
from enum import IntEnum, unique

from .base_obb import BaseInstanceOBBs


@unique
class OBBMode(IntEnum):
    r"""Enum of different ways to represent a box.

    """
    THETAOBB = 0
    POINTOBB = 1
    HOBB = 2
    MASKOBB = 3

    @staticmethod
    def convert(box, src, dst, rt_mat=None):
        """Convert boxes from `src` mode to `dst` mode.

        Args:
            box (tuple | list | np.dnarray |
                torch.Tensor | BaseInstanceOBBs):
                Can be a k-tuple, k-list or an Nxk array/tensor, where k = 7.
            src (:obj:`OBBMode`): The src OBB mode.
            dst (:obj:`OBBMode`): The target OBB mode.

        Returns:
            (tuple | list | np.dnarray | torch.Tensor | BaseInstanceOBBs): \
                The converted box of the same type.
        """
        if src == dst:
            return box

        is_numpy = isinstance(box, np.ndarray)
        is_InstanceOBBs = isinstance(box, BaseInstanceOBBs)
        single_box = isinstance(box, (list, tuple))
        if single_box:
            assert len(box) >= 5, (
                'BoxMode.convert takes either a k-tuple/list or '
                'an Nxk array/tensor, where k >= 7')
            arr = torch.tensor(box)[None, :]
        else:
            # avoid modifying the input box
            if is_numpy:
                arr = torch.from_numpy(np.asarray(box)).clone()
            elif is_InstanceOBBs:
                arr = box.tensor.clone()
            else:
                arr = box.clone()

        # convert box from `src` mode to `dst` mode.
        if src == OBBMode.THETAOBB and dst == OBBMode.POINTOBB:
            pass
        elif src == OBBMode.POINTOBB and dst == OBBMode.THETAOBB:
            pass
        else:
            raise NotImplementedError(
                f'Conversion from OBBMode {src} to {dst} '
                'is not supported yet')