from abc import ABCMeta, abstractmethod

import torch.nn as nn

from mmdet.models.roi_heads.base_roi_head import BaseRoIHead
from mmdet.models.builder import build_shared_head

class BaseROITRANSRoIHead(BaseRoIHead):
    """Base class for RoIHeads."""
    def __init__(self,
                 bbox_roi_extractor=None, #SingleRoIExtractor
                 bbox_head=None,#FC还是
                 mask_roi_extractor=None,
                 rbbox_roi_extractor=None,
                 rbbox_head=None,
                 mask_head=None,
                 shared_head=None,
                 train_cfg=None,
                 test_cfg=None):
        #TODO: Test
        super(BaseRoIHead, self).__init__()
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        if shared_head is not None:
            self.shared_head = build_shared_head(shared_head)

        if bbox_head is not None:
            self.init_bbox_head(bbox_roi_extractor, bbox_head, rbbox_roi_extractor, rbbox_head)

        if mask_head is not None:
            self.init_mask_head(mask_roi_extractor, mask_head)

        self.init_assigner_sampler()

    @property
    def with_rbbox(self):
        """bool: whether the RRoI head contains a `rbbox_head`"""
        return hasattr(self, 'rbbox_head') and self.rbbox_head is not None

    @property
    def with_bbox(self):
        return hasattr(self, 'bbox_head') and self.bbox_head is not None

    @property
    def with_shared_head(self):
        return hasattr(self, 'shared_head') and self.shared_head is not None

    @property
    def with_shared_head_rbbox(self):
        return hasattr(self, 'shared_head_rbbox') and self.shared_head_rbbox is not None
