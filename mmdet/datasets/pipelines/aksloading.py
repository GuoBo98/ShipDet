import os.path as osp

import mmcv
import numpy as np
import pycocotools.mask as maskUtils

from mmdet.core import BitmapMasks, PolygonMasks
from ..builder import PIPELINES
from .loading import LoadAnnotations
@PIPELINES.register_module()
class aksLoadAnnotations(LoadAnnotations):
    def __init__(self,
                 with_bbox=True,
                 with_label=True,
                 with_mask=False,
                 with_seg=False,
                 poly2mask=True,
                 file_client_args=dict(backend='disk')):
        super(AKSHead, self).__init__(with_bbox,with_label,with_mask,with_seg,poly2mask,file_client_args)
    

    def _load_bboxes(self, results):
        """Private function to load bounding box annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded bounding box annotations.
        """

        ann_info = results['ann_info']
        results['tb_gt_bboxes'] = ann_info['tb_bboxes'].copy()

        gt_bboxes_ignore = ann_info.get('bboxes_ignore', None)
        if gt_bboxes_ignore is not None:
            results['gt_bboxes_ignore'] = gt_bboxes_ignore.copy()
            results['bbox_fields'].append('gt_bboxes_ignore')
        results['bbox_fields'].append('tb_gt_bboxes')

        results['lr_gt_bboxes'] = ann_info['lr_bboxes'].copy()

        gt_bboxes_ignore = ann_info.get('bboxes_ignore', None)
        if gt_bboxes_ignore is not None:
            results['gt_bboxes_ignore'] = gt_bboxes_ignore.copy()
            results['bbox_fields'].append('gt_bboxes_ignore')
        results['bbox_fields'].append('lr_gt_bboxes')
        return results
    def _load_labels(self, results):
        """Private function to load label annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded label annotations.
        """

        results['tb_gt_labels'] = results['ann_info']['tb_abels'].copy()
        results['lr_gt_labels'] = results['ann_info']['lr_abels'].copy()
        return results