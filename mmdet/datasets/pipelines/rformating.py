from collections.abc import Sequence

import mmcv
import numpy as np
import torch
from mmcv.parallel import DataContainer as DC

from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import DefaultFormatBundle, to_tensor


@PIPELINES.register_module()
class DefaultFormatBundleOBB(DefaultFormatBundle):
    def __call__(self, results):
        super().__call__(results)
        for key in ['gt_obbs']:
            if key not in results:
                continue
            results[key] = DC(to_tensor(results[key]))
        return results
