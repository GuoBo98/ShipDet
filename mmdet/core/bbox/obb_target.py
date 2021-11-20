import mmcv
import numpy as np
import torch
from torch.nn.modules.utils import _pair


def obb_target(pos_proposals_list, 
               pos_assigned_gt_inds_list, 
               gt_obbs_list,
               cfg,
               obb_coder):
    cfg_list = [cfg for _ in range(len(pos_proposals_list))]
    obb_coder_list = [obb_coder for _ in range(len(pos_proposals_list))]
    obb_targets = map(obb_target_single, 
                      pos_proposals_list,
                      pos_assigned_gt_inds_list, 
                      gt_obbs_list, 
                      cfg_list,
                      obb_coder_list)
    obb_targets = torch.cat(list(obb_targets))
    return obb_targets

def obb_target_single(pos_proposals, pos_assigned_gt_inds, gt_obbs, cfg, obb_coder):
    num_pos = pos_proposals.size(0)
    obb_targets = []
    if num_pos > 0:
        proposals_np = pos_proposals.cpu().numpy()
        pos_assigned_gt_inds = pos_assigned_gt_inds.cpu().numpy()
        pos_gt_obbs = gt_obbs[pos_assigned_gt_inds]

        pos_bbox_targets = self.bbox_coder.encode(
                    pos_bboxes, pos_gt_bboxes)


        for i in range(num_pos):
            gt_obb = gt_obbs[pos_assigned_gt_inds[i]]
            bbox = proposals_np[i, :].astype(np.int32)
            x1, y1, x2, y2 = bbox
            w = np.maximum(x2 - x1 + 1, 1)
            h = np.maximum(y2 - y1 + 1, 1)
            # mask is uint8 both before and after resizing
            # mask_size (h, w) to (w, h)
            target = mmcv.imresize(gt_obb[y1:y1 + h, x1:x1 + w],
                                   mask_size[::-1])
            mask_targets.append(target)
        mask_targets = torch.from_numpy(np.stack(mask_targets)).float().to(
            pos_proposals.device)
    else:
        mask_targets = pos_proposals.new_zeros((0, ) + mask_size)
    return mask_targets
