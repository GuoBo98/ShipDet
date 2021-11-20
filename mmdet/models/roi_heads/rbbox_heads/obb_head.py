import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.core import (multi_apply, build_bbox_coder)
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.roi_heads.bbox_heads import ConvFCBBoxHead
from mmdet.models.losses import accuracy

from mmdet.core import DeltaOBBCoder, thetaobb_rescale, multiclass_nms_obb, thetaobb2bbox


@HEADS.register_module()
class OBBHead(ConvFCBBoxHead):
    def __init__(self,
                 out_dim_reg=5,
                 obb_encode='thetaobb',
                 obb_coder=dict(
                    type='DeltaOBBCoder',
                    target_means=[.0, .0, .0, .0, .0],
                    target_stds=[0.1, 0.1, 0.2, 0.2, 0.1]),
                 loss_obb=dict(
                    type='SmoothL1Loss', beta=1.0, loss_weight=1.0),
                 *args,
                 **kwargs):
        super(OBBHead, self).__init__(*args, **kwargs)
        self.out_dim_reg = out_dim_reg
        self.fc_reg = nn.Linear(self.reg_last_dim, out_dim_reg)
        self.obb_encode = obb_encode
        self.obb_coder = build_bbox_coder(obb_coder)
        self.loss_obb = build_loss(loss_obb)

    def _get_target_single(self, 
                           pos_proposals,
                           neg_proposals,
                           pos_assigned_gt_inds, 
                           gt_obbs,
                           gt_labels,
                           cfg):
        num_pos = pos_proposals.size(0)
        num_neg = neg_proposals.size(0)

        num_samples = num_pos + num_neg
        labels = pos_proposals.new_full((num_samples, ),
                                     self.num_classes,
                                     dtype=torch.long)
        label_weights = pos_proposals.new_zeros(num_samples)
        obb_targets = pos_proposals.new_zeros(num_samples, self.out_dim_reg)
        obb_weights = pos_proposals.new_zeros(num_samples, self.out_dim_reg)
        
        pos_gt_obbs = []
        pos_gt_labels = []
        if num_pos > 0:
            pos_assigned_gt_inds = pos_assigned_gt_inds.cpu().numpy()
            for i in range(num_pos):
                gt_obb = gt_obbs[pos_assigned_gt_inds[i], :]
                gt_label = gt_labels[pos_assigned_gt_inds[i]]

                pos_gt_obbs.append(gt_obb.tolist())
                pos_gt_labels.append(gt_label.tolist())

            pos_weight = 1.0
            label_weights[:num_pos] = pos_weight
            obb_weights[:num_pos, :] = 1.0

            pos_gt_obbs = np.array(pos_gt_obbs)
            pos_gt_obbs = torch.from_numpy(np.stack(pos_gt_obbs)).float().to(pos_proposals.device)

            pos_gt_labels = np.array(pos_gt_labels)
            pos_gt_labels = torch.from_numpy(np.stack(pos_gt_labels)).float().to(pos_proposals.device)

            if not self.reg_decoded_bbox:
                pos_obb_targets = self.obb_coder.encode(
                    pos_proposals, pos_gt_obbs)
            else:
                pos_obb_targets = pos_gt_obbs
            
            obb_targets[:num_pos, :] = pos_obb_targets
            labels[:num_pos] = pos_gt_labels

        if num_neg > 0:
            label_weights[-num_neg:] = 1.0

        return labels, label_weights, obb_targets, obb_weights

    def get_targets(self, sampling_results, gt_obbs, gt_labels, rcnn_train_cfg):
        pos_proposals = [res.pos_bboxes for res in sampling_results]
        neg_proposals = [res.neg_bboxes for res in sampling_results]
        pos_assigned_gt_inds = [
            res.pos_assigned_gt_inds for res in sampling_results
        ]
        cfg_list = [rcnn_train_cfg for _ in range(len(pos_proposals))]
        labels, label_weights, obb_targets, obb_weights = multi_apply(self._get_target_single, 
            pos_proposals,
            neg_proposals,
            pos_assigned_gt_inds, 
            gt_obbs,
            gt_labels,
            cfg_list)
        labels = torch.cat(labels, 0)
        label_weights = torch.cat(label_weights, 0)
        obb_targets = torch.cat(obb_targets, 0)
        obb_weights = torch.cat(obb_weights, 0)
        
        return labels, label_weights, obb_targets, obb_weights

    def loss(self,
             cls_score,
             obb_pred,
             rois,
             labels,
             label_weights,
             obb_targets,
             obb_weights,
             reduction_override=None):
        losses = dict()
        if cls_score is not None:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
            if cls_score.numel() > 0:
                losses['loss_obb_cls'] = self.loss_cls(
                    cls_score,
                    labels,
                    label_weights,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override)
                losses['acc'] = accuracy(cls_score, labels)
        if obb_pred is not None:
            bg_class_ind = self.num_classes
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            pos_inds = (labels >= 0) & (labels < bg_class_ind)
            # do not perform bounding box regression for BG anymore.
            if pos_inds.any():
                if self.reg_decoded_bbox:
                    obb_pred = self.obb_coder.decode(rois[:, 1:], obb_pred)
                if self.reg_class_agnostic:
                    pos_obb_pred = obb_pred.view(
                        obb_pred.size(0), self.out_dim_reg)[pos_inds.type(torch.bool)]
                else:
                    pos_obb_pred = obb_pred.view(
                        obb_pred.size(0), -1, self.out_dim_reg)[pos_inds.type(torch.bool), labels[pos_inds.type(torch.bool)]]
                losses['loss_obb'] = self.loss_obb(
                    pos_obb_pred,
                    obb_targets[pos_inds.type(torch.bool)],
                    obb_weights[pos_inds.type(torch.bool)],
                    avg_factor=obb_targets.size(0),
                    reduction_override=reduction_override)
            else:
                losses['loss_obb'] = obb_pred.sum() * 0
        return losses

    def get_obbs(self,
                   rois,
                   cls_score,
                   obb_pred,
                   img_shape,
                   scale_factor,
                   rescale=False,
                   cfg=None):
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        scores = F.softmax(cls_score, dim=1) if cls_score is not None else None

        if obb_pred is not None:
            obbs = self.obb_coder.decode(
                rois[:, 1:], obb_pred, max_shape=img_shape)
        else:
            # TODO: fix the size of output
            obbs = rois[:, 1:].clone()

        if rescale and obbs.size(0) > 0:
            if isinstance(scale_factor, float):
                obbs = thetaobb_rescale(obbs, scale_factor, reverse_flag=True)
            else:
                scale_factor = obbs.new_tensor(scale_factor)
                angle_factor = obbs.new_tensor([1.0])
                scale_factor = torch.cat([scale_factor, angle_factor])

                obbs = (obbs.view(obbs.size(0), -1, 5) / scale_factor).view(obbs.size()[0], -1)

        if cfg is None:
            return obbs, scores
        else:
            hbbs = obbs.new_zeros(obbs.size(0), 4)
            obbs_np = obbs.cpu().numpy()
            for idx, obb in enumerate(obbs_np):
                hbb = thetaobb2bbox(obb)
                hbb = obbs.new_tensor(hbb)
                hbbs[idx, :] = hbb
            
            _, det_labels, keep = multiclass_nms_obb(hbbs, scores,
                                                    cfg.score_thr, cfg.nms,
                                                    cfg.max_per_img)
            obbs = obbs[keep]
            scores = scores[keep]
 
            det_obbs = obbs
            
            return det_obbs, det_labels

    def get_obbs_parallel(self,
                   rois,
                   cls_score,
                   obb_pred,
                   img_shape,
                   scale_factor,
                   rescale=False,
                   cfg=None,
                   keep_ind=None):
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        scores = F.softmax(cls_score, dim=1) if cls_score is not None else None
        if obb_pred is not None:
            obbs = self.obb_coder.decode(
                rois[:, 1:], obb_pred, max_shape=img_shape)
        else:
            # TODO: fix the size of output
            obbs = rois[:, 1:].clone()

        if rescale and obbs.size(0) > 0:
            if isinstance(scale_factor, float):
                obbs = thetaobb_rescale(obbs, scale_factor, reverse_flag=True)
            else:
                scale_factor = obbs.new_tensor(scale_factor)
                angle_factor = obbs.new_tensor([1.0])
                scale_factor = torch.cat([scale_factor, angle_factor])
                obbs = (obbs.view(obbs.size(0), -1, 5) / scale_factor).view(obbs.size()[0], -1)

        if cfg is None:
            return obbs, scores
        else:
            det_obbs = obbs[keep_ind]
            
            return det_obbs