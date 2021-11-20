import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import force_fp32
from mmdet.models.roi_heads.bbox_heads import ConvFCBBoxHead

from mmdet.core import bbox_target_rbbox, rbbox_target_rbbox,choose_best_Rroi_batch, hbb2obb_v2, multiclass_nms_rbbox
from mmdet.models.losses import accuracy
from mmdet.models.builder import HEADS


@HEADS.register_module()
class ConvFCBBoxHeadRbbox(ConvFCBBoxHead):
    r"""More general bbox head, with shared conv and fc layers and two optional
    separated branches.

    .. code-block:: none

                                    /-> cls convs -> cls fcs -> cls
        shared convs -> shared fcs
                                    \-> reg convs -> reg fcs -> reg
    """  # noqa: W605

    def __init__(self,
                 reg_class_agnostic,
                 with_module=True,
                 hbb_trans='hbb2obb_v2',
                 *args,
                 **kwargs):
        super(ConvFCBBoxHeadRbbox, self).__init__(*args, **kwargs)

        self.with_module = with_module
        self.hbb_trans = hbb_trans
        self.reg_class_agnostic = reg_class_agnostic
        if self.with_reg:
            out_dim_reg = (5 if self.reg_class_agnostic else 5 * self.num_classes)
            self.fc_reg = nn.Linear(self.reg_last_dim, out_dim_reg)

    def get_targets(self, sampling_results, gt_masks, gt_labels,
                    rcnn_train_cfg):
        """
        obb target hbb
        :param sampling_results:
        :param gt_masks:
        :param gt_labels:
        :param rcnn_train_cfg:
        :param mod: 'normal' or 'best_match', 'best_match' is used for RoI Transformer
        :return:
        """
        pos_proposals = [res.pos_bboxes for res in sampling_results]
        neg_proposals = [res.neg_bboxes for res in sampling_results]
        # pos_gt_bboxes = [res.pos_gt_bboxes for res in sampling_results]
        # TODO: first get indexs of pos_gt_bboxes, then index from gt_bboxes
        # TODO: refactor it, direct use the gt_rbboxes instead of gt_masks
        pos_assigned_gt_inds = [
            res.pos_assigned_gt_inds for res in sampling_results
        ]
        pos_gt_labels = [res.pos_gt_labels for res in sampling_results]
        #reg_classes = 1 if self.reg_class_agnostic else self.num_classes
        cls_reg_targets = bbox_target_rbbox(
            pos_proposals,
            neg_proposals,
            pos_assigned_gt_inds,
            gt_masks,
            pos_gt_labels,
            rcnn_train_cfg,
            self.num_classes,
            coder=self.bbox_coder,
            with_module=self.with_module,
            hbb_trans=self.hbb_trans)
        return cls_reg_targets

    def get_target_rbbox(self, sampling_results, gt_bboxes, gt_labels,
                         rcnn_train_cfg):
        """
        obb target obb
        :param sampling_results:
        :param gt_bboxes:
        :param gt_labels:
        :param rcnn_train_cfg:
        :return:
        """
        pos_proposals = [res.pos_bboxes for res in sampling_results]  # [-pi/4,-3pi/4]
        # pos_proposals = choose_best_Rroi_batch(pos_proposals)
        neg_proposals = [res.neg_bboxes for res in sampling_results]
        pos_gt_bboxes = [res.pos_gt_bboxes for res in sampling_results]  # [0.pi]
        pos_gt_labels = [res.pos_gt_labels for res in sampling_results]
        #reg_classes = 1 if self.reg_class_agnostic else self.num_classes
        cls_reg_targets = rbbox_target_rbbox(
            pos_proposals,
            neg_proposals,
            pos_gt_bboxes,
            pos_gt_labels,
            rcnn_train_cfg,
            self.num_classes,
            coder=self.bbox_coder)
        return cls_reg_targets

    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def loss(self,
             cls_score,
             bbox_pred,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             reduction_override=None):
        losses = dict()
        if cls_score is not None:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
            if cls_score.numel() > 0:
                losses['rbbox_loss_cls'] = self.loss_cls(
                    cls_score,
                    labels,
                    label_weights,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override)
                losses['rbbox_acc'] = accuracy(cls_score, labels)
        if bbox_pred is not None:
            bg_class_ind = self.num_classes
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            pos_inds = (labels >= 0) & (labels < bg_class_ind)
            # do not perform bounding box regression for BG anymore.
            if pos_inds.any():
                if self.reg_class_agnostic:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), 5)[pos_inds.type(torch.bool)]
                else:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), -1,
                        5)[pos_inds.type(torch.bool),
                           labels[pos_inds.type(torch.bool)]]
                losses['rbbox_loss_bbox'] = self.loss_bbox(
                    pos_bbox_pred,
                    bbox_targets[pos_inds.type(torch.bool)],
                    bbox_weights[pos_inds.type(torch.bool)],
                    avg_factor=bbox_targets.size(0),
                    reduction_override=reduction_override)
            else:
                losses['rbbox_loss_bbox'] = bbox_pred.sum() * 0
        return losses


    def get_det_rbboxes(self,
                        rrois,
                        cls_score,
                        rbbox_pred,
                        img_shape,
                        scale_factor,
                        rescale=False,
                        cfg=None):
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        scores = F.softmax(cls_score, dim=1) if cls_score is not None else None

        if rbbox_pred is not None:

            # dbboxes = delta2dbbox_v2(rrois[:, 1:], rbbox_pred, self.target_means,
            #                          self.target_stds, img_shape)

            dbboxes = self.bbox_coder.decode(rrois[:, 1:], rbbox_pred, 'delta2dbbox_v2', img_shape)
        else:
            # bboxes = rois[:, 1:]
            dbboxes = rrois[:, 1:].clone()
            # TODO: add clip here
            if img_shape is not None:
                dbboxes[:, [0, 2]].clamp_(min=0, max=img_shape[1])
                dbboxes[:, [1, 3]].clamp_(min=0, max=img_shape[0])
        if rescale:
            # bboxes /= scale_factor
            # dbboxes[:, :4] /= scale_factor
            if isinstance(scale_factor, float):
                dbboxes[:, 0::5] /= scale_factor
                dbboxes[:, 1::5] /= scale_factor
                dbboxes[:, 2::5] /= scale_factor
                dbboxes[:, 3::5] /= scale_factor
            else:
                scale_factor = dbboxes.new_tensor(scale_factor)
                dbboxes = dbboxes.view(dbboxes.size(0), -1, 5)
                # TODO: point base scale
                # TODO: check this
                # import pdb
                # pdb.set_trace()
                # print('dbboxes shape', dbboxes.size())
                # print('scale_factor size', scale_factor.size())
                dbboxes[:, :, :4] /= scale_factor
                dbboxes = dbboxes.view(dbboxes.size()[0], -1)
        if cfg is None:
            return dbboxes, scores
        else:
            # check multiscale
            det_bboxes, det_labels = multiclass_nms_rbbox(dbboxes, scores,
                                                          cfg.score_thr, cfg.nms,
                                                          cfg.max_per_img)
            # det_bboxes = torch.from_numpy(det_bboxes).to(c_device)
            # det_labels = torch.from_numpy(det_labels).to(c_device)
            return det_bboxes, det_labels

    @force_fp32(apply_to=('bbox_preds',))
    def refine_rbboxes(self, rois, labels, bbox_preds, pos_is_gts, img_metas):
        """Refine bboxes during training.

        Args:
            rois (Tensor): Shape (n*bs, 5), where n is image number per GPU,
                and bs is the sampled RoIs per image.
            labels (Tensor): Shape (n*bs, ).
            bbox_preds (Tensor): Shape (n*bs, 5) or (n*bs, 5*#class).
            pos_is_gts (list[Tensor]): Flags indicating if each positive bbox
                is a gt bbox.
            img_metas (list[dict]): Meta info of each image.

        Returns:
            list[Tensor]: Refined bboxes of each image in a mini-batch.
        """
        img_ids = rois[:, 0].long().unique(sorted=True)
        assert img_ids.numel() == len(img_metas)

        bboxes_list = []
        for i in range(len(img_metas)):
            # TODO check this
            inds = torch.nonzero(rois[:, 0] == i, as_tuple=False).squeeze(dim=1)
            num_rois = inds.numel()

            bboxes_ = rois[inds, 1:]
            label_ = labels[inds]
            bbox_pred_ = bbox_preds[inds]
            img_meta_ = img_metas[i]
            pos_is_gts_ = pos_is_gts[i]

            bboxes = self.regress_by_class_rbbox(bboxes_, label_, bbox_pred_,
                                                 img_meta_)
            # filter gt bboxes
            pos_keep = 1 - pos_is_gts_
            keep_inds = pos_is_gts_.new_ones(num_rois)
            keep_inds[:len(pos_is_gts_)] = pos_keep

            bboxes_list.append(bboxes[keep_inds.type(torch.bool)])

        return bboxes_list

    def regress_by_class_rbbox(self, rois, label, bbox_pred, img_meta):
        """Regress the bbox for the predicted class. Used in Cascade R-CNN.

        Args:
            rois (Tensor): shape (n, 5) or (n, 6)
            label (Tensor): shape (n, )
            bbox_pred (Tensor): shape (n, 5*(#class+1)) or (n, 5)
            img_meta (dict): Image meta info.

        Returns:
            Tensor: Regressed bboxes, the same shape as input rois.
        """
        # import pdb
        # pdb.set_trace()
        assert rois.size(1) == 5 or rois.size(1) == 6

        if not self.reg_class_agnostic:
            # import pdb
            # pdb.set_trace()
            label = label * 5
            inds = torch.stack((label, label + 1, label + 2, label + 3, label + 4), 1)
            bbox_pred = torch.gather(bbox_pred, 1, inds)
        assert bbox_pred.size(1) == 5

        if rois.size(1) == 5:
            if self.with_module:
                # new_rois = delta2dbbox(rois, bbox_pred, self.target_means,
                #                        self.target_stds, img_meta['img_shape'])
                new_rois = self.bbox_coder.decode(rois, bbox_pred, 'delta2dbbox', img_meta['img_shape'])
            else:
                # new_rois = delta2dbbox_v3(rois, bbox_pred, self.target_means,
                #                           self.target_stds, img_meta['img_shape'])
                new_rois = self.bbox_coder.decode(rois, bbox_pred, 'delta2dbbox_v3', img_meta['img_shape'])
            # choose best Rroi
            new_rois = choose_best_Rroi_batch(new_rois)
        else:
            if self.with_module:
                # bboxes = delta2dbbox(rois[:, 1:], bbox_pred, self.target_means,
                #                      self.target_stds, img_meta['img_shape'])
                bboxes = self.bbox_coder.decode(rois[:, 1:], bbox_pred, 'delta2dbbox', img_meta['img_shape'])
            else:
                # bboxes = delta2dbbox_v3(rois[:, 1:], bbox_pred, self.target_means,
                #                         self.target_stds, img_meta['img_shape'])
                bboxes = self.bbox_coder.decode(rois[:, 1:], bbox_pred, 'delta2dbbox_v3', img_meta['img_shape'])
            bboxes = choose_best_Rroi_batch(bboxes)
            new_rois = torch.cat((rois[:, [0]], bboxes), dim=1)

        return new_rois


@HEADS.register_module()
class SharedFCBBoxHeadRbbox(ConvFCBBoxHeadRbbox):

    def __init__(self, num_fcs=2, fc_out_channels=1024, *args, **kwargs):
        assert num_fcs >= 1
        super(SharedFCBBoxHeadRbbox, self).__init__(
            num_shared_convs=0,
            num_shared_fcs=num_fcs,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)

@HEADS.register_module()
class ConvFCBBoxHeadRbbox_NotShareCls(ConvFCBBoxHeadRbbox):
    def __init__(self,
                 reg_class_agnostic,
                 with_module=True,
                 hbb_trans='hbb2obb_v2',
                 *args,
                 **kwargs):
        super(ConvFCBBoxHeadRbbox_NotShareCls, self).__init__(reg_class_agnostic, with_module, hbb_trans, *args, **kwargs)

        if self.with_cls:
            del self.fc_cls
            self.fc_cls_share = nn.Linear(self.cls_last_dim, self.num_classes + 1)

    def init_weights(self):
        if self.with_cls:
            nn.init.normal_(self.fc_cls_share.weight, 0, 0.01)
            nn.init.constant_(self.fc_cls_share.bias, 0)
        if self.with_reg:
            nn.init.normal_(self.fc_reg.weight, 0, 0.001)
            nn.init.constant_(self.fc_reg.bias, 0)

        # conv layers are already initialized by ConvModule
        for module_list in [self.shared_fcs, self.cls_fcs, self.reg_fcs]:
            for m in module_list.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # shared part
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)

            x = x.flatten(1)

            for fc in self.shared_fcs:
                x = self.relu(fc(x))
        # separate branches
        x_cls = x
        x_reg = x

        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        cls_score = self.fc_cls_share(x_cls) if self.with_cls else None
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
        return cls_score, bbox_pred

@HEADS.register_module()
class SharedFCBBoxHeadRbbox_NotShareCls(ConvFCBBoxHeadRbbox_NotShareCls):

    def __init__(self, num_fcs=2, fc_out_channels=1024, *args, **kwargs):
        assert num_fcs >= 1
        super(SharedFCBBoxHeadRbbox_NotShareCls, self).__init__(
            num_shared_convs=0,
            num_shared_fcs=num_fcs,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)