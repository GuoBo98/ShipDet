import torch
import torch.nn as nn

from mmdet.core import (bbox2result, bbox2roi, bbox_mapping, build_assigner,
                        build_sampler, merge_aug_bboxes, merge_aug_masks,
                        multiclass_nms)
from mmdet.models.builder import HEADS, build_head, build_roi_extractor
from .base_roitrans_roi_head import BaseROITRANSRoIHead

from mmdet.models.roi_heads.test_mixins import BBoxTestMixin, MaskTestMixin
from mmdet.core import gt_mask_bp_obbs_list, roi2droi, choose_best_Rroi_batch, dbbox2roi, dbbox2result, \
    multiclass_nms_rbbox, merge_aug_rbboxes
import copy


@HEADS.register_module()
class ROITransRoIHead(BaseROITRANSRoIHead, BBoxTestMixin, MaskTestMixin):
    """
    A ROI Transformer head implement based on cascade rcnn
    """

    def __init__(self,
                 bbox_roi_extractor=None,
                 bbox_head=None,
                 rbbox_roi_extractor=None,
                 rbbox_head=None,
                 mask_roi_extractor=None,
                 mask_head=None,
                 shared_head=None,
                 train_cfg=None,
                 test_cfg=None):
        assert bbox_roi_extractor is not None
        assert bbox_head is not None
        assert shared_head is None, \
            'Shared head is not supported in Cascade RCNN anymore'
        super(ROITransRoIHead, self).__init__(
            bbox_roi_extractor=bbox_roi_extractor,
            bbox_head=bbox_head,
            rbbox_roi_extractor=rbbox_roi_extractor,
            rbbox_head=rbbox_head,
            mask_roi_extractor=mask_roi_extractor,
            mask_head=mask_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg)

    def init_bbox_head(self, bbox_roi_extractor, bbox_head, rbbox_roi_extractor, rbbox_head):
        """Initialize box head, rbbox head and box roi extractor, rbbox roi extract.

        Args:
            bbox_roi_extractor (dict): Config of box roi extractor.
            bbox_head (dict): Config of box in box head.
            rbbox_roi_extractor (dict): Config of rotated box roi extractor.
            rbbox_head (dict): Config of rotated box in rotated box head.
        """
        self.bbox_roi_extractor = nn.ModuleList()
        self.bbox_head = nn.ModuleList()
        self.rbbox_roi_extractor = nn.ModuleList()
        self.rbbox_head = nn.ModuleList()

        # default stage=0, TODO:cascade ROI transformer
        if not isinstance(bbox_roi_extractor, list):
            bbox_roi_extractor = [bbox_roi_extractor]
        if not isinstance(bbox_head, list):
            # bbox_head = [bbox_head for _ in range(self.num_stages)]
            bbox_head = [bbox_head]
        if not isinstance(rbbox_roi_extractor, list):
            rbbox_roi_extractor = [rbbox_roi_extractor]
        if not isinstance(rbbox_head, list):
            rbbox_head = [rbbox_head]

        assert len(bbox_roi_extractor) == len(bbox_head)
        assert len(rbbox_roi_extractor) == len(rbbox_head)

        for roi_extractor, head, rroi_extractor, rhead in zip(bbox_roi_extractor, bbox_head, rbbox_roi_extractor,
                                                              rbbox_head):
            self.bbox_roi_extractor.append(build_roi_extractor(roi_extractor))
            self.bbox_head.append(build_head(head))

            # TODO: ADD_ROTATED RBBOX_ROI_EXACTOROR
            self.rbbox_roi_extractor.append(build_roi_extractor(rroi_extractor))
            self.rbbox_head.append(build_head(rhead))

    def init_mask_head(self, mask_roi_extractor, mask_head):
        """Initialize mask head and mask roi extractor.

        Args:
            mask_roi_extractor (dict): Config of mask roi extractor.
            mask_head (dict): Config of mask in mask head.
        """
        self.mask_head = nn.ModuleList()
        if not isinstance(mask_head, list):
            mask_head = [mask_head for _ in range(self.num_stages)]
        assert len(mask_head) == self.num_stages
        for head in mask_head:
            self.mask_head.append(build_head(head))
        if mask_roi_extractor is not None:
            self.share_roi_extractor = False
            self.mask_roi_extractor = nn.ModuleList()
            if not isinstance(mask_roi_extractor, list):
                mask_roi_extractor = [
                    mask_roi_extractor for _ in range(self.num_stages)
                ]
            assert len(mask_roi_extractor) == self.num_stages
            for roi_extractor in mask_roi_extractor:
                self.mask_roi_extractor.append(
                    build_roi_extractor(roi_extractor))
        else:
            self.share_roi_extractor = True
            self.mask_roi_extractor = self.bbox_roi_extractor

    def init_assigner_sampler(self):
        """Initialize assigner and sampler for each stage."""
        self.bbox_assigner = []
        self.bbox_sampler = []
        if self.train_cfg is not None:
            for idx, rcnn_train_cfg in enumerate(self.train_cfg):
                self.bbox_assigner.append(
                    build_assigner(rcnn_train_cfg.assigner))
                self.current_stage = idx
                self.bbox_sampler.append(
                    build_sampler(rcnn_train_cfg.sampler, context=self))

    def init_weights(self, pretrained):
        """Initialize the weights in head.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        if self.with_shared_head:
            self.shared_head.init_weights(pretrained=pretrained)
        for i in range(1):
            if self.with_bbox:
                self.bbox_roi_extractor[i].init_weights()
                self.bbox_head[i].init_weights()
            if self.with_rbbox:
                self.rbbox_roi_extractor[i].init_weights()
                self.rbbox_head[i].init_weights()
            if self.with_mask:
                if not self.share_roi_extractor:
                    self.mask_roi_extractor[i].init_weights()
                self.mask_head[i].init_weights()

    def forward_dummy(self, x, proposals):
        raise AssertionError('not implement')

    def _roitrans_forward(self, stage, x, rois):
        """Box head forward function used in both training and testing."""
        bbox_roi_extractor = self.bbox_roi_extractor[stage]
        bbox_head = self.bbox_head[stage]
        bbox_feats = bbox_roi_extractor(x[:bbox_roi_extractor.num_inputs],
                                        rois)
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)

        # do not support caffe_c4 model anymore
        cls_score, bbox_pred = bbox_head(bbox_feats)

        bbox_results = dict(cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)
        return bbox_results

    def _roitrans_forward_train(self, stage, x, sampling_results, gt_masks,
                                gt_labels, roi_trans_cfg):
        """Run forward function and calculate loss for roi transformer head in training."""
        rois = bbox2roi([res.bboxes for res in sampling_results])
        bbox_results = self._roitrans_forward(stage, x, rois)

        roitrans_targets = self.bbox_head[stage].get_targets(
            sampling_results, gt_masks, gt_labels, roi_trans_cfg)

        loss_bbox = self.bbox_head[stage].loss(bbox_results['cls_score'],
                                               bbox_results['bbox_pred'],
                                               *roitrans_targets)

        bbox_results.update(
            loss_bbox=loss_bbox, rois=rois, roitrans_targets=roitrans_targets)
        return bbox_results

    def _rbbox_forward(self, stage, x, rrois):
        rbbox_roi_exactor = self.rbbox_roi_extractor[stage]
        rbbox_head = self.rbbox_head[stage]
        rbbox_feats = rbbox_roi_exactor(x[:self.rbbox_roi_extractor[stage].num_inputs], rrois)

        if self.with_shared_head_rbbox:
            rbbox_feats = self.shared_head_rbbox(rbbox_feats)

        cls_score, rbbox_pred = rbbox_head(rbbox_feats)
        rbbox_results = dict(
            cls_score=cls_score, bbox_pred=rbbox_pred, bbox_feats=rbbox_feats)
        return rbbox_results

    def _rbbox_forward_train(self, stage, x, sampling_results, gt_obbs, gt_labels, rbbox_cfg):
        # TODO: dbbox2roi
        rrois = dbbox2roi([res.bboxes for res in sampling_results])
        # 特征扩大?
        rrois[:, 3] = rrois[:, 3] * self.rbbox_roi_extractor[stage].w_enlarge
        rrois[:, 4] = rrois[:, 4] * self.rbbox_roi_extractor[stage].h_enlarge

        rbbox_results = self._rbbox_forward(stage, x, rrois)
        rbbox_targets = self.rbbox_head[stage].get_target_rbbox(sampling_results, gt_obbs, gt_labels, rbbox_cfg)
        loss_rbbox = self.rbbox_head[stage].loss(rbbox_results['cls_score'], rbbox_results['bbox_pred'], *rbbox_targets)
        rbbox_results.update(loss_rbbox=loss_rbbox, rrois=rrois, rbbox_targets=rbbox_targets)

        return rbbox_results

    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None):
        """
        Args:
            x (list[Tensor]): list of multi-level img features.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            proposals (list[Tensors]): list of region proposals.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        losses = dict()

        # TODO: trans gt_masks to gt_obbs
        gt_obbs = gt_mask_bp_obbs_list(
            gt_masks)  # 返回每张图片的obbs　[[num_rot_recs, 5], [num_rot_recs, 5], [num_rot_recs, 5], [num_rot_recs, 5]]

        roi_trans_cfg = self.train_cfg[0]
        rrcnn_train_cfg = self.train_cfg[1]
        # assign gts and sample proposals
        sampling_results = []
        if self.with_bbox or self.with_mask:
            # bbox_assigner = self.bbox_assigner[i]
            roi_trans_assigner = self.bbox_assigner[0]
            # bbox_sampler = self.bbox_sampler[i]
            roi_trans_sampler = self.bbox_sampler[0]
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]

            for j in range(num_imgs):
                assign_result = roi_trans_assigner.assign(
                    proposal_list[j], gt_bboxes[j], gt_bboxes_ignore[j],
                    gt_labels[j])
                sampling_result = roi_trans_sampler.sample(
                    assign_result,
                    proposal_list[j],
                    gt_bboxes[j],
                    gt_labels[j],
                    feats=[lvl_feat[j][None] for lvl_feat in x])
                sampling_results.append(sampling_result)
        # ROI trans head forward and loss
        # ROI trans is 'stage=0' according to cascade rcnn
        if self.with_bbox:
            bbox_results = self._roitrans_forward_train(0, x, sampling_results,
                                                        gt_masks, gt_labels,
                                                        roi_trans_cfg)
            for name, value in bbox_results['loss_bbox'].items():
                losses['s{}.{}'.format(0, name)] = (value)

        # refine bboxes
        pos_is_gts = [res.pos_is_gt for res in sampling_results]
        # bbox_targets is a tuple
        roi_labels = bbox_results['roitrans_targets'][0]
        with torch.no_grad():
            # TODO: rbbox_head refine_rbboxes
            # TODO: roi2droi
            rotated_proposal_list = self.bbox_head[0].refine_rbboxes(roi2droi(bbox_results['rois']),
                                                                     roi_labels, bbox_results['bbox_pred'], pos_is_gts,
                                                                     img_metas)
        if self.with_rbbox:
            rbbox_assigner = self.bbox_assigner[1]
            rbbox_sampler = self.bbox_sampler[1]
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for j in range(num_imgs):
                # TODO: choose_best_Rroi_batch
                gt_obbs_best_roi = choose_best_Rroi_batch(gt_obbs[j])
                assign_result = rbbox_assigner.assign(rotated_proposal_list[j], gt_obbs_best_roi, gt_bboxes_ignore[j],
                                                      gt_labels[j])
                sampling_result = rbbox_sampler.sample(assign_result,
                                                       rotated_proposal_list[j],
                                                       torch.from_numpy(gt_obbs_best_roi).float().to(
                                                           rotated_proposal_list[j].device),
                                                       gt_labels[j],
                                                       feats=[lvl_feat[j][None] for lvl_feat in x])
                sampling_results.append(sampling_result)

            rbbox_results = self._rbbox_forward_train(0, x, sampling_results, gt_obbs, gt_labels, rrcnn_train_cfg)
            for name, value in rbbox_results['loss_rbbox'].items():
                losses['s{}.{}'.format(1, name)] = (value)

        return losses

    def simple_test(self, x, proposal_list, img_metas, rescale=False):

        assert self.with_bbox, 'Bbox head must be implemented.'
        num_imgs = len(proposal_list)
        img_shapes = tuple(meta['img_shape'] for meta in img_metas)
        ori_shapes = tuple(meta['ori_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)
        ms_roi_trans_result = {}
        ms_rbbox_result = {}
        ms_roi_trans_scores = []
        ms_rbbox_scores = []
        rcnn_test_cfg = self.test_cfg

        rois = bbox2roi(proposal_list)
        # Stage=0
        # TODO: cascade roi transform
        # ROI Trans forward
        bbox_results = self._roitrans_forward(0, x, rois)
        roi_trans_cls_score = bbox_results['cls_score']
        roi_trans_bbox_pred = bbox_results['bbox_pred']
        num_proposals_per_img = tuple(len(proposals) for proposals in proposal_list)
        rois = rois.split(num_proposals_per_img, 0)
        roi_trans_cls_score = roi_trans_cls_score.split(num_proposals_per_img, 0)
        roi_trans_bbox_pred = roi_trans_bbox_pred.split(num_proposals_per_img, 0)
        # bbox_label = bbox_results['cls_score'].argmax(dim=1)
        roi_trans_bbox_label = [s[:, :-1].argmax(dim=1) for s in roi_trans_cls_score]

        rrois = torch.cat([
            self.bbox_head[0].regress_by_class_rbbox(roi2droi(rois[j]), roi_trans_bbox_label[j],
                                                     roi_trans_bbox_pred[j], img_metas[j]) for j in range(num_imgs)])
        # # TODO:chech img_metas, change format
        # rrois = self.bbox_head[0].regress_by_class_rbbox(roi2droi(rois), bbox_label, bbox_results['bbox_pred'],
        #                                                  img_metas[0])

        # RBBOX forward
        rrois_enlarge = copy.deepcopy(rrois)
        rrois_enlarge[:, 3] = rrois_enlarge[:, 3] * self.rbbox_roi_extractor[0].w_enlarge
        rrois_enlarge[:, 4] = rrois_enlarge[:, 4] * self.rbbox_roi_extractor[0].h_enlarge
        rbbox_results = self._rbbox_forward(0, x, rrois_enlarge)
        rcls_score = rbbox_results['cls_score']
        rbbox_pred = rbbox_results['bbox_pred']

        num_proposals_per_img = tuple(len(proposals) for proposals in proposal_list)
        rrois = rrois.split(num_proposals_per_img, 0)
        rcls_score = rcls_score.split(num_proposals_per_img, 0)
        rbbox_pred = rbbox_pred.split(num_proposals_per_img, 0)
        ms_rbbox_scores.append(rcls_score)

        det_rbboxes = []
        det_rlabels = []
        for i in range(num_imgs):
            det_rbbox, det_label = self.rbbox_head[0].get_det_rbboxes(
                rrois[i],
                rcls_score[i],
                rbbox_pred[i],
                img_shapes[i],
                scale_factors[i],
                rescale=rescale,
                cfg=rcnn_test_cfg)
            det_rbboxes.append(det_rbbox)
            det_rlabels.append(det_label)

        # rbbox_results = dbbox2result(det_rbbox, det_label,
        #                              self.rbbox_head.num_classes)
        # rbbox_results:list(cls order),  list element shape:[n, 9], (x1, y1, x2, y2, x3, y3, x4, y4, score)
        rbbox_results = [
            dbbox2result(det_rbboxes[i], det_rlabels[i],
                         self.rbbox_head[0].num_classes)
            for i in range(num_imgs)
        ]

        ms_rbbox_result['ensemble'] = rbbox_results

        return ms_rbbox_result['ensemble']

    def aug_test(self, features, proposal_list, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        # import pdb
        # pdb.set_trace()
        rcnn_test_cfg = self.test_cfg
        aug_rbboxes = []
        aug_rscores = []
        for x, img_meta in zip(features, img_metas):
            # only one image in the batch
            img_shape = img_meta[0]['img_shape']
            scale_factor = img_meta[0]['scale_factor']
            flip = img_meta[0]['flip']
            flip_direction = img_meta[0]['flip_direction']

            proposals = bbox_mapping(proposal_list[0][:, :4], img_shape,
                                     scale_factor, flip, flip_direction)
            # "ms" in variable names means multi-stage
            ms_scores = []

            rois = bbox2roi([proposals])
            bbox_results = self._roitrans_forward(0, x, rois)
            bbox_label = bbox_results['cls_score'][:, :-1].argmax(dim=1)
            rrois = self.bbox_head[0].regress_by_class_rbbox(roi2droi(rois), bbox_label, bbox_results['bbox_pred'],
                                                             img_meta[0])
            rrois_enlarge = copy.deepcopy(rrois)
            rrois_enlarge[:, 3] = rrois_enlarge[:, 3] * self.rbbox_roi_extractor[0].w_enlarge
            rrois_enlarge[:, 4] = rrois_enlarge[:, 4] * self.rbbox_roi_extractor[0].h_enlarge
            rbbox_results = self._rbbox_forward(0, x, rrois_enlarge)
            rcls_score = rbbox_results['cls_score']
            rbbox_pred = rbbox_results['bbox_pred']

            rbboxes, rscores = self.rbbox_head[0].get_det_rbboxes(
                rrois,
                rcls_score,
                rbbox_pred,
                img_shape,
                scale_factor,
                rescale=False,
                cfg=None)
            aug_rbboxes.append(rbboxes)
            aug_rscores.append(rscores)

        # after merging, bboxes will be rescaled to the original image size
        merged_bboxes, merged_scores = merge_aug_rbboxes(
            aug_rbboxes, aug_rscores, img_metas, rcnn_test_cfg)
        det_bboxes, det_labels = multiclass_nms_rbbox(merged_bboxes, merged_scores,
                                                      rcnn_test_cfg.score_thr,
                                                      rcnn_test_cfg.nms,
                                                      rcnn_test_cfg.max_per_img)

        rbbox_results = dbbox2result(det_bboxes, det_labels, self.rbbox_head[0].num_classes)

        return [rbbox_results]
