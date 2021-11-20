import torch.nn as nn
from mmcv.cnn import ConvModule, normal_init, bias_init_with_prob
from mmcv.ops import DeformConv2d, CornerPool, batched_nms
from logging import warning
from math import ceil, log
import torch
import torch.nn.functional as F
from ..utils import gaussian_radius, gen_gaussian_target
from .base_dense_head import BaseDenseHead
from mmdet.core import multi_apply
from ..builder import HEADS, build_loss
from .corner_head import CornerHead,BiCornerPool

@HEADS.register_module()
class AKSHead(CornerHead):
    """Head of AKSNET: by pualikarl. using for MasterDong

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        num_feat_levels (int): Levels of feature from the previous module. 2
            for HourglassNet-104 and 1 for HourglassNet-52. HourglassNet-104
            outputs the final feature and intermediate supervision feature and
            HourglassNet-52 only outputs the final feature. Default: 2.
        corner_emb_channels (int): Channel of embedding vector. Default: 1.
        train_cfg (dict | None): Training config. Useless in CornerHead,
            but we keep this variable for SingleStageDetector. Default: None.
        test_cfg (dict | None): Testing config of CornerHead. Default: None.
        loss_heatmap (dict | None): Config of corner heatmap loss. Default:
            GaussianFocalLoss.
        loss_embedding (dict | None): Config of corner embedding loss. Default:
            AssociativeEmbeddingLoss.
        loss_offset (dict | None): Config of corner offset loss. Default:
            SmoothL1Loss.
        loss_guiding_shift (dict): Config of guiding shift loss. Default:
            SmoothL1Loss.
        loss_centripetal_shift (dict): Config of centripetal shift loss.
            Default: SmoothL1Loss.
        loss_attention_mask(dict): Config of attention mask loss.
            Default: CrossEntropyLoss
    """

    def __init__(self,
                 *args,
                 centripetal_shift_channels=2,
                 guiding_shift_channels=2,
                 feat_adaption_conv_kernel=3,
                 loss_guiding_shift=dict(
                     type='SmoothL1Loss', beta=1.0, loss_weight=0.05),
                 loss_centripetal_shift=dict(
                     type='SmoothL1Loss', beta=1.0, loss_weight=1),
                 loss_mask=dict(
                     type='CrossEntropyLoss', use_mask=True, loss_weight=1.0),
                 **kwargs):
        assert centripetal_shift_channels == 2, (
            'AKSHead only support centripetal_shift_channels == 2')
        self.centripetal_shift_channels = centripetal_shift_channels
        assert guiding_shift_channels == 2, (
            'AKSHead only support guiding_shift_channels == 2')
        self.guiding_shift_channels = guiding_shift_channels
        self.feat_adaption_conv_kernel = feat_adaption_conv_kernel
        super(AKSHead, self).__init__(*args, **kwargs)
        self.loss_guiding_shift = build_loss(loss_guiding_shift)
        self.loss_centripetal_shift = build_loss(loss_centripetal_shift)
        self.loss_mask = build_loss(loss_mask)

    def _init_aks_layers(self):
        """Initialize AKSnet layers.

        Including feature adaption deform convs (feat_adaption), deform offset
        prediction convs (dcn_off), guiding shift (guiding_shift) and
        centripetal shift ( centripetal_shift). Each branch has four parts:
        prefix `t` for top,'l' for left, `b` for bottom and  'r' for right.
        """
        self.t_pool, self.l_pool = nn.ModuleList(), nn.ModuleList()
        self.b_pool, self.r_pool = nn.ModuleList(), nn.ModuleList()

        self.t_heat, self.l_heat = nn.ModuleList(), nn.ModuleList()
        self.b_heat, self.r_heat = nn.ModuleList(), nn.ModuleList()

        self.t_off, self.l_off = nn.ModuleList(), nn.ModuleList()
        self.b_off, self.r_off = nn.ModuleList(), nn.ModuleList()

        self.t_feat_adaption = nn.ModuleList()
        self.l_feat_adaption = nn.ModuleList()
        self.b_feat_adaption = nn.ModuleList()
        self.r_feat_adaption = nn.ModuleList()

        self.t_dcn_offset = nn.ModuleList()
        self.l_dcn_offset = nn.ModuleList()
        self.b_dcn_offset = nn.ModuleList()
        self.r_dcn_offset = nn.ModuleList()

        self.t_guiding_shift = nn.ModuleList()
        self.l_guiding_shift = nn.ModuleList()
        self.b_guiding_shift = nn.ModuleList()
        self.r_guiding_shift = nn.ModuleList()

        self.t_centripetal_shift = nn.ModuleList()
        self.l_centripetal_shift = nn.ModuleList()
        self.b_centripetal_shift = nn.ModuleList()
        self.r_centripetal_shift = nn.ModuleList()



        for _ in range(self.num_feat_levels):
            self.t_pool.append(
                BiCornerPool(
                    self.in_channels, ['top'],
                    out_channels=self.in_channels))
            self.l_pool.append(
                BiCornerPool(
                    self.in_channels, ['left'],
                    out_channels=self.in_channels))
            self.b_pool.append(
                BiCornerPool(
                    self.in_channels, ['bottom'],
                    out_channels=self.in_channels))
            self.r_pool.append(
                BiCornerPool(
                    self.in_channels, [ 'right'],
                    out_channels=self.in_channels))

            self.t_heat.append(
                self._make_layers(
                    out_channels=self.num_classes,
                    in_channels=self.in_channels))
            self.l_heat.append(
                self._make_layers(
                    out_channels=self.num_classes,
                    in_channels=self.in_channels))
            self.b_heat.append(
                self._make_layers(
                    out_channels=self.num_classes,
                    in_channels=self.in_channels))     
            self.r_heat.append(
                self._make_layers(
                    out_channels=self.num_classes,
                    in_channels=self.in_channels))

            self.tl_off.append(
                self._make_layers(
                    out_channels=self.corner_offset_channels,
                    in_channels=self.in_channels))
            self.br_off.append(
                self._make_layers(
                    out_channels=self.corner_offset_channels,
                    in_channels=self.in_channels))


            self.t_feat_adaption.append(
                DeformConv2d(self.in_channels, self.in_channels,
                             self.feat_adaption_conv_kernel, 1, 1))
            self.l_feat_adaption.append(
                DeformConv2d(self.in_channels, self.in_channels,
                             self.feat_adaption_conv_kernel, 1, 1))
            self.b_feat_adaption.append(
                DeformConv2d(self.in_channels, self.in_channels,
                             self.feat_adaption_conv_kernel, 1, 1))
            self.r_feat_adaption.append(
                DeformConv2d(self.in_channels, self.in_channels,
                             self.feat_adaption_conv_kernel, 1, 1))

            self.t_guiding_shift.append(
                self._make_layers(
                    out_channels=self.guiding_shift_channels,
                    in_channels=self.in_channels))
            self.l_guiding_shift.append(
                self._make_layers(
                    out_channels=self.guiding_shift_channels,
                    in_channels=self.in_channels))
            self.b_guiding_shift.append(
                self._make_layers(
                    out_channels=self.guiding_shift_channels,
                    in_channels=self.in_channels))
            self.r_guiding_shift.append(
                self._make_layers(
                    out_channels=self.guiding_shift_channels,
                    in_channels=self.in_channels))


            self.t_dcn_offset.append(
                ConvModule(
                    self.guiding_shift_channels,
                    self.feat_adaption_conv_kernel**2 *
                    self.guiding_shift_channels,
                    1,
                    bias=False,
                    act_cfg=None))
            self.l_dcn_offset.append(
                ConvModule(
                    self.guiding_shift_channels,
                    self.feat_adaption_conv_kernel**2 *
                    self.guiding_shift_channels,
                    1,
                    bias=False,
                    act_cfg=None))
            self.b_dcn_offset.append(
                ConvModule(
                    self.guiding_shift_channels,
                    self.feat_adaption_conv_kernel**2 *
                    self.guiding_shift_channels,
                    1,
                    bias=False,
                    act_cfg=None))
            self.r_dcn_offset.append(
                ConvModule(
                    self.guiding_shift_channels,
                    self.feat_adaption_conv_kernel**2 *
                    self.guiding_shift_channels,
                    1,
                    bias=False,
                    act_cfg=None))

            self.t_centripetal_shift.append(
                self._make_layers(
                    out_channels=self.centripetal_shift_channels,
                    in_channels=self.in_channels))
            self.l_centripetal_shift.append(
                self._make_layers(
                    out_channels=self.centripetal_shift_channels,
                    in_channels=self.in_channels))
            self.b_centripetal_shift.append(
                self._make_layers(
                    out_channels=self.centripetal_shift_channels,
                    in_channels=self.in_channels))
            self.r_centripetal_shift.append(
                self._make_layers(
                    out_channels=self.centripetal_shift_channels,
                    in_channels=self.in_channels))

    def _init_layers(self):
        """Initialize layers for AKSHead.

        Including two parts: CornerHead layers and AKSHead layers
        """
        super()._init_layers()  # using _init_layers in CornerHead
        self._init_aks_layers()


    def init_weights(self):
        """Initialize weights of the head."""
        bias_init = bias_init_with_prob(0.1)
        for i in range(self.num_feat_levels):
            self.t_heat[i][-1].conv.reset_parameters()
            self.t_heat[i][-1].conv.bias.data.fill_(bias_init)
            self.l_heat[i][-1].conv.reset_parameters()
            self.l_heat[i][-1].conv.bias.data.fill_(bias_init)
            self.b_heat[i][-1].conv.reset_parameters()
            self.b_heat[i][-1].conv.bias.data.fill_(bias_init)
            self.r_heat[i][-1].conv.reset_parameters()
            self.r_heat[i][-1].conv.bias.data.fill_(bias_init)

            self.t_off[i][-1].conv.reset_parameters()
            self.l_off[i][-1].conv.reset_parameters()
            self.b_off[i][-1].conv.reset_parameters()
            self.r_off[i][-1].conv.reset_parameters()


            normal_init(self.t_feat_adaption[i], std=0.01)
            normal_init(self.l_feat_adaption[i], std=0.01)
            normal_init(self.b_feat_adaption[i], std=0.01)
            normal_init(self.r_feat_adaption[i], std=0.01)


            normal_init(self.t_dcn_offset[i].conv, std=0.1)
            normal_init(self.l_dcn_offset[i].conv, std=0.1)
            normal_init(self.b_dcn_offset[i].conv, std=0.1)
            normal_init(self.r_dcn_offset[i].conv, std=0.1)


            _ = [x.conv.reset_parameters() for x in self.t_guiding_shift[i]]
            _ = [x.conv.reset_parameters() for x in self.l_guiding_shift[i]]
            _ = [x.conv.reset_parameters() for x in self.b_guiding_shift[i]]
            _ = [x.conv.reset_parameters() for x in self.r_guiding_shift[i]]
        
            _ = [
                x.conv.reset_parameters() for x in self.t_centripetal_shift[i]
            ]
            _ = [
                x.conv.reset_parameters() for x in self.l_centripetal_shift[i]
            ]
            _ = [
                x.conv.reset_parameters() for x in self.b_centripetal_shift[i]
            ]
            _ = [
                x.conv.reset_parameters() for x in self.r_centripetal_shift[i]
            ]

    def forward_single(self, x, lvl_ind):
        """Forward feature of a single level.

        Args:
            x (Tensor): Feature of a single level.
            lvl_ind (int): Level index of current feature.

        Returns:
            tuple[Tensor]: A tuple of AKSHead's output for current
            feature level. Containing the following Tensors:

                - t_heat (Tensor): Predicted top heatmap.
                - l_heat (Tensor): Predicted left heatmap.
                - b_heat (Tensor): Predicted bottom heatmap.
                - r_heat (Tensor): Predicted right heatmap.

                - t_off (Tensor): Predicted top offset heatmap.
                - l_off (Tensor): Predicted left offset heatmap.
                - b_off (Tensor): Predicted bottom offset heatmap.
                - r_off (Tensor): Predicted right offset heatmap.
 
                - t_guiding_shift (Tensor): Predicted top guiding shift heatmap.
                - l_guiding_shift (Tensor): Predicted left guiding shift heatmap.
                - b_guiding_shift (Tensor): Predicted bottom guiding shift heatmap.
                - r_guiding_shift (Tensor): Predicted right guiding shift heatmap.

                - t_centripetal_shift (Tensor): Predicted top centripetal shift heatmap.
                - l_centripetal_shift (Tensor): Predicted left centripetal shift heatmap.
                - b_centripetal_shift (Tensor): Predicted bottom centripetal shift heatmap.
                - r_centripetal_shift (Tensor): Predicted right centripetal shift heatmap.
        """
        t_pool = self.t_pool[lvl_ind](x)
        t_heat = self.t_heat[lvl_ind](t_pool)
        l_pool = self.l_pool[lvl_ind](x)
        l_heat = self.l_heat[lvl_ind](l_pool)
        b_pool = self.b_pool[lvl_ind](x)
        b_heat = self.b_heat[lvl_ind](b_pool)
        r_pool = self.r_pool[lvl_ind](x)
        r_heat = self.r_heat[lvl_ind](r_pool)

        t_off = self.t_off[lvl_ind](t_pool)
        l_off = self.l_off[lvl_ind](l_pool)
        b_off = self.b_off[lvl_ind](b_pool)
        r_off = self.r_off[lvl_ind](r_pool)


        t_guiding_shift = self.t_guiding_shift[lvl_ind](t_pool)
        l_guiding_shift = self.l_guiding_shift[lvl_ind](l_pool)
        b_guiding_shift = self.b_guiding_shift[lvl_ind](b_pool)
        r_guiding_shift = self.r_guiding_shift[lvl_ind](r_pool)

        t_dcn_offset = self.t_dcn_offset[lvl_ind](t_guiding_shift.detach())
        l_dcn_offset = self.l_dcn_offset[lvl_ind](l_guiding_shift.detach())
        b_dcn_offset = self.b_dcn_offset[lvl_ind](b_guiding_shift.detach())
        r_dcn_offset = self.r_dcn_offset[lvl_ind](r_guiding_shift.detach())

        t_feat_adaption = self.t_feat_adaption[lvl_ind](t_pool,
                                                          t_dcn_offset)
        l_feat_adaption = self.l_feat_adaption[lvl_ind](l_pool,
                                                          l_dcn_offset)
        b_feat_adaption = self.b_feat_adaption[lvl_ind](b_pool,
                                                          b_dcn_offset)
        r_feat_adaption = self.r_feat_adaption[lvl_ind](r_pool,
                                                          r_dcn_offset)

        t_centripetal_shift = self.t_centripetal_shift[lvl_ind](
            t_feat_adaption)
        l_centripetal_shift = self.l_centripetal_shift[lvl_ind](
            l_feat_adaption)
        b_centripetal_shift = self.b_centripetal_shift[lvl_ind](
            b_feat_adaption)
        r_centripetal_shift = self.r_centripetal_shift[lvl_ind](
            r_feat_adaption)

        result_list = [
            t_heat, l_heat,b_heat, r_heat,
            t_off, l_off, b_off, r_off, 
            t_guiding_shift,l_guiding_shift,
            b_guiding_shift,r_guiding_shift, t_centripetal_shift, l_centripetal_shift,
            b_centripetal_shift, r_centripetal_shift
        ]
        return result_list

    def get_targets(self,
                    gt_bboxes,
                    gt_keypoints,
                    gt_labels,
                    feat_shape,
                    img_shape,
                    with_corner_emb=False,
                    with_guiding_shift=False,
                    with_centripetal_shift=False):
        """Generate 4-keypoints targets.

        Including 4-keypoints heatmap, 4-keypoints offset.

        Optional: 4-keypoints guiding shift, centripetal shift.

        For CentripetalNet, we generate corner heatmap, corner offset, guiding
        shift and centripetal shift from this function.

        For AKSnet, we generate 4-keypoints heatmap, 4-keypoints offset, guiding
        shift and centripetal shift from this function.

        Args:
            gt_bboxes (list[Tensor]): Ground truth bboxes of each image, each
                has shape (num_gt, 4).
            gt_keypoints (list[Tensor]):Ground truth bboxes of each image, each
                has shape (num_gt, 8).top-left-bottom-right
            gt_labels (list[Tensor]): Ground truth labels of each box, each has
                shape (num_gt,).
            feat_shape (list[int]): Shape of output feature,
                [batch, channel, height, width].
            img_shape (list[int]): Shape of input image,
                [height, width, channel].
            with_corner_emb (bool): Generate corner embedding target or not.
                Default: False.
            with_guiding_shift (bool): Generate guiding shift target or not.
                Default: False.
            with_centripetal_shift (bool): Generate centripetal shift target or
                not. Default: False.

        Returns:
            dict: Ground truth of 4-keypoints heatmap, 4-keypoints offset, corner
            embedding, guiding shift and centripetal shift. Containing the
            following keys:

                - top_heatmap (Tensor): Ground truth top keypoint
                  heatmap.
                - left_heatmap (Tensor): Ground truth left keypoint
                  heatmap.
                - bottom_heatmap (Tensor): Ground truth bottom keypoint
                  heatmap.
                - right_heatmap (Tensor): Ground truth right
                  keypoint heatmap.
                - top_offset (Tensor): Ground truth top keypoint offset.
                - left_offset (Tensor): Ground truth left keypoint offset.
                - bottom_offset (Tensor): Ground truth bottom keypoint
                  offset.
                - right_offset (Tensor): Ground truth right keypoint
                  offset.
                - corner_embedding (list[list[list[int]]]): Ground truth corner
                  embedding. Not must have.
                - top_guiding_shift (Tensor): Ground truth top keypoint
                  guiding shift. Not must have.
                - left_guiding_shift (Tensor): Ground truth left keypoint
                  guiding shift. Not must have.
                - bottom_guiding_shift (Tensor): Ground truth bottom
                  keypoint guiding shift. Not must have.
                - right_guiding_shift (Tensor): Ground truth right
                  keypoint guiding shift. Not must have.
                - top_centripetal_shift (Tensor): Ground truth top
                  keypoint centripetal shift. Not must have.
                - left_centripetal_shift (Tensor): Ground truth
                  left keypoint centripetal shift. Not must have.
                - bottom_centripetal_shift (Tensor): Ground truth bottom
                  keypoint centripetal shift. Not must have.
                - right_centripetal_shift (Tensor): Ground truth
                  right keypoint centripetal shift. Not must have.
        """
        batch_size, _, height, width = feat_shape
        img_h, img_w = img_shape[:2]

        width_ratio = float(width / img_w)
        height_ratio = float(height / img_h)

        gt_t_heatmap = gt_bboxes[-1].new_zeros(
            [batch_size, self.num_classes, height, width])
        gt_l_heatmap = gt_bboxes[-1].new_zeros(
            [batch_size, self.num_classes, height, width])
        gt_b_heatmap = gt_bboxes[-1].new_zeros(
            [batch_size, self.num_classes, height, width])
        gt_r_heatmap = gt_bboxes[-1].new_zeros(
            [batch_size, self.num_classes, height, width])

        gt_t_offset = gt_bboxes[-1].new_zeros([batch_size, 2, height, width])
        gt_l_offset = gt_bboxes[-1].new_zeros([batch_size, 2, height, width])
        gt_b_offset = gt_bboxes[-1].new_zeros([batch_size, 2, height, width])
        gt_r_offset = gt_bboxes[-1].new_zeros([batch_size, 2, height, width])

        if with_corner_emb:
            match = []

        # Guiding shift is a kind of offset, from center to keypoint
        if with_guiding_shift:
            gt_t_guiding_shift = gt_bboxes[-1].new_zeros(
                [batch_size, 2, height, width])
            gt_l_guiding_shift = gt_bboxes[-1].new_zeros(
                [batch_size, 2, height, width])
            gt_b_guiding_shift = gt_bboxes[-1].new_zeros(
                [batch_size, 2, height, width])
            gt_r_guiding_shift = gt_bboxes[-1].new_zeros(
                [batch_size, 2, height, width])

        # Centripetal shift is also a kind of offset, from center to keypoint
        # and normalized by log.
        if with_centripetal_shift:
            gt_t_centripetal_shift = gt_bboxes[-1].new_zeros(
                [batch_size, 2, height, width])
            gt_l_centripetal_shift = gt_bboxes[-1].new_zeros(
                [batch_size, 2, height, width])
            gt_b_centripetal_shift = gt_bboxes[-1].new_zeros(
                [batch_size, 2, height, width])
            gt_r_centripetal_shift = gt_bboxes[-1].new_zeros(
                [batch_size, 2, height, width])

        for batch_id in range(batch_size):
            # Ground truth of corner embedding per image is a list of coord set
            corner_match = []
            for box_id in range(len(gt_labels[batch_id])):
                left, top, right, bottom = gt_bboxes[batch_id][box_id]
                center_x = (left + right) / 2.0
                center_y = (top + bottom) / 2.0
                label = gt_labels[batch_id][box_id]

                topx,topy,leftx,lefty, bottomx,bottomy, rightx,righty  = gt_keypoints[batch_id][box_id]
                # Use coords in the feature level to generate ground truth
                scale_left = left * width_ratio
                scale_right = right * width_ratio
                scale_top = top * height_ratio
                scale_bottom = bottom * height_ratio
                scale_center_x = center_x * width_ratio
                scale_center_y = center_y * height_ratio
                
                scale_topx,scale_leftx,scale_bottomx,scale_rightx = topx*width_ratio,leftx*width_ratio,bottomx*width_ratio,rightx*width_ratio

                scale_topy,scale_lefty,scale_bottomy,scale_righty = topy*height_ratio,lefty*height_ratio,bottomy*height_ratio,righty*height_ratio

                # Int coords on feature map/ground truth tensor
                left_idx = int(min(scale_left, width - 1))
                right_idx = int(min(scale_right, width - 1))
                top_idx = int(min(scale_top, height - 1))
                bottom_idx = int(min(scale_bottom, height - 1))

                leftx_idx = int(min(scale_leftx, width - 1))
                rightx_idx = int(min(scale_rightx, width - 1))
                topx_idx = int(min(scale_topx, width - 1))
                bottomx_idx = int(min(scale_bottomx, width - 1))
                lefty_idx = int(min(scale_lefty, height - 1))
                righty_idx = int(min(scale_righty, height - 1))
                topy_idx = int(min(scale_topy, height - 1))
                bottomy_idx = int(min(scale_bottomy, height - 1))

                # Generate gaussian heatmap
                scale_box_width = ceil(scale_right - scale_left)
                scale_box_height = ceil(scale_bottom - scale_top)
                radius = gaussian_radius((scale_box_height, scale_box_width),
                                         min_overlap=0.3)
                radius = max(0, int(radius))

                gt_t_heatmap[batch_id, label] = gen_gaussian_target(
                    gt_t_heatmap[batch_id, label], [topx_idx, topy_idx],
                    radius)
                gt_l_heatmap[batch_id, label] = gen_gaussian_target(
                    gt_l_heatmap[batch_id, label], [leftx_idx, lefty_idx],
                    radius)
                gt_b_heatmap[batch_id, label] = gen_gaussian_target(
                    gt_b_heatmap[batch_id, label], [bottomx_idx, bottomy_idx],
                    radius)
                gt_r_heatmap[batch_id, label] = gen_gaussian_target(
                    gt_r_heatmap[batch_id, label], [rightx_idx, righty_idx],
                    radius)
                # Generate keypoints offset

                leftx_offset = scale_leftx - leftx_idx
                topx_offset = scale_topx - topx_idx
                rightx_offset = scale_rightx - rightx_idx
                bottomx_offset = scale_bottomx - bottomx_idx
                lefty_offset = scale_lefty - lefty_idx
                topy_offset = scale_topy - topy_idx
                righty_offset = scale_righty - righty_idx
                bottomy_offset = scale_bottomy - bottomy_idx
                gt_t_offset[batch_id, 0, topx_idx, topy_idx] = topx_offset
                gt_t_offset[batch_id, 1, topx_idx, topy_idx] = topy_offset
                gt_l_offset[batch_id, 0, leftx_idx, lefty_idx] = leftx_offset
                gt_l_offset[batch_id, 1, leftx_idx,
                             lefty_idx] = lefty_offset
                gt_b_offset[batch_id, 0, bottomx_idx, bottomy_idx] = bottomx_offset
                gt_b_offset[batch_id, 1, bottomx_idx, bottomy_idx] = bottomx_offset
                gt_r_offset[batch_id, 0, rightx_idx, righty_idx] = rightx_offset
                gt_r_offset[batch_id, 1, rightx_idx,
                             righty_idx] = righty_offset
                # Generate corner embedding
                if with_corner_emb:
                    corner_match.append([[top_idx, left_idx],
                                         [bottom_idx, right_idx]])
                # Generate guiding shift
                if with_guiding_shift:
                    gt_t_guiding_shift[batch_id, 0, topx_idx,
                                        topy_idx] = scale_center_x - topx_idx
                    gt_t_guiding_shift[batch_id, 1, topx_idx,
                                        topy_idx] = scale_center_y - topy_idx
                    gt_l_guiding_shift[batch_id, 0, leftx_idx,
                                        lefty_idx] = scale_center_x - leftx_idx
                    gt_l_guiding_shift[batch_id, 1, leftx_idx,
                                        lefty_idx] = scale_center_y - lefty_idx
                    gt_b_guiding_shift[batch_id, 0, bottomx_idx,
                                        bottomy_idx] = scale_center_x - bottomx_idx
                    gt_b_guiding_shift[
                        batch_id, 1, bottomx_idx,
                        bottomy_idx] = scale_center_y - bottomy_idx

                    gt_r_guiding_shift[batch_id, 0, rightx_idx,
                                        righty_idx] = scale_center_x - rightx_idx
                    gt_r_guiding_shift[
                        batch_id, 1, rightx_idx,
                        righty_idx] = scale_center_y - righty_idx
                # Generate centripetal shift
                if with_centripetal_shift:
                    gt_t_centripetal_shift[batch_id, 0, topx_idx,
                                            topy_idx] = (scale_center_x -
                                                            scale_left)
                    gt_t_centripetal_shift[batch_id, 1, topx_idx,
                                            topy_idx] = (scale_center_y -
                                                            scale_topy)
                    gt_l_centripetal_shift[batch_id, 0, leftx_idx,
                                            lefty_idx] = (scale_center_x -
                                                            scale_leftx)
                    gt_l_centripetal_shift[batch_id, 1, leftx_idx,
                                            lefty_idx] = (scale_center_y -
                                                            scale_topy)
                    gt_b_centripetal_shift[batch_id, 0, bottomx_idx,
                                            bottomy_idx] = (scale_center_x -
                                                             scale_bottomx)
                    gt_b_centripetal_shift[batch_id, 1, bottomx_idx,
                                            bottomy_idx] = (scale_center_y -
                                                             scale_bottomy)
                    gt_r_centripetal_shift[batch_id, 0, rightx_idx,
                                            righty_idx] = (scale_center_x -
                                                             scale_rightx)
                    gt_r_centripetal_shift[batch_id, 1, rightx_idx,
                                            righty_idx] = (scale_center_y -
                                                             scale_righty)


            if with_corner_emb:
                match.append(corner_match)

        target_result = dict(
            top_heatmap=gt_t_heatmap,
            top_offset=gt_t_offset,
            left_heatmap=gt_l_heatmap,
            left_offset=gt_l_offset,
            bottom_heatmap=gt_b_heatmap,
            bottom_offset=gt_b_offset,
            right_heatmap=gt_r_heatmap,
            right_offset=gt_r_offset)

        if with_corner_emb:
            target_result.update(corner_embedding=match)
        if with_guiding_shift:
            target_result.update(
                top_guiding_shift=gt_t_guiding_shift,
                left_guiding_shift=gt_l_guiding_shift,
                bottom_guiding_shift=gt_b_guiding_shift,
                right_guiding_shift=gt_r_guiding_shift)
        if with_centripetal_shift:
            target_result.update(
                top_centripetal_shift=gt_t_centripetal_shift,
                left_centripetal_shift=gt_l_centripetal_shift,
                bottom_centripetal_shift=gt_b_centripetal_shift,
                right_centripetal_shift=gt_r_centripetal_shift)

        return target_result

    def loss(self,
             t_heats,
             l_heats,
             b_heats,
             r_heats,
             t_offs,
             l_offs,
             b_offs,
             r_offs,
             t_guiding_shifts,
             l_guiding_shifts,
             b_guiding_shifts,
             r_guiding_shifts,
             t_centripetal_shifts,
             l_centripetal_shifts,
             b_centripetal_shifts,
             r_centripetal_shifts,
             gt_bboxes,
             gt_keypoints,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute losses of the head.

        Args:
            tl_heats (list[Tensor]): Top-left corner heatmaps for each level
                with shape (N, num_classes, H, W).
            br_heats (list[Tensor]): Bottom-right corner heatmaps for each
                level with shape (N, num_classes, H, W).
            tl_embs (list[Tensor]): Top-left corner embeddings for each level
                with shape (N, corner_emb_channels, H, W).
            br_embs (list[Tensor]): Bottom-right corner embeddings for each
                level with shape (N, corner_emb_channels, H, W).
            tl_offs (list[Tensor]): Top-left corner offsets for each level
                with shape (N, corner_offset_channels, H, W).
            br_offs (list[Tensor]): Bottom-right corner offsets for each level
                with shape (N, corner_offset_channels, H, W).
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [left, top, right, bottom] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (list[Tensor] | None): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components. Containing the
            following losses:

                - det_loss (list[Tensor]): Corner keypoint losses of all
                  feature levels.
                - pull_loss (list[Tensor]): Part one of AssociativeEmbedding
                  losses of all feature levels.
                - push_loss (list[Tensor]): Part two of AssociativeEmbedding
                  losses of all feature levels.
                - off_loss (list[Tensor]): Corner offset losses of all feature
                  levels.
        """
        targets = self.get_targets(
            gt_bboxes,
            gt_keypoints,
            gt_labels,
            tl_heats[-1].shape,
            img_metas[0]['pad_shape'],
            with_corner_emb=self.with_corner_emb)
        mlvl_targets = [targets for _ in range(self.num_feat_levels)]
        [det_losses, off_losses, guiding_losses, centripetal_losses] = multi_apply(
            self.loss_single, t_heats, l_heats,b_heats, r_heats, t_offs,l_offs,
            b_offs,r_offs,t_guiding_shifts,l_guiding_shifts, b_guiding_shifts, r_guiding_shifts,
            t_centripetal_shifts,l_centripetal_shifts, b_centripetal_shifts, r_centripetal_shifts, mlvl_targets)
        loss_dict = dict(det_loss=det_losses, off_loss=off_losses,)

        loss_dict = dict(
            det_loss=det_losses,
            off_loss=off_losses,
            guiding_loss=guiding_losses,
            centripetal_loss=centripetal_losses)
        return loss_dict

    def loss_single(self, t_hmp, l_hmp,b_hmp, r_hmp, 
                    t_off,l_off,b_off,r_off,
                    t_guiding_shift,l_guiding_shift, b_guiding_shift, r_guiding_shift,
                    t_centripetal_shift,l_centripetal_shift, b_centripetal_shift, r_centripetal_shift,
                    targets):
        """Compute losses for single level.

        Args:
            tl_hmp (Tensor): Top-left corner heatmap for current level with
                shape (N, num_classes, H, W).
            br_hmp (Tensor): Bottom-right corner heatmap for current level with
                shape (N, num_classes, H, W).
            tl_emb (Tensor): Top-left corner embedding for current level with
                shape (N, corner_emb_channels, H, W).
            br_emb (Tensor): Bottom-right corner embedding for current level
                with shape (N, corner_emb_channels, H, W).
            tl_off (Tensor): Top-left corner offset for current level with
                shape (N, corner_offset_channels, H, W).
            br_off (Tensor): Bottom-right corner offset for current level with
                shape (N, corner_offset_channels, H, W).
            targets (dict): Corner target generated by `get_targets`.

        Returns:
            tuple[torch.Tensor]: Losses of the head's differnet branches
            containing the following losses:

                - det_loss (Tensor): Corner keypoint loss.
                - pull_loss (Tensor): Part one of AssociativeEmbedding loss.
                - push_loss (Tensor): Part two of AssociativeEmbedding loss.
                - off_loss (Tensor): Corner offset loss.
        """
        gt_t_hmp = targets['top_heatmap']
        gt_l_hmp = targets['left_heatmap']
        gt_b_hmp = targets['bottom_heatmap']
        gt_r_hmp = targets['right_heatmap']
        gt_t_off = targets['top_offset']
        gt_l_off = targets['left_offset']
        gt_b_off = targets['bottom_offset']
        gt_r_off = targets['right_offset']

        # Detection loss
        t_det_loss = self.loss_heatmap(
            t_hmp.sigmoid(),
            gt_t_hmp,
            avg_factor=max(1,
                           gt_t_hmp.eq(1).sum()))
        l_det_loss = self.loss_heatmap(
            l_hmp.sigmoid(),
            gt_l_hmp,
            avg_factor=max(1,
                           gt_l_hmp.eq(1).sum()))
        b_det_loss = self.loss_heatmap(
            b_hmp.sigmoid(),
            gt_b_hmp,
            avg_factor=max(1,
                           gt_b_hmp.eq(1).sum()))
        r_det_loss = self.loss_heatmap(
            r_hmp.sigmoid(),
            gt_r_hmp,
            avg_factor=max(1,
                           gt_r_hmp.eq(1).sum()))


        det_loss = (t_det_loss + l_det_loss + b_det_loss + r_det_loss) / 4.0


        # Offset loss
        # We only compute the offset loss at the real corner position.
        # The value of real corner would be 1 in heatmap ground truth.
        # The mask is computed in class agnostic mode and its shape is
        # batch * 1 * width * height.
        t_off_mask = gt_t_hmp.eq(1).sum(1).gt(0).unsqueeze(1).type_as(
            gt_t_hmp)
        l_off_mask = gt_l_hmp.eq(1).sum(1).gt(0).unsqueeze(1).type_as(
            gt_l_hmp)
        b_off_mask = gt_b_hmp.eq(1).sum(1).gt(0).unsqueeze(1).type_as(
            gt_b_hmp)
        r_off_mask = gt_r_hmp.eq(1).sum(1).gt(0).unsqueeze(1).type_as(
            gt_r_hmp)
        t_off_loss = self.loss_offset(
            t_off,
            gt_tl_off,
            t_off_mask,
            avg_factor=max(1, t_off_mask.sum()))
        l_off_loss = self.loss_offset(
            l_off,
            gt_l_off,
            l_off_mask,
            avg_factor=max(1, l_off_mask.sum()))
        b_off_loss = self.loss_offset(
            b_off,
            gt_b_off,
            b_off_mask,
            avg_factor=max(1, b_off_mask.sum()))
        r_off_loss = self.loss_offset(
            r_off,
            gt_r_off,
            r_off_mask,
            avg_factor=max(1, r_off_mask.sum()))

        off_loss = (t_off_loss + l_off_loss + b_off_loss + r_off_loss) / 4.0

        ##################
        gt_t_guiding_shift = targets['top_guiding_shift']
        gt_l_guiding_shift = targets['left_guiding_shift']
        gt_b_guiding_shift = targets['bottom_guiding_shift']
        gt_r_guiding_shift = targets['right_guiding_shift']
        gt_t_centripetal_shift = targets['top_centripetal_shift']
        gt_l_centripetal_shift = targets['left_centripetal_shift']
        gt_b_centripetal_shift = targets['bottom_centripetal_shift']
        gt_r_centripetal_shift = targets['right_centripetal_shift']

        gt_t_heatmap = targets['top_heatmap']
        gt_l_heatmap = targets['left_heatmap']
        gt_b_heatmap = targets['bottom_heatmap']
        gt_r_heatmap = targets['right_heatmap']
        # We only compute the offset loss at the real corner position.
        # The value of real corner would be 1 in heatmap ground truth.
        # The mask is computed in class agnostic mode and its shape is
        # batch * 1 * width * height.
        t_mask = gt_t_heatmap.eq(1).sum(1).gt(0).unsqueeze(1).type_as(
            gt_t_heatmap)
        l_mask = gt_l_heatmap.eq(1).sum(1).gt(0).unsqueeze(1).type_as(
            gt_l_heatmap)
        b_mask = gt_b_heatmap.eq(1).sum(1).gt(0).unsqueeze(1).type_as(
            gt_b_heatmap)
        r_mask = gt_r_heatmap.eq(1).sum(1).gt(0).unsqueeze(1).type_as(
            gt_r_heatmap)

        # Guiding shift loss
        t_guiding_loss = self.loss_guiding_shift(
            t_guiding_shift,
            gt_t_guiding_shift,
            t_mask,
            avg_factor=t_mask.sum())
        l_guiding_loss = self.loss_guiding_shift(
            l_guiding_shift,
            gt_l_guiding_shift,
            l_mask,
            avg_factor=l_mask.sum())
        b_guiding_loss = self.loss_guiding_shift(
            b_guiding_shift,
            gt_b_guiding_shift,
            b_mask,
            avg_factor=b_mask.sum())
        r_guiding_loss = self.loss_guiding_shift(
            r_guiding_shift,
            gt_r_guiding_shift,
            r_mask,
            avg_factor=r_mask.sum())
        guiding_loss = (t_guiding_loss + l_guiding_loss + b_guiding_loss + r_guiding_loss) / 4.0
        # Centripetal shift loss
        t_centripetal_loss = self.loss_centripetal_shift(
            t_centripetal_shift,
            gt_t_centripetal_shift,
            t_mask,
            avg_factor=t_mask.sum())
        l_centripetal_loss = self.loss_centripetal_shift(
            l_centripetal_shift,
            gt_l_centripetal_shift,
            l_mask,
            avg_factor=l_mask.sum())
        b_centripetal_loss = self.loss_centripetal_shift(
            b_centripetal_shift,
            gt_b_centripetal_shift,
            b_mask,
            avg_factor=b_mask.sum())
        r_centripetal_loss = self.loss_centripetal_shift(
            r_centripetal_shift,
            gt_r_centripetal_shift,
            r_mask,
            avg_factor=r_mask.sum())
        centripetal_loss = (t_centripetal_loss + l_centripetal_loss + b_centripetal_loss + r_centripetal_loss) / 4.0

        return det_loss, off_loss, guiding_loss, centripetal_loss


    def get_bboxes(self,
                   t_heats,
                   l_heats,
                   b_heats,
                   r_heats,
                   t_offs,
                   l_offs,
                   b_offs,
                   r_offs,
                   t_guiding_shifts,
                   l_guiding_shifts,
                   b_guiding_shifts,
                   r_guiding_shifts,
                   t_centripetal_shifts,
                   l_centripetal_shifts,
                   b_centripetal_shifts,
                   r_centripetal_shifts,
                   img_metas,
                   rescale=False,
                   with_nms=True):
        """Transform network output for a batch into bbox predictions.

        Args:
            tl_heats (list[Tensor]): Top-left corner heatmaps for each level
                with shape (N, num_classes, H, W).
            br_heats (list[Tensor]): Bottom-right corner heatmaps for each
                level with shape (N, num_classes, H, W).
            tl_embs (list[Tensor]): Top-left corner embeddings for each level
                with shape (N, corner_emb_channels, H, W).
            br_embs (list[Tensor]): Bottom-right corner embeddings for each
                level with shape (N, corner_emb_channels, H, W).
            tl_offs (list[Tensor]): Top-left corner offsets for each level
                with shape (N, corner_offset_channels, H, W).
            br_offs (list[Tensor]): Bottom-right corner offsets for each level
                with shape (N, corner_offset_channels, H, W).
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.
        """
        assert t_heats[-1].shape[0] == l_heats[-1].shape[0] == b_heats[-1].shape[0] == r_heats[-1].shape[0] == len(img_metas)
        result_list = []
        for img_id in range(len(img_metas)):
            result_list.append(
                self._get_bboxes_single(
                    t_heats[-1][img_id:img_id + 1, :],
                    l_heats[-1][img_id:img_id + 1, :],
                    b_heats[-1][img_id:img_id + 1, :],
                    r_heats[-1][img_id:img_id + 1, :],
                    t_offs[-1][img_id:img_id + 1, :],
                    l_offs[-1][img_id:img_id + 1, :],
                    b_offs[-1][img_id:img_id + 1, :],
                    r_offs[-1][img_id:img_id + 1, :],
                    img_metas[img_id],
                    tl_emb=None,
                    br_emb=None,
                    t_centripetal_shift=tl_centripetal_shifts[-1][
                        img_id:img_id + 1, :],
                    l_centripetal_shift=tl_centripetal_shifts[-1][
                        img_id:img_id + 1, :],
                    b_centripetal_shift=br_centripetal_shifts[-1][
                        img_id:img_id + 1, :],
                    r_centripetal_shift=br_centripetal_shifts[-1][
                        img_id:img_id + 1, :],
                    rescale=rescale,
                    with_nms=with_nms))

        return result_list



    def _get_bboxes_single(self,
                           t_heat,
                           l_heat,
                           b_heat,
                           r_heat,
                           t_off,
                           l_off,
                           b_off,
                           r_off,
                           img_meta,
                           tl_emb=None,
                           br_emb=None,
                           t_centripetal_shift=None,
                           l_centripetal_shift=None,
                           b_centripetal_shift=None,
                           r_centripetal_shift=None,
                           rescale=False,
                           with_nms=True):
        """Transform outputs for a single batch item into bbox predictions.
        """
        if isinstance(img_meta, (list, tuple)):
            img_meta = img_meta[0]
        # tb_bboxes,lr_bboxes, tb_scores,lr_scores, tb_clses,lr_clses
        tb_batch_bboxes,lr_batch_bboxes, tb_batch_scores,lr_batch_scores, tb_batch_clses,lr_batch_clses = self.decode_heatmap(
            t_heat=t_heat.sigmoid(),
            l_heat=l_heat.sigmoid(),
            b_heat=b_heat.sigmoid(),
            r_heat=r_heat.sigmoid(),
            t_off=t_off,
            l_off=l_off,
            b_off=b_off,
            r_off=r_off,
            tl_emb=tl_emb,
            br_emb=br_emb,
            t_centripetal_shift=t_centripetal_shift,
            l_centripetal_shift=l_centripetal_shift,
            b_centripetal_shift=b_centripetal_shift,
            r_centripetal_shift=r_centripetal_shift,
            img_meta=img_meta,
            k=self.test_cfg.corner_topk,
            kernel=self.test_cfg.local_maximum_kernel,
            distance_threshold=self.test_cfg.distance_threshold)

        if rescale:
            tb_batch_bboxes /= tb_batch_bboxes.new_tensor(img_meta['scale_factor'])
            lr_batch_bboxes /= lr_batch_bboxes.new_tensor(img_meta['scale_factor'])

        tb_bboxes = tb_batch_bboxes.view([-1, 4])
        tb_scores = tb_batch_scores.view([-1, 1])
        tb_clses = tb_batch_clses.view([-1, 1])
        lr_bboxes = lr_batch_bboxes.view([-1, 4])
        lr_scores = lr_batch_scores.view([-1, 1])
        lr_clses = lr_batch_clses.view([-1, 1])

        tb_idx = tb_scores.argsort(dim=0, descending=True)
        tb_bboxes = tb_bboxes[idx].view([-1, 4])
        tb_scores = tb_scores[idx].view(-1)
        tb_clses = tb_clses[idx].view(-1)
        lr_idx = lr_scores.argsort(dim=0, descending=True)
        lr_bboxes = lr_bboxes[idx].view([-1, 4])
        lr_scores = lr_scores[idx].view(-1)
        lr_clses = lr_clses[idx].view(-1)

        tb_detections = torch.cat([tb_bboxes, tb_scores.unsqueeze(-1)], -1)
        tb_keepinds = (tb_detections[:, -1] > -0.1)
        tb_detections = tb_detections[tb_keepinds]
        tb_labels = tb_clses[tb_keepinds]
        lr_detections = torch.cat([lr_bboxes, lr_scores.unsqueeze(-1)], -1)
        lr_keepinds = (lr_detections[:, -1] > -0.1)
        lr_detections = lr_detections[lr_keepinds]
        lr_labels = lr_clses[lr_keepinds]

        if with_nms:
            tb_detections, tb_labels = self._bboxes_nms(tb_detections, tb_labels,
                                                  self.test_cfg)
            lr_detections, lr_labels = self._bboxes_nms(lr_detections, lr_labels,
                                                  self.test_cfg)
        return tb_detections, tb_labels,lr_detections, lr_labels

    def _bboxes_nms(self, bboxes, labels, cfg):
        if labels.numel() == 0:
            return bboxes, labels

        if 'nms_cfg' in cfg:
            warning.warn('nms_cfg in test_cfg will be deprecated. '
                         'Please rename it as nms')
        if 'nms' not in cfg:
            cfg.nms = cfg.nms_cfg

        out_bboxes, keep = batched_nms(bboxes[:, :4], bboxes[:, -1], labels,
                                       cfg.nms)
        out_labels = labels[keep]

        if len(out_bboxes) > 0:
            idx = torch.argsort(out_bboxes[:, -1], descending=True)
            idx = idx[:cfg.max_per_img]
            out_bboxes = out_bboxes[idx]
            out_labels = out_labels[idx]

        return out_bboxes, out_labels

    def _gather_feat(self, feat, ind, mask=None):
        """Gather feature according to index.

        Args:
            feat (Tensor): Target feature map.
            ind (Tensor): Target coord index.
            mask (Tensor | None): Mask of featuremap. Default: None.

        Returns:
            feat (Tensor): Gathered feature.
        """
        dim = feat.size(2)
        ind = ind.unsqueeze(2).repeat(1, 1, dim)
        feat = feat.gather(1, ind)
        if mask is not None:
            mask = mask.unsqueeze(2).expand_as(feat)
            feat = feat[mask]
            feat = feat.view(-1, dim)
        return feat
    def _local_maximum(self, heat, kernel=3):
        """Extract local maximum pixel with given kernal.

        Args:
            heat (Tensor): Target heatmap.
            kernel (int): Kernel size of max pooling. Default: 3.

        Returns:
            heat (Tensor): A heatmap where local maximum pixels maintain its
                own value and other positions are 0.
        """
        pad = (kernel - 1) // 2
        hmax = F.max_pool2d(heat, kernel, stride=1, padding=pad)
        keep = (hmax == heat).float()
        return heat * keep

    def _transpose_and_gather_feat(self, feat, ind):
        """Transpose and gather feature according to index.

        Args:
            feat (Tensor): Target feature map.
            ind (Tensor): Target coord index.

        Returns:
            feat (Tensor): Transposed and gathered feature.
        """
        feat = feat.permute(0, 2, 3, 1).contiguous()
        feat = feat.view(feat.size(0), -1, feat.size(3))
        feat = self._gather_feat(feat, ind)
        return feat

    def _topk(self, scores, k=20):
        """Get top k positions from heatmap.

        Args:
            scores (Tensor): Target heatmap with shape
                [batch, num_classes, height, width].
            k (int): Target number. Default: 20.

        Returns:
            tuple[torch.Tensor]: Scores, indexes, categories and coords of
                topk keypoint. Containing following Tensors:

            - topk_scores (Tensor): Max scores of each topk keypoint.
            - topk_inds (Tensor): Indexes of each topk keypoint.
            - topk_clses (Tensor): Categories of each topk keypoint.
            - topk_ys (Tensor): Y-coord of each topk keypoint.
            - topk_xs (Tensor): X-coord of each topk keypoint.
        """
        batch, _, height, width = scores.size()
        topk_scores, topk_inds = torch.topk(scores.view(batch, -1), k)
        topk_clses = topk_inds // (height * width)
        topk_inds = topk_inds % (height * width)
        topk_ys = topk_inds // width
        topk_xs = (topk_inds % width).int().float()
        return topk_scores, topk_inds, topk_clses, topk_ys, topk_xs

    def decode_heatmap(self,
                       t_heat,
                       l_heat,
                       b_heat,
                       r_heat,
                       t_off,
                       l_off,
                       b_off,
                       r_off,
                       tl_emb=None,
                       br_emb=None,
                       t_centripetal_shift=None,
                       l_centripetal_shift=None,
                       b_centripetal_shift=None,
                       r_centripetal_shift=None,
                       img_meta=None,
                       k=100,
                       kernel=3,
                       distance_threshold=0.5,
                       num_dets=1000):
        """Transform outputs for a single batch item into raw bbox predictions.
        Returns:
            tuple[torch.Tensor]: Decoded output of CornerHead, containing the
            following Tensors:

            - bboxes (Tensor): Coords of each box.
            - scores (Tensor): Scores of each box.
            - clses (Tensor): Categories of each box.
        """
        with_embedding = tl_emb is not None and br_emb is not None
        with_centripetal_shift = (
            tl_centripetal_shift is not None
            and br_centripetal_shift is not None)
        assert with_embedding + with_centripetal_shift == 1
        batch, _, height, width = t_heat.size()
        inp_h, inp_w, _ = img_meta['pad_shape']

        # perform nms on heatmaps
        t_heat = self._local_maximum(t_heat, kernel=kernel)
        l_heat = self._local_maximum(l_heat, kernel=kernel)
        b_heat = self._local_maximum(b_heat, kernel=kernel)
        r_heat = self._local_maximum(r_heat, kernel=kernel)

        t_scores, t_inds, t_clses, t_ys, t_xs = self._topk(t_heat, k=k)
        l_scores, l_inds, l_clses, l_ys, l_xs = self._topk(l_heat, k=k)
        b_scores, b_inds, b_clses, b_ys, b_xs = self._topk(b_heat, k=k)
        r_scores, r_inds, r_clses, r_ys, r_xs = self._topk(r_heat, k=k)

        # We use repeat instead of expand here because expand is a
        # shallow-copy function. Thus it could cause unexpected testing result
        # sometimes. Using expand will decrease about 10% mAP during testing
        # compared to repeat.
        t_ys = t_ys.view(batch, k, 1).repeat(1, 1, k)
        t_xs = t_xs.view(batch, k, 1).repeat(1, 1, k)
        b_ys = b_ys.view(batch, 1, k).repeat(1, k, 1)
        b_xs = b_xs.view(batch, 1, k).repeat(1, k, 1)

        l_ys = l_ys.view(batch, k, 1).repeat(1, 1, k)
        l_xs = l_xs.view(batch, k, 1).repeat(1, 1, k)
        r_ys = r_ys.view(batch, 1, k).repeat(1, k, 1)
        r_xs = r_xs.view(batch, 1, k).repeat(1, k, 1)

        t_off = self._transpose_and_gather_feat(t_off, t_inds)
        t_off = t_off.view(batch, k, 1, 2)
        b_off = self._transpose_and_gather_feat(b_off, b_inds)
        b_off = b_off.view(batch, 1, k, 2)

        l_off = self._transpose_and_gather_feat(l_off, l_inds)
        l_off = l_off.view(batch, k, 1, 2)
        r_off = self._transpose_and_gather_feat(r_off, r_inds)
        r_off = r_off.view(batch, 1, k, 2)

        t_xs = t_xs + t_off[..., 0]
        t_ys = t_ys + t_off[..., 1]
        b_xs = b_xs + b_off[..., 0]
        b_ys = b_ys + b_off[..., 1]

        l_xs = l_xs + l_off[..., 0]
        l_ys = l_ys + l_off[..., 1]
        r_xs = r_xs + r_off[..., 0]
        r_ys = r_ys + r_off[..., 1]

        if with_centripetal_shift:
            t_centripetal_shift = self._transpose_and_gather_feat(
                t_centripetal_shift, t_inds).view(batch, k, 1, 2).exp()
            b_centripetal_shift = self._transpose_and_gather_feat(
                b_centripetal_shift, b_inds).view(batch, 1, k, 2).exp()
            l_centripetal_shift = self._transpose_and_gather_feat(
                l_centripetal_shift, l_inds).view(batch, k, 1, 2).exp()
            r_centripetal_shift = self._transpose_and_gather_feat(
                r_centripetal_shift, r_inds).view(batch, 1, k, 2).exp()

            t_ctxs = t_xs + t_centripetal_shift[..., 0]
            t_ctys = t_ys + t_centripetal_shift[..., 1]
            b_ctxs = b_xs - b_centripetal_shift[..., 0]
            b_ctys = b_ys - b_centripetal_shift[..., 1]

            l_ctxs = l_xs + l_centripetal_shift[..., 0]
            l_ctys = l_ys + l_centripetal_shift[..., 1]
            r_ctxs = r_xs - r_centripetal_shift[..., 0]
            r_ctys = r_ys - r_centripetal_shift[..., 1]

        # all possible boxes based on top k top-bottoms (ignoring class)
        t_xs *= (inp_w / width)
        t_ys *= (inp_h / height)
        b_xs *= (inp_w / width)
        b_ys *= (inp_h / height)

        # all possible boxes based on top k left-rights (ignoring class)
        l_xs *= (inp_w / width)
        l_ys *= (inp_h / height)
        r_xs *= (inp_w / width)
        r_ys *= (inp_h / height)

        if with_centripetal_shift:
            t_ctxs *= (inp_w / width)
            t_ctys *= (inp_h / height)
            b_ctxs *= (inp_w / width)
            b_ctys *= (inp_h / height)

            l_ctxs *= (inp_w / width)
            l_ctys *= (inp_h / height)
            r_ctxs *= (inp_w / width)
            r_ctys *= (inp_h / height)

        x_off = img_meta['border'][2]
        y_off = img_meta['border'][0]

        t_xs -= x_off
        t_ys -= y_off
        b_xs -= x_off
        b_ys -= y_off
        l_xs -= x_off
        l_ys -= y_off
        r_xs -= x_off
        r_ys -= y_off

        t_xs *= t_xs.gt(0.0).type_as(t_xs)
        t_ys *= t_ys.gt(0.0).type_as(t_ys)
        b_xs *= b_xs.gt(0.0).type_as(b_xs)
        b_ys *= b_ys.gt(0.0).type_as(b_ys)

        l_xs *= l_xs.gt(0.0).type_as(l_xs)
        l_ys *= l_ys.gt(0.0).type_as(l_ys)
        r_xs *= r_xs.gt(0.0).type_as(r_xs)
        r_ys *= r_ys.gt(0.0).type_as(r_ys)
        

        tb_bboxes = torch.stack((t_xs, t_ys, b_xs, b_ys), dim=3)
        tb_area_bboxes = ((b_xs - t_xs) * (b_ys - t_ys)).abs()

        lr_bboxes = torch.stack((l_xs, l_ys, r_xs, r_ys), dim=3)
        lr_area_bboxes = ((l_xs - r_xs) * (l_ys - r_ys)).abs()
        if with_centripetal_shift:
            t_ctxs -= x_off
            t_ctys -= y_off
            b_ctxs -= x_off
            b_ctys -= y_off
            l_ctxs -= x_off
            l_ctys -= y_off
            r_ctxs -= x_off
            r_ctys -= y_off

            t_ctxs *= t_ctxs.gt(0.0).type_as(t_ctxs)
            t_ctys *= t_ctys.gt(0.0).type_as(t_ctys)
            b_ctxs *= b_ctxs.gt(0.0).type_as(b_ctxs)
            b_ctys *= b_ctys.gt(0.0).type_as(b_ctys)
            l_ctxs *= l_ctxs.gt(0.0).type_as(l_ctxs)
            l_ctys *= l_ctys.gt(0.0).type_as(l_ctys)
            r_ctxs *= r_ctxs.gt(0.0).type_as(r_ctxs)
            r_ctys *= r_ctys.gt(0.0).type_as(r_ctys)

            tb_ct_bboxes = torch.stack((t_ctxs, t_ctys, b_ctxs, b_ctys),
                                    dim=3)
            tb_area_ct_bboxes = ((b_ctxs - t_ctxs) * (b_ctys - t_ctys)).abs()
            lr_ct_bboxes = torch.stack((l_ctxs, l_ctys, r_ctxs, r_ctys),
                                    dim=3)
            lr_area_ct_bboxes = ((r_ctxs - l_ctxs) * (r_ctys - l_ctys)).abs()

            tb_rcentral = torch.zeros_like(tb_ct_bboxes)
            lr_rcentral = torch.zeros_like(lr_ct_bboxes)
            # magic nums from paper section 4.1
            tb_mu = torch.ones_like(tb_area_bboxes) / 2.4
            tb_mu[tb_area_bboxes > 3500] = 1 / 2.1  # large bbox have smaller mu
            lr_mu = torch.ones_like(lr_area_bboxes) / 2.4
            lr_mu[lr_area_bboxes > 3500] = 1 / 2.1  # large bbox have smaller mu

            tb_bboxes_center_x = (tb_bboxes[..., 0] + tb_bboxes[..., 2]) / 2
            tb_bboxes_center_y = (tb_bboxes[..., 1] + tb_bboxes[..., 3]) / 2

            lr_bboxes_center_x = (lr_bboxes[..., 0] + lr_bboxes[..., 2]) / 2
            lr_bboxes_center_y = (lr_bboxes[..., 1] + lr_bboxes[..., 3]) / 2

            # bboxes_center_x = (lr_bboxes_center_x + tb_bboxes_center_x) / 2
            # bboxes_center_y = (tb_bboxes_center_y + lr_bboxes_center_y) / 2

            tb_rcentral[..., 0] = tb_bboxes_center_x - tb_mu * (tb_bboxes[..., 2] -
                                                       tb_bboxes[..., 0]) / 2
            tb_rcentral[..., 1] = tb_bboxes_center_y - tb_mu * (tb_bboxes[..., 3] -
                                                       tb_bboxes[..., 1]) / 2
            tb_rcentral[..., 2] = tb_bboxes_center_x + tb_mu * (tb_bboxes[..., 2] -
                                                       tb_bboxes[..., 0]) / 2
            tb_rcentral[..., 3] = tb_bboxes_center_y + tb_mu * (tb_bboxes[..., 3] -
                                                       tb_bboxes[..., 1]) / 2
            tb_area_rcentral = ((tb_rcentral[..., 2] - tb_rcentral[..., 0]) *
                             (tb_rcentral[..., 3] - tb_rcentral[..., 1])).abs()
            tb_dists = tb_area_ct_bboxes / tb_area_rcentral

            lr_rcentral[..., 0] = lr_bboxes_center_x - lr_mu * (lr_bboxes[..., 2] -
                                                       lr_bboxes[..., 0]) / 2
            lr_rcentral[..., 1] = lr_bboxes_center_y - lr_mu * (lr_bboxes[..., 3] -
                                                       lr_bboxes[..., 1]) / 2
            lr_rcentral[..., 2] = lr_bboxes_center_x + lr_mu * (lr_bboxes[..., 2] -
                                                       lr_bboxes[..., 0]) / 2
            lr_rcentral[..., 3] = lr_bboxes_center_y + lr_mu * (lr_bboxes[..., 3] -
                                                       lr_bboxes[..., 1]) / 2
            lr_area_rcentral = ((lr_rcentral[..., 2] - lr_rcentral[..., 0]) *
                             (lr_rcentral[..., 3] - lr_rcentral[..., 1])).abs()
            lr_dists = lr_area_ct_bboxes / lr_area_rcentral

            # dists = (tb_dists+lr_dists) / 2

            t_ctx_inds = (tb_ct_bboxes[..., 0] <= tb_rcentral[..., 0]) | (
                tb_ct_bboxes[..., 0] >= tb_rcentral[..., 2])
            t_cty_inds = (tb_ct_bboxes[..., 1] <= tb_rcentral[..., 1]) | (
                tb_ct_bboxes[..., 1] >= tb_rcentral[..., 3])
            b_ctx_inds = (tb_ct_bboxes[..., 2] <= tb_rcentral[..., 0]) | (
                tb_ct_bboxes[..., 2] >= tb_rcentral[..., 2])
            b_cty_inds = (tb_ct_bboxes[..., 3] <= tb_rcentral[..., 1]) | (
                tb_ct_bboxes[..., 3] >= tb_rcentral[..., 3])
            l_ctx_inds = (lr_ct_bboxes[..., 0] <= lr_rcentral[..., 0]) | (
                lr_ct_bboxes[..., 0] >= lr_rcentral[..., 2])
            l_cty_inds = (lr_ct_bboxes[..., 1] <= lr_rcentral[..., 1]) | (
                lr_ct_bboxes[..., 1] >= lr_rcentral[..., 3])
            r_ctx_inds = (lr_ct_bboxes[..., 2] <= lr_rcentral[..., 0]) | (
                lr_ct_bboxes[..., 2] >= lr_rcentral[..., 2])
            r_cty_inds = (lr_ct_bboxes[..., 3] <= lr_rcentral[..., 1]) | (
                lr_ct_bboxes[..., 3] >= lr_rcentral[..., 3])

        if with_embedding:
            tl_emb = self._transpose_and_gather_feat(tl_emb, tl_inds)
            tl_emb = tl_emb.view(batch, k, 1)
            br_emb = self._transpose_and_gather_feat(br_emb, br_inds)
            br_emb = br_emb.view(batch, 1, k)
            dists = torch.abs(tl_emb - br_emb)

        t_scores = t_scores.view(batch, k, 1).repeat(1, 1, k)
        b_scores = b_scores.view(batch, 1, k).repeat(1, k, 1)
        l_scores = t_scores.view(batch, k, 1).repeat(1, 1, k)
        r_scores = b_scores.view(batch, 1, k).repeat(1, k, 1)

        tb_scores = (t_scores + b_scores) / 2  # scores for all possible boxes top bottom
        lr_scores = (l_scores + r_scores) / 2  # scores for all possible boxes left right

        scores = (tb_scores + lr_scores) / 2
        # top and bottom should have same class
        t_clses = t_clses.view(batch, k, 1).repeat(1, 1, k)
        b_clses = b_clses.view(batch, 1, k).repeat(1, k, 1)
        tb_cls_inds = (t_clses != b_clses)

        l_clses = l_clses.view(batch, k, 1).repeat(1, 1, k)
        r_clses = r_clses.view(batch, 1, k).repeat(1, k, 1)
        lr_cls_inds = (l_clses != r_clses)

        # reject boxes based on distances
        tb_dist_inds = tb_dists > distance_threshold
        lr_dist_inds = lr_dists > distance_threshold
        # reject boxes based on widths and heights
        tb_height_inds = (b_ys <= t_ys)
        tb_scores[tb_cls_inds] = -1
        tb_scores[tb_height_inds] = -1
        tb_scores[tb_dist_inds] = -1


        lr_width_inds = (r_ys <= l_ys)
        lr_scores[lr_cls_inds] = -1
        lr_scores[lr_width_inds] = -1
        lr_scores[lr_dist_inds] = -1


        if with_centripetal_shift:
            tb_scores[t_ctx_inds] = -1
            tb_scores[t_cty_inds] = -1
            tb_scores[b_ctx_inds] = -1
            tb_scores[b_cty_inds] = -1
            lr_scores[l_ctx_inds] = -1
            lr_scores[l_cty_inds] = -1
            lr_scores[r_ctx_inds] = -1
            lr_scores[r_cty_inds] = -1

        tb_scores = tb_scores.view(batch, -1)
        tb_scores, tb_inds = torch.topk(tb_scores, num_dets)
        tb_scores = tb_scores.unsqueeze(2)
        lr_scores = lr_scores.view(batch, -1)
        lr_scores, lr_inds = torch.topk(lr_scores, num_dets)
        lr_scores = lr_scores.unsqueeze(2)

        tb_bboxes = tb_bboxes.view(batch, -1, 4)
        tb_bboxes = self._gather_feat(tb_bboxes, tb_inds)
        lr_bboxes = lr_bboxes.view(batch, -1, 4)
        lr_bboxes = self._gather_feat(lr_bboxes, lr_inds)

        tb_clses = t_clses.contiguous().view(batch, -1, 1)
        tb_clses = self._gather_feat(tb_clses, tb_inds).float()
        lr_clses = l_clses.contiguous().view(batch, -1, 1)
        lr_clses = self._gather_feat(lr_clses, lr_inds).float()
        
        # ###all tb points and lr points possible
        # tb_bboxes_center_x = (tb_bboxes[..., 0] + tb_bboxes[..., 2]) / 2
        # tb_bboxes_center_y = (tb_bboxes[..., 1] + tb_bboxes[..., 3]) / 2

        # lr_bboxes_center_x = (lr_bboxes[..., 0] + lr_bboxes[..., 2]) / 2
        # lr_bboxes_center_y = (lr_bboxes[..., 1] + lr_bboxes[..., 3]) / 2


        # bboxes = torch.stack((tb_bboxes_center_x, tb_bboxes_center_y, lr_bboxes_center_x, lr_bboxes_center_y), dim=3)
        # area_bboxes = ((tb_bboxes_center_x - lr_bboxes_center_x) * (tb_bboxes_center_y - lr_bboxes_center_y)).abs()

        return tb_bboxes,lr_bboxes, tb_scores,lr_scores, tb_clses,lr_clses

        # return bboxes, scores, clses
