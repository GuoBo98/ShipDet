import numpy as np
import torch

from mmdet.core.bbox.builder import BBOX_CODERS
from mmdet.core.bbox.coder.base_bbox_coder import BaseBBoxCoder
import math


@BBOX_CODERS.register_module()
class DeltaXYWHThetaBBoxCoder(BaseBBoxCoder):
    """Delta XYWHTheta BBox coder.

    Following the practice in `R-CNN <https://arxiv.org/abs/1311.2524>`_,
    this coder encodes bbox (x1, y1, x2, y2, x3, y3, x4, y4) into delta (dx, dy, dw, dh, theta) and
    decodes delta (dx, dy, dw, dh, dtheta) back to original bbox (x1, y1, x2, y2, x3, y3, x4, y4).

    Args:
        target_means (Sequence[float]): Denormalizing means of target for
            delta coordinates
        target_stds (Sequence[float]): Denormalizing standard deviation of
            target for delta coordinates
    """

    def __init__(self,
                 target_means=(0., 0., 0., 0., 0.),
                 target_stds=(1., 1., 1., 1., 1.)):
        super(BaseBBoxCoder, self).__init__()
        self.means = target_means
        self.stds = target_stds

    def encode(self, bboxes, gt_bboxes, encoder_name):
        """Get box regression transformation deltas that can be used to
        transform the ``bboxes`` into the ``gt_bboxes``.

        Args:
            bboxes (torch.Tensor): Source boxes, e.g., object proposals.
            gt_bboxes (torch.Tensor): Target of the transformation, e.g.,
                ground-truth boxes.

        Returns:
            torch.Tensor: Box transformation deltas
        """

        assert bboxes.size(0) == gt_bboxes.size(0)
        assert bboxes.size(-1) == gt_bboxes.size(-1) == 5

        encode_func = getattr(self, encoder_name, None)
        assert encode_func is not None

        encoded_bboxes = encode_func(bboxes, gt_bboxes, self.means, self.stds)
        return encoded_bboxes

    def decode(self,
               bboxes,
               pred_bboxes,
               decoder_name,
               max_shape=None,
               wh_ratio_clip=16 / 1000):
        """Apply transformation `pred_bboxes` to `boxes`.

        Args:
            boxes (torch.Tensor): Basic boxes.
            pred_bboxes (torch.Tensor): Encoded boxes with shape
            max_shape (tuple[int], optional): Maximum shape of boxes.
                Defaults to None.
            wh_ratio_clip (float, optional): The allowed ratio between
                width and height.

        Returns:
            torch.Tensor: Decoded boxes.
        """

        assert pred_bboxes.size(0) == bboxes.size(0)
        decode_func = getattr(self, decoder_name, None)
        assert decode_func is not None

        decoded_bboxes = decode_func(bboxes, pred_bboxes, self.means, self.stds,
                                     max_shape, wh_ratio_clip)

        return decoded_bboxes

    def dbbox2delta(self, proposals, gt, means=[0, 0, 0, 0, 0], stds=[1, 1, 1, 1, 1]):
        """
        :param proposals: (x_ctr, y_ctr, w, h, angle)
                shape (n, 5)
        :param gt: (x_ctr, y_ctr, w, h, angle)
        :param means:
        :param stds:
        :return: encoded targets: shape (n, 5)
        """
        proposals = proposals.float()
        gt = gt.float()
        gt_widths = gt[..., 2]
        gt_heights = gt[..., 3]
        gt_angle = gt[..., 4]

        proposals_widths = proposals[..., 2]
        proposals_heights = proposals[..., 3]
        proposals_angle = proposals[..., 4]

        coord = gt[..., 0:2] - proposals[..., 0:2]
        dx = (torch.cos(proposals[..., 4]) * coord[..., 0] +
              torch.sin(proposals[..., 4]) * coord[..., 1]) / proposals_widths
        dy = (-torch.sin(proposals[..., 4]) * coord[..., 0] +
              torch.cos(proposals[..., 4]) * coord[..., 1]) / proposals_heights
        dw = torch.log(gt_widths / proposals_widths)
        dh = torch.log(gt_heights / proposals_heights)
        dangle = (gt_angle - proposals_angle) % (2 * math.pi) / (2 * math.pi)
        deltas = torch.stack((dx, dy, dw, dh, dangle), -1)

        means = deltas.new_tensor(means).unsqueeze(0)
        stds = deltas.new_tensor(stds).unsqueeze(0)
        deltas = deltas.sub_(means).div_(stds)

        # TODO: expand bbox regression
        return deltas

    def delta2dbbox(self, Rrois,
                       deltas,
                       means=[0, 0, 0, 0, 0],
                       stds=[1, 1, 1, 1, 1],
                       max_shape=None,
                       wh_ratio_clip=16 / 1000):
        """

        :param Rrois: (cx, cy, w, h, theta)
        :param deltas: (dx, dy, dw, dh, dtheta)
        :param means:
        :param stds:
        :param max_shape:
        :param wh_ratio_clip:
        :return:
        """
        means = deltas.new_tensor(means).repeat(1, deltas.size(1) // 5)
        stds = deltas.new_tensor(stds).repeat(1, deltas.size(1) // 5)
        denorm_deltas = deltas * stds + means

        dx = denorm_deltas[:, 0::5]
        dy = denorm_deltas[:, 1::5]
        dw = denorm_deltas[:, 2::5]
        dh = denorm_deltas[:, 3::5]
        dangle = denorm_deltas[:, 4::5]

        max_ratio = np.abs(np.log(wh_ratio_clip))
        dw = dw.clamp(min=-max_ratio, max=max_ratio)
        dh = dh.clamp(min=-max_ratio, max=max_ratio)
        Rroi_x = (Rrois[:, 0]).unsqueeze(1).expand_as(dx)
        Rroi_y = (Rrois[:, 1]).unsqueeze(1).expand_as(dy)
        Rroi_w = (Rrois[:, 2]).unsqueeze(1).expand_as(dw)
        Rroi_h = (Rrois[:, 3]).unsqueeze(1).expand_as(dh)
        Rroi_angle = (Rrois[:, 4]).unsqueeze(1).expand_as(dangle)
        # import pdb
        # pdb.set_trace()
        gx = dx * Rroi_w * torch.cos(Rroi_angle) \
             - dy * Rroi_h * torch.sin(Rroi_angle) + Rroi_x
        gy = dx * Rroi_w * torch.sin(Rroi_angle) \
             + dy * Rroi_h * torch.cos(Rroi_angle) + Rroi_y
        gw = Rroi_w * dw.exp()
        gh = Rroi_h * dh.exp()

        # TODO: check the hard code
        gangle = (2 * np.pi) * dangle + Rroi_angle
        gangle = gangle % (2 * np.pi)

        if max_shape is not None:
            pass

        bboxes = torch.stack([gx, gy, gw, gh, gangle], dim=-1).view_as(deltas)
        return bboxes

    def dbbox2delta_v3(self, proposals, gt, means=[0, 0, 0, 0, 0], stds=[1, 1, 1, 1, 1]):
        """
        This version removes the module operation
        :param proposals: (x_ctr, y_ctr, w, h, angle)
                shape (n, 5)
        :param gt: (x_ctr, y_ctr, w, h, angle)
        :param means:
        :param stds:
        :return: encoded targets: shape (n, 5)
        """
        proposals = proposals.float()
        gt = gt.float()
        gt_widths = gt[..., 2]
        gt_heights = gt[..., 3]
        gt_angle = gt[..., 4]

        proposals_widths = proposals[..., 2]
        proposals_heights = proposals[..., 3]
        proposals_angle = proposals[..., 4]

        coord = gt[..., 0:2] - proposals[..., 0:2]
        dx = (torch.cos(proposals[..., 4]) * coord[..., 0] +
              torch.sin(proposals[..., 4]) * coord[..., 1]) / proposals_widths
        dy = (-torch.sin(proposals[..., 4]) * coord[..., 0] +
              torch.cos(proposals[..., 4]) * coord[..., 1]) / proposals_heights
        dw = torch.log(gt_widths / proposals_widths)
        dh = torch.log(gt_heights / proposals_heights)
        # import pdb
        # print('in dbbox2delta v3')
        # pdb.set_trace()
        # dangle = (gt_angle - proposals_angle) % (2 * math.pi) / (2 * math.pi)
        # TODO: debug for it, proposals_angle are -1.5708, gt_angle should close to -1.57, actully they close to 5.0153
        dangle = gt_angle - proposals_angle
        deltas = torch.stack((dx, dy, dw, dh, dangle), -1)

        means = deltas.new_tensor(means).unsqueeze(0)
        stds = deltas.new_tensor(stds).unsqueeze(0)
        deltas = deltas.sub_(means).div_(stds)

        return deltas

    def delta2dbbox_v3(self, Rrois,
                       deltas,
                       means=[0, 0, 0, 0, 0],
                       stds=[1, 1, 1, 1, 1],
                       max_shape=None,
                       wh_ratio_clip=16 / 1000):
        """
        This version removes the module operation
        :param Rrois: (cx, cy, w, h, theta)
        :param deltas: (dx, dy, dw, dh, dtheta)
        :param means:
        :param stds:
        :param max_shape:
        :param wh_ratio_clip:
        :return:
        """
        means = deltas.new_tensor(means).repeat(1, deltas.size(1) // 5)
        stds = deltas.new_tensor(stds).repeat(1, deltas.size(1) // 5)
        denorm_deltas = deltas * stds + means

        dx = denorm_deltas[:, 0::5]
        dy = denorm_deltas[:, 1::5]
        dw = denorm_deltas[:, 2::5]
        dh = denorm_deltas[:, 3::5]
        dangle = denorm_deltas[:, 4::5]

        max_ratio = np.abs(np.log(wh_ratio_clip))
        dw = dw.clamp(min=-max_ratio, max=max_ratio)
        dh = dh.clamp(min=-max_ratio, max=max_ratio)
        Rroi_x = (Rrois[:, 0]).unsqueeze(1).expand_as(dx)
        Rroi_y = (Rrois[:, 1]).unsqueeze(1).expand_as(dy)
        Rroi_w = (Rrois[:, 2]).unsqueeze(1).expand_as(dw)
        Rroi_h = (Rrois[:, 3]).unsqueeze(1).expand_as(dh)
        Rroi_angle = (Rrois[:, 4]).unsqueeze(1).expand_as(dangle)
        # import pdb
        # pdb.set_trace()
        gx = dx * Rroi_w * torch.cos(Rroi_angle) \
             - dy * Rroi_h * torch.sin(Rroi_angle) + Rroi_x
        gy = dx * Rroi_w * torch.sin(Rroi_angle) \
             + dy * Rroi_h * torch.cos(Rroi_angle) + Rroi_y
        gw = Rroi_w * dw.exp()
        gh = Rroi_h * dh.exp()

        # TODO: check the hard code
        # gangle = (2 * np.pi) * dangle + Rroi_angle
        gangle = dangle + Rroi_angle
        # gangle = gangle % ( 2 * np.pi)

        if max_shape is not None:
            pass

        bboxes = torch.stack([gx, gy, gw, gh, gangle], dim=-1).view_as(deltas)
        return bboxes

    def dbbox2delta_v2(self, proposals, gt, means=[0, 0, 0, 0, 0], stds=[1, 1, 1, 1, 1]):
        """
        :param proposals: (x_ctr, y_ctr, w, h, angle)
                shape (n, 5)
        :param gt: (x_ctr, y_ctr, w, h, angle)
        :param means:
        :param stds:
        :return: encoded targets: shape (n, 5)
        """
        gt_widths = gt[..., 2]
        gt_heights = gt[..., 3]
        gt_angle = gt[..., 4]

        roi_widths = proposals[..., 2]
        roi_heights = proposals[..., 3]
        roi_angle = proposals[..., 4]

        coord = gt[..., 0:2] - proposals[..., 0:2]
        targets_dx = (torch.cos(roi_angle) * coord[..., 0] + torch.sin(roi_angle) * coord[:, 1]) / roi_widths
        targets_dy = (-torch.sin(roi_angle) * coord[..., 0] + torch.cos(roi_angle) * coord[:, 1]) / roi_heights
        targets_dw = torch.log(gt_widths / roi_widths)
        targets_dh = torch.log(gt_heights / roi_heights)
        targets_dangle = (gt_angle - roi_angle)
        dist = targets_dangle % (2 * np.pi)#refer to  choose_best_match_batch# line 26, line 27
        dist = torch.min(dist, np.pi * 2 - dist)
        try:
            assert np.all(dist.cpu().numpy() <= (np.pi / 2. + 0.001))
        except:
            import pdb
            pdb.set_trace()

        inds = torch.sin(targets_dangle) < 0
        dist[inds] = -dist[inds]#change pos, neg
        # TODO: change the norm value
        dist = dist / (np.pi / 2.)
        deltas = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh, dist), -1)

        means = deltas.new_tensor(means).unsqueeze(0)
        stds = deltas.new_tensor(stds).unsqueeze(0)
        deltas = deltas.sub_(means).div_(stds)

        return deltas

    def delta2dbbox_v2(self, Rrois,
                       deltas,
                       means=[0, 0, 0, 0, 0],
                       stds=[1, 1, 1, 1, 1],
                       max_shape=None,
                       wh_ratio_clip=16 / 1000):
        means = deltas.new_tensor(means).repeat(1, deltas.size(1) // 5)
        stds = deltas.new_tensor(stds).repeat(1, deltas.size(1) // 5)
        denorm_deltas = deltas * stds + means

        dx = denorm_deltas[:, 0::5]
        dy = denorm_deltas[:, 1::5]
        dw = denorm_deltas[:, 2::5]
        dh = denorm_deltas[:, 3::5]
        dangle = denorm_deltas[:, 4::5]

        max_ratio = np.abs(np.log(wh_ratio_clip))
        dw = dw.clamp(min=-max_ratio, max=max_ratio)
        dh = dh.clamp(min=-max_ratio, max=max_ratio)
        Rroi_x = (Rrois[:, 0]).unsqueeze(1).expand_as(dx)
        Rroi_y = (Rrois[:, 1]).unsqueeze(1).expand_as(dy)
        Rroi_w = (Rrois[:, 2]).unsqueeze(1).expand_as(dw)
        Rroi_h = (Rrois[:, 3]).unsqueeze(1).expand_as(dh)
        Rroi_angle = (Rrois[:, 4]).unsqueeze(1).expand_as(dangle)

        gx = dx * Rroi_w * torch.cos(Rroi_angle) \
             - dy * Rroi_h * torch.sin(Rroi_angle) + Rroi_x
        gy = dx * Rroi_w * torch.sin(Rroi_angle) \
             + dy * Rroi_h * torch.cos(Rroi_angle) + Rroi_y
        gw = Rroi_w * dw.exp()
        gh = Rroi_h * dh.exp()

        gangle = (np.pi / 2.) * dangle + Rroi_angle

        if max_shape is not None:
            # TODO: finish it
            pass

        bboxes = torch.stack([gx, gy, gw, gh, gangle], dim=-1).view_as(deltas)
        return bboxes
