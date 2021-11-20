import numpy as np
import torch

from mmdet.core.bbox.builder import BBOX_CODERS
from mmdet.core.bbox.coder.base_bbox_coder import BaseBBoxCoder


@BBOX_CODERS.register_module()
class DeltaOBBCoder(BaseBBoxCoder):
    def __init__(self,
                 target_means=(0., 0., 0., 0., 0.),
                 target_stds=(0.1, 0.1, 0.2, 0.2, 0.1),
                 obb_encode='thetaobb'):
        super(BaseBBoxCoder, self).__init__()
        self.means = target_means
        self.stds = target_stds
        self.obb_encode = obb_encode

    def encode(self, obbs, gt_obbs):
        assert obbs.size(0) == gt_obbs.size(0)
        if self.obb_encode == 'thetaobb':
            assert obbs.size(0) == gt_obbs.size(0)
            encoded_obbs = thetaobb2delta(obbs, gt_obbs, self.means, self.stds)
        else:
            raise(RuntimeError('do not support the encode mthod: {}'.format(self.obb_encode)))
        
        return encoded_obbs

    def decode(self,
               obbs,
               pred_obbs,
               max_shape=None,
               wh_ratio_clip=16 / 1000):
        assert pred_obbs.size(0) == obbs.size(0)
        if self.obb_encode == 'thetaobb':
            decoded_obbs = delta2thetaobb(obbs, pred_obbs, self.means, self.stds,
                                        max_shape, wh_ratio_clip)
        else:
            raise(RuntimeError('do not support the encode mthod: {}'.format(self.obb_encode)))

        return decoded_obbs

def thetaobb2delta(proposals, gt, means=(0., 0., 0., 0., 0.), stds=(0.1, 0.1, 0.2, 0.2, 0.1)):
    # proposals: (x1, y1, x2, y2)
    # gt: (cx, cy, w, h, theta)
    assert proposals.size(0) == gt.size(0)

    proposals = proposals.float()
    gt = gt.float()

    px = (proposals[..., 0] + proposals[..., 2]) * 0.5
    py = (proposals[..., 1] + proposals[..., 3]) * 0.5
    pw = proposals[..., 2] - proposals[..., 0]
    ph = proposals[..., 3] - proposals[..., 1]
    pa = np.ones(proposals.shape[0], dtype=np.int32) * (-np.pi / 2.0)
    pa = torch.from_numpy(np.stack(pa)).float().to(proposals.device)

    gx = gt[..., 0]
    gy = gt[..., 1]
    gw = gt[..., 2]
    gh = gt[..., 3]
    ga = gt[..., 4]

    dx = (gx - px) / pw
    dy = (gy - py) / ph
    dw = torch.log(gw / pw)
    dh = torch.log(gh / ph)
    da = ga - pa

    deltas = torch.stack([dx, dy, dw, dh, da], dim=-1)

    means = deltas.new_tensor(means).unsqueeze(0)
    stds = deltas.new_tensor(stds).unsqueeze(0)
    deltas = deltas.sub_(means).div_(stds)

    return deltas

def delta2thetaobb(rois,
                   deltas,
                   means=[0., 0., 0., 0., 0.],
                   stds=[0.1, 0.1, 0.2, 0.2, 0.1],
                   max_shape=None,
                   wh_ratio_clip=16 / 1000):
    means = deltas.new_tensor(means).repeat(1, deltas.size(1) // 5)
    stds = deltas.new_tensor(stds).repeat(1, deltas.size(1) // 5)
    denorm_deltas = deltas * stds + means

    dx = denorm_deltas[:, 0::5]
    dy = denorm_deltas[:, 1::5]
    dw = denorm_deltas[:, 2::5]
    dh = denorm_deltas[:, 3::5]
    da = denorm_deltas[:, 4::5]

    max_ratio = np.abs(np.log(wh_ratio_clip))

    dw = dw.clamp(min=-max_ratio, max=max_ratio)
    dh = dh.clamp(min=-max_ratio, max=max_ratio)

    px = ((rois[:, 0] + rois[:, 2]) * 0.5).unsqueeze(1).expand_as(dx)
    py = ((rois[:, 1] + rois[:, 3]) * 0.5).unsqueeze(1).expand_as(dy)
    pw = (rois[:, 2] - rois[:, 0]).unsqueeze(1).expand_as(dw)
    ph = (rois[:, 3] - rois[:, 1]).unsqueeze(1).expand_as(dh)
    pa = np.ones(rois.shape[0], dtype=np.int32) * (-np.pi / 2.0)
    pa = torch.from_numpy(np.stack(pa)).float().to(rois.device).unsqueeze(1).expand_as(da)
    
    gw = pw * dw.exp()
    gh = ph * dh.exp()
    gx = torch.addcmul(px, 1, pw, dx)  # gx = px + pw * dx
    gy = torch.addcmul(py, 1, ph, dy)  # gy = py + ph * dy
    ga = da + pa

    if max_shape is not None:
        gx = gx.clamp(min=0, max=max_shape[1])
        gy = gy.clamp(min=0, max=max_shape[0])
        gw = gw.clamp(min=0, max=max_shape[1])
        gh = gh.clamp(min=0, max=max_shape[0])
    thetaobbs = torch.stack([gx, gy, gw, gh, ga], dim=-1).view_as(deltas)
    return thetaobbs