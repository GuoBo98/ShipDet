from ..bbox import dbbox_mapping_back
import torch


def merge_aug_rbboxes(aug_dbboxes, aug_scores, img_metas, rcnn_test_cfg):
    """Merge augmented detection dbboxes and scores.

    Args:
        aug_bboxes (list[Tensor]): shape (n, 5*#class)
        aug_scores (list[Tensor] or None): shape (n, #class)
        img_shapes (list[Tensor]): shape (3, ).
        rcnn_test_cfg (dict): rcnn test config.

    Returns:
        tuple: (bboxes, scores)
    """
    recovered_bboxes = []
    for bboxes, img_info in zip(aug_dbboxes, img_metas):
        img_shape = img_info[0]['img_shape']
        scale_factor = img_info[0]['scale_factor']
        flip = img_info[0]['flip']
        flip_direction = img_info[0]['flip_direction']
        # if flip_direction:
        #     import pdb
        #     pdb.set_trace()
        #     print('flip', flip)
        #     print('flip_direction', flip_direction)
        bboxes = dbbox_mapping_back(bboxes, img_shape, scale_factor, flip)
        recovered_bboxes.append(bboxes)
    # import pdb; pdb.set_trace()
    # bboxes = torch.stack(recovered_bboxes).mean(dim=0)
    bboxes = torch.cat(recovered_bboxes, 0)
    if aug_scores is None:
        return bboxes
    else:
        # scores = torch.stack(aug_scores).mean(dim=0)
        scores = torch.cat(aug_scores, 0)
        return bboxes, scores