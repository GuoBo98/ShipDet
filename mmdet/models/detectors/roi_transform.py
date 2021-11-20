import mmcv
import numpy as np
import os
import json

from mmdet.models.builder import DETECTORS
from mmdet.models.detectors.two_stage import TwoStageDetector
from mmdet.datasets._utils import ROI_trans_imshow_det_bboxes
# from mmdet.datasets import clip_boundary_polygon, polygon2mask, find_max_area_5point
from shapely.geometry import Polygon

@DETECTORS.register_module()
class ROI_Transfomer(TwoStageDetector):
    r"""Implementation of `Cascade R-CNN: Delving into High Quality Object
    Detection <https://arxiv.org/abs/1906.09756>`_"""

    def __init__(self,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(ROI_Transfomer, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)



    # def show_result(self,
    #                 img,
    #                 result,
    #                 score_thr=0.3,
    #                 thickness=1,
    #                 font_scale=0.5,
    #                 win_name='',
    #                 show=False,
    #                 wait_time=0,
    #                 out_file=None):
    #     """Draw `result` over `img`.

    #     Args:
    #         img (str or Tensor): The image to be displayed.
    #         result (Tensor or tuple): The results to draw over `img`
    #             bbox_result or (bbox_result, segm_result).
    #         score_thr (float, optional): Minimum score of bboxes to be shown.
    #             Default: 0.3.
    #         bbox_color (str or tuple or :obj:`Color`): Color of bbox lines.
    #         text_color (str or tuple or :obj:`Color`): Color of texts.
    #         thickness (int): Thickness of lines.
    #         font_scale (float): Font scales of texts.
    #         win_name (str): The window name.
    #         wait_time (int): Value of waitKey param.
    #             Default: 0.
    #         show (bool): Whether to show the image.
    #             Default: False.
    #         out_file (str or None): The filename to write the image.
    #             Default: None.

    #     Returns:
    #         img (Tensor): Only if not `show` or `out_file`
    #     """
    #     img = mmcv.imread(img)
    #     img = img.copy()
    #     if isinstance(result, tuple):
    #         bbox_result, segm_result = result
    #         if isinstance(segm_result, tuple):
    #             segm_result = segm_result[0]  # ms rcnn
    #     else:
    #         bbox_result, segm_result = result, None
    #     bboxes = np.vstack(bbox_result)
    #     labels = [
    #         np.full(bbox.shape[0], i, dtype=np.int32)
    #         for i, bbox in enumerate(bbox_result)
    #     ]
    #     labels = np.concatenate(labels)
    #     # draw segmentation masks
    #     if segm_result is not None and len(labels) > 0:  # non empty
    #         segms = mmcv.concat_list(segm_result)
    #         inds = np.where(bboxes[:, -1] > score_thr)[0]
    #         np.random.seed(42)
    #         color_masks = [
    #             np.random.randint(0, 256, (1, 3), dtype=np.uint8)
    #             for _ in range(max(labels) + 1)
    #         ]
    #         for i in inds:
    #             i = int(i)
    #             color_mask = color_masks[labels[i]]
    #             mask = segms[i]
    #             img[mask] = img[mask] * 0.5 + color_mask * 0.5
    #     # if out_file specified, do not show image in window
    #     if out_file is not None:
    #         show = False
    #     # draw bounding boxes
    #     ROI_trans_imshow_det_bboxes(
    #         img,
    #         bboxes,
    #         labels,
    #         class_names=self.CLASSES,
    #         score_thr=score_thr,
    #         thickness=thickness,
    #         font_scale=font_scale,
    #         win_name=win_name,
    #         show=show,
    #         wait_time=wait_time,
    #         out_file=out_file)

    #     if not (show or out_file):
    #         return img

    # def save_result(self, fp, img_name, img_shape, result, is_latest_img=False):
    #     assert len(result) == 5
    #     for label in range(len(result)):
    #         rbboxes = result[label]
    #         cls_name = self.CLASSES[label]
    #         for i in range(rbboxes.shape[0]):
    #             poly = rbboxes[i][:-1]
    #             score = (rbboxes[i][-1]).astype(np.float32)
    #             cls_name = int(cls_name)

    #             x_min = min(poly[::2])
    #             x_max = max(poly[::2])
    #             y_min = min(poly[1::2])
    #             y_max = max(poly[1::2])
    #             if x_min < 0 or y_min < 0 or x_max > img_shape[0] or y_max > img_shape[1]:
    #                 print(img_name)
    #                 print('clip before', poly)
    #                 poly_to_clip = [Polygon([(poly[0], poly[1]), (poly[2], poly[3]), (poly[4], poly[5]), (poly[6], poly[7])])]
    #                 poly_mask = clip_boundary_polygon(poly_to_clip, img_shape)
    #                 cliped_poly = polygon2mask(poly_mask[0])

    #                 if len(cliped_poly) > 8:
    #                     cliped_poly = find_max_area_5point(cliped_poly)
    #                 poly = cliped_poly
    #                 print('clip after', poly)
    #             assert len(poly) == 8
    #             poly = list(map(round, poly))
    #             outline = img_name + ' ' + str(cls_name) + ' ' + str(score) + ' ' + ' '.join(list(map(str, poly)))
    #             if (not is_latest_img) or i != (rbboxes.shape[0] - 1) or label != (len(result) - 1):
    #                 fp.write(outline + '\n')
    #             else:
    #                 print('last_outline', outline)
    #                 fp.write(outline)


    # def save_result_hjjjson(self, result, img_name, img_height, img_width, jsonfile_prefix=None, **kwargs):
    #     """
    #     Convert cnn output results to special format results

    #     :param results: cnn output results
    #     :param jsonfile_prefix: the path of special format results
    #     :param kwargs:
    #     """

    #     if not os.path.exists(os.path.join(jsonfile_prefix)):
    #         os.makedirs(os.path.join(jsonfile_prefix))

    #     filename = img_name
    #     basename = os.path.basename(os.path.splitext(filename)[0])
    #     data_dict = {}
    #     data_dict['width'] = img_width
    #     data_dict['height'] = img_height
    #     data_dict['rotate'] = 0
    #     data_dict['valid'] = True
    #     data_dict['step_1'] = {}
    #     data_dict['step_1']['dataSourceStep'] = 0
    #     data_dict['step_1']['toolName'] = "polygonTool"
    #     data_dict['step_1']['result'] = []
    #     id_count = 1
    #     for label in range(len(result)):
    #         cls_name = self.CLASSES[label]
    #         rbboxes = result[label]
    #         for i in range(rbboxes.shape[0]):
    #             single_res = {}
    #             single_res['sourceID'] = "0"
    #             single_res['id'] = str(id_count).zfill(8)
    #             poly = rbboxes[i][:-1]

    #             poly = list(map(int, poly))
    #             x_min = min(poly[::2])
    #             x_max = max(poly[::2])
    #             y_min = min(poly[1::2])
    #             y_max = max(poly[1::2])
    #             if x_min < 0 or y_min < 0 or x_max > img_width or y_max > img_height:
    #                 print(img_name)
    #                 print('clip before', poly)
    #                 poly_to_clip = [Polygon([(poly[0], poly[1]), (poly[2], poly[3]), (poly[4], poly[5]), (poly[6], poly[7])])]
    #                 poly_mask = clip_boundary_polygon(poly_to_clip, (img_width, img_height))
    #                 cliped_poly = polygon2mask(poly_mask[0])

    #                 if len(cliped_poly) > 8:
    #                     cliped_poly = find_max_area_5point(cliped_poly)
    #                 poly = cliped_poly
    #                 print('clip after', poly)
    #             assert len(poly) == 8

    #             pointList = []
    #             num_points = len(poly) / 2
    #             assert num_points == 4
    #             for j in range(int(num_points)):
    #                 pointList.append(dict(x=poly[2 * j], y=poly[2 * j + 1]))
    #             single_res['pointList'] = pointList
    #             single_res['valid'] = True
    #             single_res['attribute'] = cls_name
    #             single_res['order'] = id_count
    #             data_dict['step_1']['result'].append(single_res)
    #             id_count += 1

    #     with open(os.path.join(jsonfile_prefix, basename + '.png.json'), 'w') as obb_f_out:
    #         json.dump(data_dict, obb_f_out)
