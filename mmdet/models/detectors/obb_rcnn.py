import mmcv
import numpy as np
import cv2

from mmdet.models.builder import DETECTORS
from mmdet.models.detectors.two_stage import TwoStageDetector


@DETECTORS.register_module()
class OBBRCNN(TwoStageDetector):
    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None):
        super(OBBRCNN, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)

    def _show_thetaobb(self, img, thetaobb, color=(0, 0, 255)):
        """show single theteobb

        Args:
            im (np.array): input image
            thetaobb (list): [cx, cy, w, h, theta]
            color (tuple, optional): draw color. Defaults to (0, 0, 255).

        Returns:
            np.array: image with thetaobb
        """
        cx, cy, w, h, theta = thetaobb

        rect = ((cx, cy), (w, h), theta / np.pi * 180.0)
        rect = cv2.boxPoints(rect)
        rect = np.int0(rect)
        cv2.drawContours(img, [rect], -1, color, 3)

        return img

    def _show_bbox(self, img, bbox, color=(0, 0, 255)):
        """show rectangle (bbox)

        Args:
            img (np.array): input image
            bbox (list): [xmin, ymin, xmax, ymax]
            color (tuple, optional): draw color. Defaults to (0, 0, 255).

        Returns:
            np.array: output image
        """
        cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 3)

        return img


    def show_result(self,
                img,
                result,
                score_thr=0.5,
                bbox_color='green',
                text_color='green',
                thickness=1,
                font_scale=0.5,
                win_name='',
                show=False,
                wait_time=0,
                out_file=None,
                show_bbox=True,
                show_obb=True):
        """Draw `result` over `img`.

        Args:
            img (str or Tensor): The image to be displayed.
            result (Tensor or tuple): The results to draw over `img`
                bbox_result or (bbox_result, segm_result).
            score_thr (float, optional): Minimum score of bboxes to be shown.
                Default: 0.3.
            bbox_color (str or tuple or :obj:`Color`): Color of bbox lines.
            text_color (str or tuple or :obj:`Color`): Color of texts.
            thickness (int): Thickness of lines.
            font_scale (float): Font scales of texts.
            win_name (str): The window name.
            wait_time (int): Value of waitKey param.
                Default: 0.
            show (bool): Whether to show the image.
                Default: False.
            out_file (str or None): The filename to write the image.
                Default: None.

        Returns:
            img (Tensor): Only if not `show` or `out_file`
        """
        img = mmcv.imread(img)
        img = img.copy()
        if isinstance(result, tuple):
            bbox_result, obb_result = result
            if isinstance(obb_result, tuple):
                obb_result = obb_result[0]  # ms rcnn
        else:
            bbox_result, obb_result = result, None
        bboxes = np.vstack(bbox_result)
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
        labels = np.concatenate(labels)
        # draw segmentation masks
        if obb_result is not None and len(labels) > 0:  # non empty
            thetaobbs_np = mmcv.concat_list(obb_result)
            # inds = np.where(bboxes[:, -1] > score_thr)[0]
            inds = np.where(bboxes[:, -1] > 0.5)[0]
            thetaobbs_show = []
            bboxes_show = []
            for i in inds:
                i = int(i)
                thetaobb = thetaobbs_np[i]
                thetaobbs_show.append(thetaobb)
                bboxes_show.append(bboxes[i, :-1].tolist())
        # if out_file specified, do not show image in window
        if out_file is not None:
            show = False
        # draw bounding boxes
        if show_bbox:
            for bbox in bboxes_show:
                img = self._show_bbox(img, bbox, color=(255, 0, 0))

        if show_obb:
            for thetaobb in thetaobbs_show:
                img = self._show_thetaobb(img, thetaobb)

        mmcv.imshow(img, win_name=win_name, wait_time=wait_time)

        if out_file is not None:
            cv2.imwrite(out_file, img)

        if not (show or out_file):
            return img