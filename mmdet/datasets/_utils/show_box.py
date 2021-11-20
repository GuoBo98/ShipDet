import cv2
from mmcv.image import imread, imwrite
import numpy as np

Color_dict = {
    '0': (0, 0, 255),  # red
    '1': (0, 255, 0),  # green
    '2': (255, 0, 0),  # blue
    '3': (255, 255, 0),  # cyan
    '4': (0, 255, 255),  # yellow
    '5': (255, 0, 255),  # magenta
    '6': (255, 255, 255),  # white
    '7': (0, 0, 0)  # black

}

def imshow(img, win_name='', wait_time=0):
    """Show an image.

    Args:
        img (str or ndarray): The image to be displayed.
        win_name (str): The window name.
        wait_time (int): Value of waitKey param.
    """
    cv2.imshow(win_name, imread(img))
    if wait_time == 0:  # prevent from hangning if windows was closed
        while True:
            ret = cv2.waitKey(1)

            closed = cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1
            # if user closed window or if some key pressed
            if closed or ret != -1:
                break
    else:
        ret = cv2.waitKey(wait_time)

def ROI_trans_imshow_det_bboxes(img,
                                bboxes,
                                labels,
                                class_names=None,
                                score_thr=0,
                                thickness=1,
                                font_scale=0.5,
                                show=True,
                                win_name='',
                                wait_time=0,
                                out_file=None):
    """Draw bboxes and class labels (with scores) on an image.

    Args:
        img (str or ndarray): The image to be displayed.
        bboxes (ndarray): Bounding boxes (with scores), shaped (n, 8) or
            (n, 9).
        labels (ndarray): Labels of bboxes.
        class_names (list[str]): Names of each classes.
        score_thr (float): Minimum score of bboxes to be shown.
        bbox_color (str or tuple or :obj:`Color`): Color of bbox lines.
        text_color (str or tuple or :obj:`Color`): Color of texts.
        thickness (int): Thickness of lines.
        font_scale (float): Font scales of texts.
        show (bool): Whether to show the image.
        win_name (str): The window name.
        wait_time (int): Value of waitKey param.
        out_file (str or None): The filename to write the image.

    Returns:
        ndarray: The image with bboxes drawn on it.
    """
    assert bboxes.ndim == 2
    assert labels.ndim == 1
    assert bboxes.shape[0] == labels.shape[0]
    assert bboxes.shape[1] == 8 or bboxes.shape[1] == 9
    img = imread(img)

    if score_thr > 0:
        assert bboxes.shape[1] == 9
        scores = bboxes[:, -1]
        inds = scores > score_thr
        bboxes = bboxes[inds, :]
        labels = labels[inds]


    img = np.ascontiguousarray(img)
    for bbox, label in zip(bboxes, labels):
        bbox_polygon = np.array([[bbox[0], bbox[1]], [bbox[2], bbox[3]],
                                 [bbox[4], bbox[5]], [bbox[6], bbox[7]]], np.int32)
        bbox_color = Color_dict[str(label)]
        cv2.polylines(img, [bbox_polygon], isClosed=True, color=bbox_color, thickness=thickness)

    if show:
        imshow(img, win_name, wait_time)
    if out_file is not None:
        imwrite(img, out_file)
    return img
