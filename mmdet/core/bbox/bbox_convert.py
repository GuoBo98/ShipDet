import numpy as np
import cv2
import pycocotools.mask as maskUtils


def segm2obb(mask):
    """convert segms to obb

    Args:
        segms (coco format): input segmentation

    Returns:
        list: pointobb and thetaobb
    """
    # mask = maskUtils.decode(segms).astype(np.bool)
    gray = np.array(mask * 255, dtype=np.uint8)
    
    contours = cv2.findContours(gray.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    if contours != []:
        cnt = max(contours, key = cv2.contourArea)
        rect = cv2.minAreaRect(cnt)
        x, y, w, h, theta = rect[0][0], rect[0][1], rect[1][0], rect[1][1], rect[2]
        theta = theta * np.pi / 180.0
        thetaobb = [x, y, w, h, theta]
        pointobb = thetaobb2pointobb([x, y, w, h, theta])
    else:
        thetaobb = [0, 0, 0, 0, 0]
        pointobb = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    
    return thetaobb, pointobb

def thetaobb2pointobb(thetaobb):
    """convert thetaobb to pointobb

    Args:
        thetaobb (list): [cx, cy, w, h, theta (rad/s)]

    Returns:
        list: [x1, y1, x2, y2, x3, y3, x4, y4]
    """
    box = cv2.boxPoints(((thetaobb[0], thetaobb[1]), (thetaobb[2], thetaobb[3]), thetaobb[4] * 180.0 / np.pi))
    box = np.reshape(box, [-1, ]).tolist()
    pointobb = [box[0], box[1], box[2], box[3], box[4], box[5], box[6], box[7]]

    return pointobb

def pointobb2thetaobb(pointobb):
    """convert pointobb to thetaobb

    Args:
        pointobb (list): [x1, y1, x2, y2, x3, y3, x4, y4]

    Returns:
        list: [cx, cy, w, h, theta (rad/s)]
    """
    pointobb = np.int0(np.array(pointobb))
    pointobb.resize(4, 2)
    rect = cv2.minAreaRect(pointobb)
    x, y, w, h, theta = rect[0][0], rect[0][1], rect[1][0], rect[1][1], rect[2]
    theta = theta / 180.0 * np.pi
    thetaobb = [x, y, w, h, theta]
    
    return thetaobb

def thetaobb2bbox(thetaobb):
    """convert thetaobb to bbox

    Args:
        thetaobb (list): [cx, cy, w, h, theta]

    Returns:
        list: [xmin, ymin, xmax, ymax]
    """
    pointobb = thetaobb2pointobb(thetaobb)
    bbox = pointobb2bbox(pointobb)

    return bbox

def pointobb2bbox(pointobb):
    """convert pointobb to bbox

    Args:
        pointobb (list): [x1, y1, x2, y2, x3, y3, x4, y4]

    Returns:
        list: [xmin, ymin, xmax, ymax]
    """
    xmin = min(pointobb[0::2])
    ymin = min(pointobb[1::2])
    xmax = max(pointobb[0::2])
    ymax = max(pointobb[1::2])
    bbox = [xmin, ymin, xmax, ymax]
    
    return bbox

def bbox2pointobb(bbox):
    """convert bbox to pointobb

    Args:
        bbox (list): [xmin, ymin, xmax, ymax]

    Returns:
        list: [x1, y1, x2, y2, x3, y3, x4, y4]
    """
    xmin, ymin, xmax, ymax = bbox
    x1, y1 = xmin, ymin
    x2, y2 = xmax, ymin
    x3, y3 = xmax, ymax
    x4, y4 = xmin, ymax

    pointobb = [x1, y1, x2, y2, x3, y3, x4, y4]
    
    return pointobb

def pointobb_best_point_sort(pointobb):
    """Find the "best" point and sort all points as the order that best point is first point

    Args:
        pointobb (list): unsorted points

    Returns:
        list: sorted points
    """
    xmin, ymin, xmax, ymax = pointobb2bbox(pointobb)
    reference_bbox = [xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax]
    reference_bbox = np.array(reference_bbox)
    normalize = np.array([1.0, 1.0] * 4)
    combinate = [np.roll(pointobb, 0), np.roll(pointobb, 2), np.roll(pointobb, 4), np.roll(pointobb, 6)]
    distances = np.array([np.sum(((coord - reference_bbox) / normalize)**2) for coord in combinate])
    sorted = distances.argsort()

    return combinate[sorted[0]].tolist()


# ================== rescale obb =======================

def thetaobb_rescale(thetaobbs, scale_factor, reverse_flag=False, model='simple'):
    """
    rescale thetaobb
        :param self: 
        :param thetaobb: np.array, [[x, y, w, h, theta]], (..., 5)
        :param reverse_flag: bool, if reverse_flag=Ture -> reverse rescale
    """
    if model == 'simple':
        thetaobbs_ = thetaobbs.clone()
        if reverse_flag == False:
            thetaobbs *= scale_factor
        else:
            thetaobbs /= scale_factor
        thetaobbs[..., 4::5] = thetaobbs_[..., 4::5]
    else:
        rescaled_thetaobbs = []
        for thetaobb in thetaobbs:
            pointobb = thetaobb2pointobb(thetaobb)
            pointobb = pointobb_rescale(np.array(pointobb), scale_factor, reverse_flag)
            thetaobb = pointobb2thetaobb(pointobb)

            rescaled_thetaobbs.append(thetaobb)
        thetaobbs = np.array(rescaled_thetaobbs)

    return thetaobbs

def pointobb_rescale(pointobbs, scale_factor, reverse_flag=False):
    """
    rescale pointobb
        :param self: 
        :param pointobb: np.array, [[x, y, w, h, theta]], (..., 5)
        :param reverse_flag: bool, if reverse_flag=Ture -> reverse rescale
    """
    if reverse_flag == False:
        pointobbs *= scale_factor
    else:
        pointobbs /= scale_factor
    return pointobbs

def obb2result(obbs, labels, num_classes):
    """Convert detection results to a list of numpy arrays.

    Args:
        obbs (Tensor): shape (n, 6 or 9)
        labels (Tensor): shape (n, )
        num_classes (int): class number, including background class

    Returns:
        list(ndarray): bbox results of each class
    """
    if obbs.shape[0] == 0:
        return [
            np.zeros((0, 5), dtype=np.float32) for i in range(num_classes)
        ]
    else:
        obbs = obbs.cpu().numpy()
        labels = labels.cpu().numpy()
        return [obbs[labels == i, :] for i in range(num_classes)]