from mmdet.core import pointobb_best_point_sort
from mmdet.core import BitmapMasks
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import Resize, RandomFlip

import copy
import mmcv
import numpy as np
from numpy import random
from ..builder import PIPELINES
import pycocotools.mask as maskUtils
from mmdet.core import BitmapMasks, PolygonMasks

def _poly2mask(mask_ann, img_h, img_w):
    """Private function to convert masks represented with polygon to
    bitmaps.

    Args:
        mask_ann (list | dict): Polygon mask annotation input.
        img_h (int): The height of output mask.
        img_w (int): The width of output mask.

    Returns:
        numpy.ndarray: The decode bitmap mask of shape (img_h, img_w).
    """

    if isinstance(mask_ann, list):
        # polygon -- a single object might consist of multiple parts
        # we merge all parts into one mask rle code
        rles = maskUtils.frPyObjects(mask_ann, img_h, img_w)
        rle = maskUtils.merge(rles)
    elif isinstance(mask_ann['counts'], list):
        # uncompressed RLE
        rle = maskUtils.frPyObjects(mask_ann, img_h, img_w)
    else:
        # rle
        rle = mask_ann
    mask = maskUtils.decode(rle)
    return mask

@PIPELINES.register_module()
class MixUp(object):
    def __init__(self, p=0.3, lambd=0.5,with_mask=True):
        self.lambd = lambd
        self.p = p
        self.img2 = None
        self.boxes2 = None
        self.labels2= None
        self.with_mask = with_mask
        if self.with_mask:
            self.masks2 = None


    def __call__(self, results):
        if self.with_mask:
            masks1 = results['mask_bak']
            #LoadAnnotations._load_mask()中备份mask

        img1, boxes1, labels1= [
            results[k] for k in ('img', 'gt_bboxes', 'gt_labels')
        ]
        
        if random.random() < self.p and self.img2 is not None and img1.shape[1]==self.img2.shape[1]:
            #print("********** start mixup **********")  
            #print('label:', labels1,self.labels2)
            #print('boxes:', boxes1,self.boxes2)
            #self.lambd = np.random.beta(2, 2)
            # self.lambd = np.random.beta(1.5,1.5)

            height = max(img1.shape[0], self.img2.shape[0])
            width = max(img1.shape[1], self.img2.shape[1])
            mixup_image = np.zeros([height, width, 3], dtype='float32')
            mixup_image[:img1.shape[0], :img1.shape[1], :] = img1.astype('float32') * self.lambd
            mixup_image[:self.img2.shape[0], :self.img2.shape[1], :] += self.img2.astype('float32') * (1. - self.lambd)
            mixup_image = mixup_image.astype('uint8')
            #mixup_image = np.zeros([height, width, 3])

            #mixup_image[:img1.shape[0], :img1.shape[1], :] = img1 * self.lambd
            #mixup_image[:self.img2.shape[0], :self.img2.shape[1], :] += self.img2 * (1. - self.lambd) # �ϲ�
            #y1 = np.vstack((boxes1, np.full((boxes1.shape[0], 1), self.lambd)))
            #y2 = np.hstack((self.boxes2, np.full((self.boxes2.shape[0], 1), 1. - self.lambd)))
            #mixup_boxes = np.vstack((y1, y2))
            mixup_boxes = np.vstack((boxes1, self.boxes2))
            mixup_label = np.hstack((labels1,self.labels2))

            #print(self.lambd,mixup_boxes,mixup_label)
            results['img'] = mixup_image
            results['gt_bboxes'] = mixup_boxes
            results['gt_labels'] = mixup_label

            if self.with_mask:
                mixup_mask = masks1
                mixup_mask.extend(self.masks2)
                h, w = height,width
                mix_masks = BitmapMasks(
                    [_poly2mask(mask, h, w) for mask in mixup_mask], h, w)            
                results['gt_masks'] = mix_masks
            #cv2.imwrite('./mixup/mixup'+str(mixup_label)+'.jpg',mixup_image)
            #print(mixup_label)
        else: 
            pass
            #print("********** not mixup **********")
        self.img2 = img1
        self.boxes2 = boxes1
        self.labels2 =  labels1
        if self.with_mask:
            self.masks2 = masks1
        return results



@PIPELINES.register_module()
class ResizeOBB(Resize):
    def _resize_obbs(self, results):
        img_shape = results['img_shape']
        for key in results.get('obb_fields', []):
            obbs = results[key] * np.hstack((results['scale_factor'], results['scale_factor']))
            obbs[:, 0::2] = np.clip(obbs[:, 0::2], 0, img_shape[1] - 1)
            obbs[:, 1::2] = np.clip(obbs[:, 1::2], 0, img_shape[0] - 1)
            results[key] = obbs

    def __call__(self, results):
        super().__call__(results)
        self._resize_obbs(results)

        return results


@PIPELINES.register_module()
class RandomFlipOBB(RandomFlip):
    def obb_flip(self, obbs, img_shape, direction):
        """Flip obbs horizontally.

        Args:
            obbs(ndarray): shape (..., 8*k) (x1, y1, x2, y2, x3, y3, x4, y4)
            img_shape(tuple): (height, width)
        """
        assert obbs.shape[-1] % 8 == 0
        flipped = obbs.copy()
        if direction == 'horizontal':
            w = img_shape[1]
            flipped[..., 0::2] = w - flipped[..., 0::2] - 1

        elif direction == 'vertical':
            h = img_shape[0]
            flipped[..., 1::2] = h - flipped[..., 1::2] - 1
        else:
            raise ValueError(
                'Invalid flipping direction "{}"'.format(direction))
        flipped = np.array([pointobb_best_point_sort(pointobb) for pointobb in flipped.tolist()])
        
        return flipped

    def __call__(self, results):
        super().__call__(results)
        # flip obb (pointobb)
        if results['flip']:
            for key in results.get('obb_fields', []):
                results[key] = self.obb_flip(results[key],
                                            results['img_shape'],
                                            results['flip_direction'])
        return results

@PIPELINES.register_module()
class OBBConverter(object):
    """convert pointobb to corresponding regression-based obbs
    """

    def __init__(self,
                encoding_method='thetaobb'):
        self.encoding_method = encoding_method

    def _pointobb2thetaobb(self, results):
        """
        convert pointobb to thetaobb
            :param self: 
            :param pointobb: list, [x1, y1, x2, y2, x3, y3, x4, y4]
        """
        for key in results.get('obb_fields', []):
            obbs = results[key]
            thetaobbs = []
            for pointobb in obbs.tolist():
                pointobb = np.int0(np.array(pointobb))
                pointobb.resize(4, 2)
                rect = cv2.minAreaRect(pointobb)
                x, y, w, h, theta = rect[0][0], rect[0][1], rect[1][0], rect[1][1], rect[2]
                theta = theta / 180.0 * np.pi
                thetaobbs.append([x, y, w, h, theta])

            results[key] = np.array(thetaobbs)

    def __call__(self, results):
        if self.encoding_method == 'thetaobb':
            self._pointobb2thetaobb(results)
        elif self.encoding_method == 'pointobb':
            pass
        else:
            raise NotImplementedError(f'error encoding method: {self.encoding_method}')
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += ('(encoding_method={})').format(self.encoding_method)
        return repr_str
@PIPELINES.register_module()
class RandomRotate(object):
    """Rotate the image & bbox & mask.

    If the input dict contains the key "rotate", then the flag will be used,
    otherwise it will be randomly decided by a ratio specified in the init
    method.

    Args:
        rotate_ratio (float, optional): The rotation probability.
        choice (float or str, optional): The rotation angle choice. If choice == 'any', angle = range(0, 359, 1)
    """

    def __init__(self, rotate_ratio=1.0, choice=(0, 90, 180, 270)):
        self.rotate_ratio = rotate_ratio
        if isinstance(choice, (list, tuple)):
            self.choice = choice
        elif isinstance(choice, str):
            self.choice = list(range(0, 359, 1))
        else:
            raise NotImplementedError

        if rotate_ratio is not None:
            assert rotate_ratio >= 0 and rotate_ratio <= 1
        assert isinstance(self.choice, (list, tuple))

    def get_corners(self, bboxes):
        """Get corners of bounding boxes
        
        Parameters
        ----------
        
        bboxes: numpy.ndarray
            Numpy array containing bounding boxes of shape `N X 4` where N is the 
            number of bounding boxes and the bounding boxes are represented in the
            format `x1 y1 x2 y2`
        
        returns
        -------
        
        numpy.ndarray
            Numpy array of shape `N x 8` containing N bounding boxes each described by their 
            corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`      
            
        """
        width = (bboxes[:,2] - bboxes[:,0]).reshape(-1,1)
        height = (bboxes[:,3] - bboxes[:,1]).reshape(-1,1)
        
        x1 = bboxes[:,0].reshape(-1,1)
        y1 = bboxes[:,1].reshape(-1,1)
        
        x2 = x1 + width
        y2 = y1 
        
        x3 = x1
        y3 = y1 + height
        
        x4 = bboxes[:,2].reshape(-1,1)
        y4 = bboxes[:,3].reshape(-1,1)
        
        corners = np.hstack((x1,y1,x2,y2,x3,y3,x4,y4))
        
        return corners

    def bbox_rotate(self, bboxes, img_shape, rotate_angle):
        """rotate bboxes.

        Args:
            bboxes(ndarray): shape (..., 4*k)
            img_shape(tuple): (height, width)
        """
        assert bboxes.shape[-1] % 4 == 0
        if bboxes.shape[0] == 0:
            return bboxes
        corners = self.get_corners(bboxes)
        corners = np.hstack((corners, bboxes[:, 4:]))

        corners = corners.reshape(-1, 2)
        corners = np.hstack((corners, np.ones((corners.shape[0], 1), dtype = type(corners[0][0]))))
        angle = rotate_angle
        h, w, _ = img_shape
        cx, cy = w / 2, h / 2
        M = cv2.getRotationMatrix2D((cx, cy), -angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))
        M[0, 2] += (nW / 2) - cx
        M[1, 2] += (nH / 2) - cy

        calculated = np.dot(M, corners.T).T
        calculated = np.array(calculated, dtype=np.float32)
        calculated = calculated.reshape(-1, 8)

        x_ = calculated[:, [0, 2, 4, 6]]
        y_ = calculated[:, [1, 3, 5, 7]]
        
        xmin = np.min(x_, 1).reshape(-1, 1)
        ymin = np.min(y_, 1).reshape(-1, 1)
        xmax = np.max(x_, 1).reshape(-1, 1)
        ymax = np.max(y_, 1).reshape(-1, 1)
        
        rotated = np.hstack((xmin, ymin, xmax, ymax))
        
        return rotated

    def __call__(self, results):
        if 'rotate' not in results:
            rotate = True if np.random.rand() < self.rotate_ratio else False
            results['rotate'] = rotate
        if 'rotate_angle' not in results:
            results['rotate_angle'] = np.random.choice(self.choice)
        if results['rotate']:
            # rotate image
            results['img'] = mmcv.imrotate(
                results['img'], results['rotate_angle'], auto_bound=False)
            
            # rotate bboxes
            for key in results.get('bbox_fields', []):
                results[key] = self.bbox_rotate(results[key],
                                              results['img_shape'],
                                              results['rotate_angle'])
            # rotate masks
            for key in results.get('mask_fields', []):
                masks = [
                    mmcv.imrotate(mask, results['rotate_angle'], auto_bound=False)
                    for mask in results[key]
                ]
                if masks:
                    rotated_masks = np.stack(masks)
                else:
                    rotated_masks = np.empty(
                        (0, ) + results['img_shape'], dtype=np.uint8)

                results[key] = BitmapMasks(rotated_masks, results['img_shape'][0], results['img_shape'][1])

            # rotate segs
            for key in results.get('seg_fields', []):
                results[key] = mmcv.imrotate(
                    results[key], results['rotate_angle'], auto_bound=False)

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(rotate_ratio={self.rotate_ratio})'
        
        return repr_str


@PIPELINES.register_module()
class Show(object):
    """convert pointobb to corresponding regression-based obbs
    """
    def __init__(self, item=None):
        pass

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

    def __call__(self, results):
        img = results['img']
        thetaobbs = results['gt_obbs']
        for thetaobb in thetaobbs:
            img = self._show_thetaobb(img, thetaobb)
        
        mmcv.imshow(img)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str