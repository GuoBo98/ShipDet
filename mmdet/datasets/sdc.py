import os
import numpy as np
import cv2
from pycocotools.coco import COCO
import tempfile
from mmdet.datasets import CocoDataset
from mmdet.datasets import DATASETS

@DATASETS.register_module()
class SDCDataset(CocoDataset):
    CLASSES=('Cargo vessel','Ship','Motorboat','Fishing boat','Destroyer','Tugboat','Loose pulley','Warship','Engineering ship','Amphibious ship','Cruiser','Frigate','Submarine','Aircraft carrier','Hovercraft','Command ship')
    def __init__(self,
                ann_file,
                pipeline,
                classes=None,
                data_root=None,
                img_prefix='',
                seg_prefix=None,
                proposal_file=None,
                test_mode=False,
                filter_empty_gt=True):
        super(SDCDataset, self).__init__(ann_file=ann_file,
                                                pipeline=pipeline,
                                                classes=classes,
                                                data_root=data_root,
                                                img_prefix=img_prefix,
                                                seg_prefix=seg_prefix,
                                                proposal_file=proposal_file,
                                                test_mode=test_mode,
                                                filter_empty_gt=filter_empty_gt)
        self.ann_file = ann_file
    def pre_pipeline(self, results):
        super(DOTADataset, self).pre_pipeline(results=results)
        results['obb_fields'] = []
    def get_properties(self, idx):
        img_id = self.data_infos[idx]['id']
        ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
        ann_info = self.coco.load_anns(ann_ids)

        return ann_info[0].keys()
    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds = []
        ids_with_ann = set(_['image_id'] for _ in self.coco.anns.values())
        for i, img_info in enumerate(self.data_infos):
            img_id = img_info['id']
            ann_ids = self.coco.getAnnIds(imgIds=[img_id])
            ann_info = self.coco.loadAnns(ann_ids)
            all_iscrowd = all([_['iscrowd'] for _ in ann_info])
            if self.filter_empty_gt and (self.img_ids[i] not in ids_with_ann
                                         or all_iscrowd):
                continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds
    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,
                labels, masks, seg_map. "masks" are raw annotations and not
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []
        gt_obbs = [] ## add points bboxes

        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']

            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                gt_masks_ann.append(ann['segmentation'])
                gt_obbs.append(ann['pointobb'])

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        if gt_obbs:
            gt_obbs = np.array(gt_obbs, dtype=np.float32)
        else:
            gt_obbs = np.zeros((0, 8), dtype=np.float32)
        seg_map = img_info['filename'].replace('jpg', 'png')

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            seg_map=seg_map,
            obbs=gt_obbs)

        return ann
    def theta_or_points2bbox(self,thetaORpoints):
        if list(thetaORpoints) == 8:
            points = thetaORpoints
        elif list(thetaORpoints) == 5:
            thetaobb = thetaORpoints
            box = cv2.boxPoints(((thetaobb[0], thetaobb[1]), (thetaobb[2], thetaobb[3]), thetaobb[4] * 180.0 / np.pi))
            box = np.reshape(box, [-1, ]).tolist()
            points = [box[0], box[1], box[2], box[3], box[4], box[5], box[6], box[7]]
        else:
            raise NotImplementedError('not implemement now')
        xmin = min(points[0::2])
        ymin = min(points[1::2])
        xmax = max(points[0::2])
        ymax = max(points[1::2])
        bbox_w = xmax - xmin
        bbox_h = ymax - ymin
        bbox = [xmin,ymin,bbox_w,bbox_h]
        return bbox
    def _det2json(self, results):
        """Convert obb detection results to COCO json style."""
        json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            result = results[idx]
            for label in range(len(result)):
                rbboxes = result[label]
                for i in range(rbboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.theta_or_points2bbox(rbboxes[i][:-1])
                    data['rbbox'] = rbboxes[i][:-1]
                    data['score'] = float(bboxes[i][-1])
                    data['category_id'] = self.cat_ids[label]
                    json_results.append(data)
        return json_results

    def _segm2json(self, results):
        """Convert instance segmentation results to COCO json style."""
        bbox_json_results = []
        segm_json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            det, seg = results[idx]
            for label in range(len(det)):
                # bbox results
                bboxes = det[label]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.theta_or_points2bbox(bboxes[i][:-1])
                    data['rbbox'] = bboxes[i][:-1]
                    data['score'] = float(bboxes[i][-1])
                    data['category_id'] = self.cat_ids[label]
                    bbox_json_results.append(data)

                # segm results
                # some detectors use different scores for bbox and mask
                if isinstance(seg, tuple):
                    segms = seg[0][label]
                    mask_score = seg[1][label]
                else:
                    segms = seg[label]
                    mask_score = [bbox[4] for bbox in bboxes]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.theta_or_points2bbox(bboxes[i][:-1])
                    data['rbbox'] = bboxes[i][:-1]
                    data['score'] = float(mask_score[i])
                    data['category_id'] = self.cat_ids[label]
                    if isinstance(segms[i]['counts'], bytes):
                        segms[i]['counts'] = segms[i]['counts'].decode()
                    data['segmentation'] = segms[i]
                    segm_json_results.append(data)
        return bbox_json_results, segm_json_results

    def format_results(self, results, jsonfile_prefix=None, **kwargs):
        """Format the results to json (standard format for COCO evaluation).

        Args:
            results (list[tuple | numpy.ndarray]): Testing results of the
                dataset.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            tuple: (result_files, tmp_dir), result_files is a dict containing \
                the json filepaths, tmp_dir is the temporal directory created \
                for saving json files when jsonfile_prefix is not specified.
        """
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: {} != {}'.
            format(len(results), len(self)))

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None
        result_files = self.results2json(results, jsonfile_prefix)
        return result_files, tmp_dir