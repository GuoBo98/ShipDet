from argparse import ArgumentParser
import mmcv
from mmdet.apis import inference_detector, init_detector, show_result_pyplot
import os 
import xml.etree.ElementTree as ET
import numpy as np
from pktool.visualization import imshow_bboxes, imshow_rbboxes
from pktool import mkdir_or_exist,mask2rbbox
import cv2

def roRect_soft_nms(rbboxes, scores, iou_threshold=0.5, score_threshold=0.001, **kwargs):
    """rotation non-maximum suppression (NMS) on the boxes according to their intersection-over-union (IoU)
    Arguments:
        rboxes {np.array} -- [N * 5] (cx, cy, w, h, theta (rad/s))
        scores {np.array} -- [N * 1]
        iou_threshold {float} -- threshold for IoU
    """
    rbboxes = np.array(rbboxes)
    scores = np.array(scores)
    cx = rbboxes[:, 0]
    cy = rbboxes[:, 1]
    w = rbboxes[:, 2]
    h = rbboxes[:, 3]
    theta = rbboxes[:, 4] * 180.0 / np.pi

    order = scores.argsort()[::-1]

    areas = w * h
    
    keep = []
    while order.size > 0:
        best_rbox_idx = order[0]
        keep.append(best_rbox_idx)

        best_rbbox = np.array([cx[best_rbox_idx], 
                               cy[best_rbox_idx], 
                               w[best_rbox_idx], 
                               h[best_rbox_idx], 
                               theta[best_rbox_idx]])
        remain_rbboxes = np.hstack((cx[order[1:]].reshape(1, -1).T, 
                                    cy[order[1:]].reshape(1,-1).T, 
                                    w[order[1:]].reshape(1,-1).T, 
                                    h[order[1:]].reshape(1,-1).T, 
                                    theta[order[1:]].reshape(1,-1).T))

        inters = []
        for remain_rbbox in remain_rbboxes:
            rbbox1 = ((best_rbbox[0], best_rbbox[1]), (best_rbbox[2], best_rbbox[3]), best_rbbox[4])
            rbbox2 = ((remain_rbbox[0], remain_rbbox[1]), (remain_rbbox[2], remain_rbbox[3]), remain_rbbox[4])
            inter = cv2.rotatedRectangleIntersection(rbbox1, rbbox2)[1]
            if inter is not None:
                inter_pts = cv2.convexHull(inter, returnPoints=True)
                inter = cv2.contourArea(inter_pts)
                inters.append(inter)
            else:
                inters.append(0)

        inters = np.array(inters)

        iou = inters / (areas[best_rbox_idx] + areas[order[1:]] - inters)

        inds = np.where(iou <= iou_threshold)[0]

        ##softNMS
        # weights = np.ones(iou.shape) - iou
        # scores[order[1:]] = weights * scores[order[1:]]
        # inds = np.where(scores[order[1:]] > score_threshold)[0]
        
        order = order[inds + 1]

    return keep

def __xml_parse__(label_file):
    tree = ET.parse(label_file)
    root = tree.getroot()
    bboxes = []
    labels = []
    for single_object in root.findall('object'):
        bndbox = single_object.find('bndbox')
        object_struct = {}

        xmin = float(bndbox.find('xmin').text)
        ymin = float(bndbox.find('ymin').text)
        xmax = float(bndbox.find('xmax').text)
        ymax = float(bndbox.find('ymax').text)


        bboxes.append([xmin, ymin, xmax, ymax])
        labels.append(1)
    return bboxes,labels


def main():
    '''
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    args = parser.parse_args()
    '''
    colors_map = ['black','Red','Red','yellow','purple','cyan','brown','Red','orange','magenta','Lime','Teal','Maroon','Grey','Apricot','Olive','Beige']

    classes = {'Cargo vessel':1,'Ship':2,'Motorboat':3,'Fishing boat':4,'Destroyer':5,'Tugboat':6,'Loose pulley':7,'Warship':8,'Engineering ship':9,'Amphibious ship':10,'Cruiser':11,'Frigate':12,'Submarine':13,'Aircraft carrier':14,'Hovercraft':15,'Command ship':16}


    config = '/data2/pd/sdc/shipdet/v1/works_dir/mmdet/detectors_cascade_mask/detectors_cascade_mask_r50_1x_coco.py'
    img_floder = '/data2/pd/sdc/shipdet/v1/test/images/'
    label_floder = None 
    checkpoint = '/data2/pd/sdc/shipdet/v1/works_dir/mmdet/detectors_cascade_mask/latest.pth'
    device = 'cuda:0'

    selectDir = '/data2/pd/sdc/shipdet/v1/testvis/diffimg/'

    mkdir_or_exist(selectDir)
    without_gt = None
    if label_floder is None:
        without_gt = False
    else:
        without_gt = True
    # build the model from a config file and a checkpoint file
    model = init_detector(config, checkpoint, device=device)


    img_list = ['P0706_0_158.png']
    # for imgfile in os.listdir(img_floder):
    for imgfile in img_list:
        img = img_floder + '/' + imgfile

        print('detecting {}'.format(imgfile))
        # test a single image
        result = inference_detector(model, img)
        # show the results
        if isinstance(result, tuple):
            bbox_result, segm_result = result
            if isinstance(segm_result, tuple):
                segm_result = segm_result[0]  # ms rcnn
        else:
            bbox_result, segm_result = result, None
        
        print('detected {}'.format(imgfile))
        # bboxes = np.vstack(bbox_result)
        rbboxes = []
        scores = []
        labels = []
        for catID in range(len(segm_result)):
            catID_bbox_result,catID_segm_result = bbox_result[catID],segm_result[catID]
            if len(catID_segm_result)>0:
                for numID in range(len(catID_segm_result)):
                    numID_segm_result=catID_segm_result[numID]
                    numID_bbox_result = catID_bbox_result[numID]

                    thetaobb, _ = mask2rbbox(numID_segm_result)
                    score = numID_bbox_result[-1]
                    rbboxes.append(thetaobb)
                    scores.append(score)
                    labels.append(catID+1)
        nms_rbboxes = []
        nms_scores = []
        snme_labels = []        
        #nms
        need_nms = False
        if need_nms and len(rbboxes)>1:
            keep = roRect_soft_nms(rbboxes, scores, iou_threshold=0.5, score_threshold=0.001)

            for idx in keep:
                nms_rbboxes.append(rbboxes[idx])
                nms_scores.append(scores[idx])
                snme_labels.append(labels[idx])
        else:
            nms_rbboxes = rbboxes
            nms_scores = scores
            snme_labels = labels
        img = cv2.imread(img)

        print('show {}'.format(imgfile))
        save_img_file = selectDir + imgfile
        imshow_rbboxes(img,nms_rbboxes,labels=snme_labels,colors_map=colors_map,show=False,cls_map=classes,out_file=save_img_file,scores=nms_scores)#scores=nms_scores,show_score=True,
        # scores = bboxes[:,4]
        # bboxes = bboxes[:,:4]
        # labels = [0 for i in range(len(bboxes))] 
        # if without_gt:
        #     labelFile=label_floder+'/'+imgfile.split('.png')[0] + '.xml'
        #     gt_bboxes, gt_label = __xml_parse__(labelFile)
        #     bboxes=np.vstack((bboxes,np.array(gt_bboxes)))
        #     labels.extend(gt_label)
        #     scores = np.vstack((scores,np.ones((len(gt_bboxes),1))))
        # imshow_bboxes(img,bboxes,labels,scores=scores,score_threshold=0.5,show_score=True,selectDir=selectDir)#蓝色为检测结果，红色gt
if __name__ == '__main__':
    main()
