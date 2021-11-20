_base_ = [
    './retinanet_r50_fpn.py',
    '../faster_rcnn/coco_detection.py',
    '../faster_rcnn/schedule_1x.py', '../faster_rcnn/default_runtime.py'
]
# optimizer
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)
