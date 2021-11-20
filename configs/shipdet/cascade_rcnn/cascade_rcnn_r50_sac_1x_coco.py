_base_ = [
    './cascade_rcnn_r50_fpn.py',
    '../faster_rcnn/coco_detection.py',
    '../faster_rcnn/schedule_1x.py', '../faster_rcnn/default_runtime.py'
]
model = dict(
    backbone=dict(
        type='DetectoRS_ResNet',
        conv_cfg=dict(type='ConvAWS'),
        sac=dict(type='SAC', use_deform=True),
        stage_with_sac=(False, True, True, True)))
