dataset_type = 'CocoDataset'
data_root = '/home/pd/data/ship/v3/coco/'
classes = ('船-两栖指挥舰-蓝岭级','船-航空母舰-尼米兹级','船-驱逐舰-阿利伯克级',
            '船-巡洋舰-提康德罗加级','船-军辅船-NULL','船-驱逐舰-旗风级',
            '船-驱逐舰-金刚级','船-驱逐舰-朝雾级','船-驱逐舰-秋月级',
            '船-驱逐舰-高波级','船-扫雷母舰-浦贺级','船-海洋观测舰--二见级',
            '船-潜艇救难舰--NULL','船-潜艇-NULL-NULL','船-巡洋舰-天雾级',
            '船-护卫舰-阿武隈级','船-巡洋舰-金刚级','船-巡洋舰-高波级',
            '船-两栖登陆舰-海洋之子级','船-直升机驱逐舰--出云级','船-驱逐舰-村雨级',
            '船-驱逐舰--飞鸟级')
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', img_scale=(1024, 1024), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'annotations/ship_trainval_v3.json',
        img_prefix='/home/pd/data/images/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'annotations/ship_trainval_v3.json',
        img_prefix='/home/pd/data/images/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'annotations/ship_trainval_v3.json',
        img_prefix='/home/pd/data/images/',
        pipeline=test_pipeline))
evaluation = dict(metric=['bbox','segm'])
