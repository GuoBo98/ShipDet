dataset_type = 'SDCDataset'
classes = ('ship',)
data_root = '/data2/pd/sdc/shipdet/v1/coco/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotationsOBB',
         with_bbox=True,
         with_obb=True),
    dict(type='ResizeOBB', img_scale=(1024, 1024), keep_ratio=True),
    dict(type='RandomFlipOBB', flip_ratio=0.5),
    dict(type='OBBConverter', encoding_method='thetaobb'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    # dict(type='Show'),
    dict(type='DefaultFormatBundleOBB'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_obbs']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.5),
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
        ann_file=data_root + 'annotations/shipdet_trainval_v1_obb.json',
        img_prefix=data_root + 'trainval/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'annotations/shipdet_test_v1_obb.json',
        img_prefix=data_root + 'test/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'annotations/shipdet_test_v1_obb.json',
        img_prefix=data_root + 'test/',
        pipeline=test_pipeline))

