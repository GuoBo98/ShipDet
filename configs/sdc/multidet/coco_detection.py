dataset_type = 'CocoDataset'
# classes = ('ship',)
classes = ('Cargo vessel','Ship','Motorboat','Fishing boat','Destroyer','Tugboat','Loose pulley','Warship','Engineering ship','Amphibious ship','Cruiser','Frigate','Submarine','Aircraft carrier','Hovercraft','Command ship')

data_root = '/data2/pd/sdc/multidet/v0/coco/'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1024, 1024), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
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
        ann_file=data_root + 'annotations/shipdet_trainval_v1.json',
        img_prefix=data_root + 'trainval//',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'annotations/shipdet_test_v1.json',
        img_prefix=data_root + 'test/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'annotations/shipdet_test_v1.json',
        img_prefix=data_root + 'test/',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')
