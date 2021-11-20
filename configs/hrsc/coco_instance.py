dataset_type = 'CocoDataset'
classes = ('航母','塔拉瓦级通用两栖攻击舰','提康德罗加级巡洋舰','奥斯汀级两栖船坞运输舰','阿利伯克级驱逐舰','船','佩里级护卫舰','惠德贝岛级船坞登陆舰','圣安东尼奥级两栖船坞运输舰','指挥舰','潜艇','蓝岭级指挥舰')
data_root = '/data/pd/hrsc2016/v1/coco/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
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
        img_scale=(1333, 800),
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
        ann_file=data_root + 'annotations/hrsc2016_trainval_v1.json',
        img_prefix=data_root + 'trainval/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'annotations/hrsc2016_test_v1.json',
        img_prefix=data_root + 'test/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'annotations/hrsc2016_test_v1.json',
        img_prefix=data_root + 'test/',
        pipeline=test_pipeline))
evaluation = dict(metric=['bbox', 'segm'])
