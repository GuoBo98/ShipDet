checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None #'/data2/pd/sdc/shipdet/v1/works_dir/mmdet/detectors/epoch_12.pth'
work_dir = '/data2/pd/sdc/shipdet/v1/works_dir/mmdet/detectors_cascade_mask/'
resume_from = None#'/data2/pd/sdc/shipdet/v1/works_dir/mmdet/detectors_mixup_mstrain_ablu_3x/latest.pth'
workflow = [('train', 1)]
#workflow = [('train', 1), ('val' , 1)]