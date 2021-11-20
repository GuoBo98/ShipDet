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
load_from =  '/data2/pd/sdc/multidet/v0/works_dir/mmdet/detectoRSmask_ablu/epoch_00.pth'
resume_from = None
workflow = [('train', 1)]
#workflow = [('train', 1), ('val' , 1)]