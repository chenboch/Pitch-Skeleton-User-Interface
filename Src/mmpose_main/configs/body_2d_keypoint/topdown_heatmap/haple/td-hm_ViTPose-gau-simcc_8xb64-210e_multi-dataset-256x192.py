_base_ = ['../../../_base_/default_runtime.py']

# runtime
train_cfg = dict(max_epochs=400, val_interval=10)# 训练轮数，测试间隔

# optimizer
custom_imports = dict(
    imports=['mmpose.engine.optim_wrappers.layer_decay_optim_wrapper'],
    allow_failed_imports=False)

optim_wrapper = dict(# 优化器和学习率
    optimizer=dict(
        type='AdamW', lr=1e-3, betas=(0.9, 0.999), weight_decay=0.1),
    paramwise_cfg=dict(
        num_layers=12,
        layer_decay_rate=0.75,
        custom_keys={
            'bias': dict(decay_multi=0.0),
            'pos_embed': dict(decay_mult=0.0),
            'relative_position_bias_table': dict(decay_mult=0.0),
            'norm': dict(decay_mult=0.0),
        },
    ),
    constructor='LayerDecayOptimWrapperConstructor',
    clip_grad=dict(max_norm=1., norm_type=2),
)

# learning policy
param_scheduler = [
    dict(# warmup策略
        type='LinearLR', begin=0, end=500, start_factor=0.001,
        by_epoch=False),  # warm-up
    dict(# scheduler
        type='MultiStepLR',
        begin=0,
        end=210,
        milestones=[170, 200],
        gamma=0.1,
        by_epoch=True)
]

# automatically scaling LR based on the actual training batch size
# auto_scale_lr = dict(base_batch_size=512)# 根据batch_size自动缩放学习率

# hooks
default_hooks = dict(
    checkpoint=dict(save_best='coco/AP', rule='greater', max_keep_ckpts=1))

# codec settings
 # 定义数据编解码器，用于生成target和对pred进行解码，同时包含了输入图片和输出heatmap尺寸等信息
# codec = dict(
#     type='UDPHeatmap', input_size=(192, 256), heatmap_size=(48, 64), sigma=2)
num_keypoints = 26
input_size = (192, 256)
# codec settings
codec = dict(
    type='SimCCLabel',
    input_size=input_size,
    sigma=(4.9, 5.66),
    simcc_split_ratio=2.0,
    normalize=False,
    use_dark=False)
# resume = True
# load_from = 'work_dirs/multi_dataset/epoch_210.pth'

# model settings
model = dict(
    type='TopdownPoseEstimator', # 模型结构决定了算法流程
    data_preprocessor=dict(# 数据归一化和通道顺序调整，作为模型的一部分
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        type='mmpretrain.VisionGAUTransformer',
        arch='base',
        img_size=(256, 192),
        patch_size=16,
        qkv_bias=True,
        drop_path_rate=0.3,
        with_cls_token=False,
        out_type='featmap',
        patch_cfg=dict(padding=2),
#         frozen_stages=9, #真禛測試出來凍結第九層效果不錯
#         init_cfg=dict(# 预训练参数，只加载backbone权重用于迁移学习
#             type='Pretrained',
#             checkpoint= rf"checkpoint/epoch_210.pth"),
    ),
    head=dict(
        type='RTMCCHead',
        in_channels=768,
        out_channels=num_keypoints,
        input_size=codec['input_size'],
        in_featuremap_size=tuple([s // 16 for s in codec['input_size']]),
        simcc_split_ratio=codec['simcc_split_ratio'],
        final_layer_kernel_size=3,
        gau_cfg=dict(
            hidden_dims=256,
            s=128,
            expansion_factor=2,
            dropout_rate=0.,
            drop_path=0.,
            act_fn='SiLU',
            use_rel_bias=False,
            pos_enc=False),
        loss=dict(
            type='KLDiscretLoss',
            use_target_weight=True,
            beta=10.,
            label_softmax=True),
        decoder=codec),
    test_cfg=dict(
        flip_test=True,# 开启测试时水平翻转集成
        # flip_mode='heatmap', # 对heatmap进行翻转
        # shift_heatmap=False, # 对翻转后的结果进行平移提高精度
    ))

# base dataset settings
data_root = 'data/'# 数据存放路径
dataset_type = 'Halpe26Dataset'# 数据集类名
data_mode = 'topdown'# 算法结构类型，用于指定标注信息加载策略

# pipelines
train_pipeline = [ # 训练时数据增强
    dict(type='LoadImage'),# 加载图片
    dict(type='GetBBoxCenterScale'),# 根据bbox获取center和scale
    dict(type='RandomFlip', direction='horizontal'),# 生成随机翻转变换矩阵
    dict(type='RandomHalfBody'),# 随机半身增强
    dict(type='RandomBBoxTransform'), # 生成随机位移、缩放、旋转变换矩阵
    dict(type='TopdownAffine', input_size=codec['input_size'], use_udp=True), # 根据变换矩阵更新目标数据
    dict(type='GenerateTarget', encoder=codec),# 根据目标数据生成监督信息
    dict(type='PackPoseInputs')# 对target进行打包用于训练
]
val_pipeline = [
    dict(type='LoadImage'),# 加载图片
    dict(type='GetBBoxCenterScale'),# 根据bbox获取center和scale
    dict(type='TopdownAffine', input_size=codec['input_size'], use_udp=True),# 根据变换矩阵更新目标数据
    dict(type='PackPoseInputs')# 对target进行打包用于训练
]

# mapping
coco_halpe26 = [(i, i) for i in range(17)]

aic_halpe26 = [(0, 6), (1, 8), (2, 10), (3, 5), (4, 7),
               (5, 9), (6, 12), (7, 14), (8, 16), (9, 11), (10, 13), (11, 15),
               (12, 17), (13, 18)]

crowdpose_halpe26 = [(0, 5), (1, 6), (2, 7), (3, 8), (4, 9), (5, 10), (6, 11),
                     (7, 12), (8, 13), (9, 14), (10, 15), (11, 16), (12, 17),
                     (13, 18)]

mpii_halpe26 = [
    (0, 16),
    (1, 14),
    (2, 12),
    (3, 11),
    (4, 13),
    (5, 15),
    (8, 18),
    (9, 17),
    (10, 10),
    (11, 8),
    (12, 6),
    (13, 5),
    (14, 7),
    (15, 9),
]

halpe_halpe26 = [(i, i) for i in range(26)]

ochuman_halpe26 = [(i, i) for i in range(17)]

# train datasets
dataset_coco = dict(
    type=dataset_type,
    data_root=data_root,
    data_mode=data_mode,
    ann_file='coco/annotations/person_keypoints_train2017.json',
    data_prefix=dict(img='coco/train2017/'),
    pipeline=[
        dict(
            type='KeypointConverter',
            num_keypoints=num_keypoints,
            mapping=coco_halpe26)
    ],
)

dataset_aic = dict(
    type='AicDataset',
    data_root=data_root,
    data_mode=data_mode,
    ann_file='aic/annotations/aic_train.json',
    data_prefix=dict(img='aic/ai_challenger_keypoint'
                     '_train_20170902/keypoint_train_images_20170902/'),
    pipeline=[
        dict(
            type='KeypointConverter',
            num_keypoints=num_keypoints,
            mapping=aic_halpe26)
    ],
)

dataset_crowdpose = dict(
    type='CrowdPoseDataset',
    data_root=data_root,
    data_mode=data_mode,
    ann_file='crowdpose/annotations/mmpose_crowdpose_trainval.json',
    data_prefix=dict(img='crowdpose/images/'),
    pipeline=[
        dict(
            type='KeypointConverter',
            num_keypoints=num_keypoints,
            mapping=crowdpose_halpe26)
    ],
)

dataset_mpii = dict(
    type='MpiiDataset',
    data_root=data_root,
    data_mode=data_mode,
    ann_file='mpii/annotations/train.json',
    data_prefix=dict(img='mpii/images/'),
    pipeline=[
        dict(
            type='KeypointConverter',
            num_keypoints=num_keypoints,
            mapping=mpii_halpe26)
    ],
)

dataset_halpe = dict(
    type='HalpeDataset',
    data_root=data_root,
    data_mode=data_mode,
    ann_file='halpe/annotations/halpe_train_v1_26.json',
    data_prefix=dict(img='halpe/images/train2015/'),
    pipeline=[
        dict(
            type='KeypointConverter',
            num_keypoints=num_keypoints,
            mapping=halpe_halpe26)
    ],
)


dataset_ochuman = dict(
    type='OCHumanDataset',
    data_root=data_root,
    data_mode=data_mode,
    ann_file='ochuman/annotations/'
    'ochuman_coco_format_val_range_0.00_1.00.json',
    data_prefix=dict(img='ochuman/images/'),
    pipeline=[
        dict(
            type='KeypointConverter',
            num_keypoints=num_keypoints,
            mapping=ochuman_halpe26)
    ],
)

# data loaders
train_dataloader = dict(# 训练数据加载
    batch_size=64,# 批次大小
    num_workers=4, # 数据加载进程数
    persistent_workers=True,# 在不活跃时维持进程不终止，避免反复启动进程的开销
    sampler=dict(type='DefaultSampler', shuffle=True), # 采样策略，打乱数据
    dataset=dict(
        type='CombinedDataset',# 数据集类名
        metainfo=dict(from_file="configs/_base_/datasets/halpe26.py"), # 指定元信息配置文件
        datasets=[
            dataset_coco,
            dataset_aic,
            dataset_crowdpose,
            dataset_mpii,
            dataset_halpe,
        ],
        pipeline=train_pipeline,# 数据流水线
        test_mode = False
    ))


test_dataloader = dict(
    batch_size=32,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type='CombinedDataset',
        data_root=data_root,
        data_mode=data_mode,
        metainfo=dict(from_file="configs/_base_/datasets/halpe26.py"),
        ann_file='halpe/annotations/halpe_val_v1_26.json',
        bbox_file=None,
        data_prefix=dict(img='coco/val2017/'),
        test_mode=True,
        pipeline=val_pipeline,
    ))
# val_dataloader = test_dataloader

# evaluators
# val_evaluator = dict(
#     type='PCKAccuracy')
val_cfg=None
# test_evaluator = dict(
#     type='CocoMetric',
#     ann_file= 'data/halpe/annotations/halpe_val_v1_26.json')

test_evaluator = [
    dict(type="PCKAccuracy"),
    dict(
        type="CocoMetric", ann_file=data_root + 'coco/annotations/person_keypoints_val2017.json'
#         type="CocoMetric", ann_file=data_root + "halpe/annotations/halpe_val_v1_26.json"
    ),
]

# val_dataloader = test_evaluator
