# we follow the online training settings  from solofusion
num_gpus = 8
samples_per_gpu = 2
# num_iters_per_epoch = int(28130 // (num_gpus * samples_per_gpu) * 4.554)
num_iters_per_epoch = int(28130 // (num_gpus * samples_per_gpu) * 1.0)
num_epochs = 48
checkpoint_epoch_interval = 1
use_custom_eval_hook = True

# Each nuScenes sequence is ~40 keyframes long. Our training procedure samples
# sequences first, then loads frames from the sampled sequence in order
# starting from the first frame. This reduces training step-to-step diversity,
# lowering performance. To increase diversity, we split each training sequence
# in half to ~20 keyframes, and sample these shorter sequences during training.
# During testing, we do not do this splitting.
train_sequences_split_num = 2
test_sequences_split_num = 1

# By default, 3D detection datasets randomly choose another sample if there is
# no GT object in the current sample. This does not make sense when doing
# sequential sampling of frames, so we disable it.
filter_empty_gt = False

# Long-Term Fusion Parameters
do_history = False
history_cat_num = 16
history_cat_conv_out_channels = 160

# Copyright (c) Phigent Robotics. All rights reserved.

_base_ = [ '../_base_/default_runtime.py']
# Global
# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-40, -40, -1.0, 40, 40, 5.4]
# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

data_config = {
    'cams': [
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT',
        'CAM_BACK', 'CAM_BACK_RIGHT'
    ],
    'Ncams':
        6,
    'input_size': (640, 1600),
    'src_size': (900, 1600),

    # Augmentation
    'resize': (-0.06, 0.11),
    'rot': (-5.4, 5.4),
    'flip': True,
    'crop_h': (0.0, 0.0),
    'resize_test': 0.00,
}

bda_aug_conf = dict(
    rot_lim=(-22.5, 22.5),
    scale_lim=(1., 1.),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5)


use_checkpoint = True
sync_bn = True

# Model
grid_config = {
    'x': [-40, 40, 0.8],
    'y': [-40, 40, 0.8],
    'z': [-1, 5.4, 0.8],
    'depth': [2.0, 42.0, 0.5],
}
depth_categories = 80  # (grid_config['depth'][1]-grid_config['depth'][0])//grid_config['depth'][2]

## config for bevformer
grid_config_bevformer = {
    'x': [-40, 40, 0.8],
    'y': [-40, 40, 0.8],
    'z': [-1, 5.4, 1.6],
}
bev_h_ = 100
bev_w_ = 100
numC_Trans = 80
_dim_ = 256
_pos_dim_ = 40
_ffn_dim_ = numC_Trans * 4
_num_levels_ = 1

empty_idx = 16  # noise 0-->255
num_cls = 17  # 0 others, 1-16 obj, 17 free
fix_void = num_cls == 19
img_norm_cfg = None

occ_size = [200, 200, 16]
voxel_out_indices = (0, 1, 2)
voxel_out_channel = 256
voxel_channels = [64, 64 * 2, 64 * 4]
gau_channel=1

model = dict(
    type='FBOCC',
    use_depth_supervision=True,
    fix_void=fix_void,
    do_history=do_history,
    history_cat_num=history_cat_num,
    single_bev_num_channels=numC_Trans,
    readd=True,
    img_backbone=dict(
        type='InternV2Impl16',
        with_cp=True,
        cp_level=3,
        deform_points=9,
        embed_dim=160,
        depths=[6, 6, 32, 6],
        num_heads=[10, 20, 40, 40],
        deform_padding=True,
        dilation_rates=[1],
        kernel_size=3,
        mlp_ratio=4.0,
        drop_path_rate=0.2,
        layer_scale=1e-5,
        op_types=['D', 'D', 'D', 'D'],
        out_indices=(2, 3),
        use_hw_scaler=1.71,
        init_cfg=dict(type='Pretrained', checkpoint="./ckpts/a-13-a_ep12.pth")
    ),
    img_neck=dict(
        type='CustomFPN',
        in_channels=[640, 1280],
        out_channels=_dim_,
        num_outs=1,
        start_level=0,
        with_cp=use_checkpoint,
        out_ids=[0]),
    depth_net=dict(
        type='CM_DepthNet',  # camera-aware depth net
        in_channels=_dim_,
        context_channels=numC_Trans,
        downsample=16,
        grid_config=grid_config,
        depth_channels=depth_categories,
        with_cp=use_checkpoint,
        loss_depth_weight=1.,
        use_dcn=False,
    ),
    forward_projection=dict(
        type='LSSViewTransformerFunction3D',
        grid_config=grid_config,
        input_size=data_config['input_size'],
        downsample=16),
    frpn=None,
    backward_projection=dict(
        type='BackwardProjection',
        bev_h=bev_h_,
        bev_w=bev_w_,
        in_channels=numC_Trans,
        out_channels=numC_Trans,
        pc_range=point_cloud_range,
        transformer=dict(
            type='BEVFormer',
            use_cams_embeds=False,
            embed_dims=numC_Trans,
            encoder=dict(
                type='bevformer_encoder',
                num_layers=1,
                pc_range=point_cloud_range,
                grid_config=grid_config_bevformer,
                data_config=data_config,
                return_intermediate=False,
                transformerlayers=dict(
                    type='BEVFormerEncoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiScaleDeformableAttention',
                            embed_dims=numC_Trans,
                            dropout=0.0,
                            num_levels=1),
                        dict(
                            type='DA_SpatialCrossAttention',
                            pc_range=point_cloud_range,
                            dbound=[2.0, 42.0, 0.5],
                            dropout=0.0,
                            deformable_attention=dict(
                                type='DA_MSDeformableAttention',
                                embed_dims=numC_Trans,
                                num_points=8,
                                num_levels=_num_levels_),
                            embed_dims=numC_Trans,
                        )
                    ],
                    ffn_cfgs=dict(
                        type='FFN',
                        embed_dims=numC_Trans,
                        feedforward_channels=_ffn_dim_,
                        ffn_drop=0.0,
                        act_cfg=dict(type='ReLU', inplace=True), ),
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.0,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm'))),
            # operation_order=('cross_attn', 'norm', 'ffn', 'norm'))),
            # operation_order=('cross_attn', 'norm'))),
        ),
        positional_encoding=dict(
            type='CustormLearnedPositionalEncoding',
            num_feats=_pos_dim_,
            row_num_embed=bev_h_,
            col_num_embed=bev_w_,
        ),
    ),
    img_bev_encoder_backbone=dict(
        type='CustomResNet3D',
        depth=18,
        with_cp=use_checkpoint,
        block_strides=[1, 2, 2],
        n_input_channels=numC_Trans,
        block_inplanes=voxel_channels,
        out_indices=voxel_out_indices,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
    ),
    img_bev_encoder_neck=dict(
        type='FPN3D',
        with_cp=use_checkpoint,
        in_channels=voxel_channels,
        out_channels=voxel_out_channel,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
    ),
    occupancy_head=dict(
        type='OccFlowHead',
        with_cp=use_checkpoint,
        use_focal_loss=True,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        soft_weights=True,
        final_occ_size=occ_size,
        empty_idx=empty_idx,
        num_level=len(voxel_out_indices),
        in_channels=[voxel_out_channel] * len(voxel_out_indices),
        out_channel=num_cls,
        point_cloud_range=point_cloud_range,
        use_flow_weight=True,
        use_3d_loss=True,
        loss_weight_cfg=dict(
            loss_voxel_ce_weight=1.0,
            loss_voxel_sem_scal_weight=1.0,
            loss_voxel_geo_scal_weight=1.0,
            loss_voxel_lovasz_weight=1.0,
        ),
        gaussian_head=dict(
                type='GaussianHead',
                loss_sem=dict(type='CrossEntropyLoss',
                            use_sigmoid=False,
                            loss_weight=0.5),
                render_future=[True, False],
                use_depth_loss=True,
                gaussian_cfg=dict(
                in_channels=gau_channel,
                max_size=0.12),
                loss_weight_cfg=dict(
                    loss_sem_weight=1.0,
                    loss_depth_weight=1.0,
                        ),),
    ),
    pts_bbox_head=None)

# Data
dataset_type = 'NuScenesOccDataset'
data_root = 'data/nuscenes/'
file_client_args = dict(backend='disk')
occupancy_path = 'data/nuscenes/gts'
gaussian_label_path = 'data/nuscenes/openocc_gt2d' # not use

train_pipeline = [
    dict(
        type='PrepareImageInputs',
        is_train=True,
        data_config=data_config),
    dict(
        type='LoadAnnotationsBEVDepthOcc',
        bda_aug_conf=bda_aug_conf,
        classes=class_names),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(type='PointToMultiViewDepth', downsample=1, grid_config=grid_config),
    dict(type='LoadOccupancy', ignore_nonvisible=False, fix_void=fix_void, occupancy_path=occupancy_path, sample=True,
         visible_mask_path='data/nuscenes/openocc_v2_ray_mask'),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='Collect3D', keys=['img_inputs', 'gt_occupancy', 'gt_depth', 'flow',
                                'gaussian_labels'])
]

test_pipeline = [
    dict(
        type='CustomDistMultiScaleFlipAug3D',
        tta=False,
        transforms=[
            dict(type='PrepareImageInputs', data_config=data_config),
            dict(
                type='LoadAnnotationsBEVDepthOcc',
                bda_aug_conf=bda_aug_conf,
                classes=class_names,
                is_train=False),
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=5,
                use_dim=5,
                file_client_args=file_client_args),
            dict(type='LoadOccupancy', occupancy_path=occupancy_path),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['img_inputs', 'gt_occupancy'])
        ]
    )
]

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

share_data_config = dict(
    type=dataset_type,
    classes=class_names,
    load_interval=1,
    img_info_prototype='bevdet',
    occupancy_path=occupancy_path,
    use_sequence_group_flag=True,
    gaussian_label_path=gaussian_label_path,
    save_sample=False

)

test_data_config = dict(
    pipeline=test_pipeline,
    sequences_split_num=test_sequences_split_num,
    ann_file=data_root + 'bevdetv2-nuscenes_infos_val.pkl')

data = dict(
    samples_per_gpu=samples_per_gpu,
    workers_per_gpu=6,
    test_dataloader=dict(runner_type='IterBasedRunnerEval'),
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'bevdetv2-nuscenes_infos_train.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        test_mode=False,
        img_info_prototype='bevdet',
        sequences_split_num=train_sequences_split_num,
        use_sequence_group_flag=True,
        gaussian_label_path=gaussian_label_path,
        aux_frames=[-2, -1, 1, 2, 3]),
    val=test_data_config,
    test=test_data_config)

for key in ['val', 'test']:
    data[key].update(share_data_config)

# Optimizer
lr = 2e-4
# optimizer = dict(type='AdamW', lr=lr, weight_decay=1e-2)
optimizer = dict(
    type='AdamW',
    lr=lr,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
        }),
    weight_decay=0.01)

optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=0.001,
    step=[num_iters_per_epoch * num_epochs, ])
runner = dict(type='IterBasedRunner', max_iters=num_epochs * num_iters_per_epoch)
checkpoint_config = dict(
    interval=checkpoint_epoch_interval * num_iters_per_epoch)
evaluation = dict(
    interval=num_epochs * num_iters_per_epoch, pipeline=test_pipeline)

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
    ])
custom_hooks = [
    dict(
        type='MEGVIIEMAHook',
        init_updates=10560,
        priority='NORMAL',
        interval=2 * num_iters_per_epoch,
    ),
    dict(
        type='SequentialControlHook',
        temporal_start_iter=num_iters_per_epoch * 2,
    ),
]
load_from = './ckpts/a-13-a_ep12.pth'
fp16 = dict(loss_scale='dynamic')