# Copyright (c) OpenMMLab. All rights reserved.
# from mmdet.models.backbones import SSDVGG, HRNet, ResNet, ResNetV1d, ResNeXt
from mmdet.models.backbones import ResNet
# from .dgcnn import DGCNNBackbone
# from .dla import DLANet
# from .mink_resnet import MinkResNet
# from .multi_backbone import MultiBackbone
# from .nostem_regnet import NoStemRegNet
# from .pointnet2_sa_msg import PointNet2SAMSG
# from .pointnet2_sa_ssg import PointNet2SASSG
from .resnet import CustomResNet
# from .second import SECOND
# from .convnext import ConvNeXt
# from .vovnet import VoVNetCP
# from .swin import SwinTransformer
from .internv2_impl16 import InternV2Impl16
from .eva_vit import EVAViT
# __all__ = [
#     'ResNet', 'ResNetV1d', 'ResNeXt', 'SSDVGG', 'HRNet', 'NoStemRegNet',
#     'SECOND', 'DGCNNBackbone', 'PointNet2SASSG', 'PointNet2SAMSG',
#     'MultiBackbone', 'DLANet', 'MinkResNet', 'CustomResNet', 'EVAViT'
# ]

__all__ = [
    'ResNet', 'EVAViT'
]
