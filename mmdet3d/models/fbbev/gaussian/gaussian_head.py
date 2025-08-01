
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from mmcv.cnn.bricks.conv_module import ConvModule
from torch import Tensor 
from typing import NamedTuple
import os
from mmdet.models.builder import build_loss
from mmdet.models.losses import CrossEntropyLoss
from .ops import trunc_exp, gen_grid, get_device, Camera, silog_loss, normalize_coords, RangeNormalizedL1Loss, transform_c2w
from mmdet.models.losses import l1_loss
from einops import rearrange
import gsplat

class GaussianModel(NamedTuple):
    xyz: Tensor
    opacity: Tensor
    rotation: Tensor
    scaling: Tensor
    shs: Tensor

class Feats2Gau(nn.Module):
    def __init__(self,
             gaussian_cfg=None,
             ):
        super().__init__()

        self.cfg = gaussian_cfg
        self.max_size = self.cfg.max_size
        self.register_buffer('opacity', torch.tensor([[1.]]).float())
        self.register_buffer('rotation', torch.tensor([[1., 0., 0., 0.]]).float())
        self.register_buffer('scaling', torch.ones(1, 3).float() * self.max_size)

    def forward(self, pts, semantics, flows, current=True, is_gt=False):
        ret = {}
        
        if current:
            ret['xyz'] = pts
        else:
            ret['xyz'] = torch.cat(((pts[..., :2] + flows), pts[..., -1][..., None]), dim=-1)
        
        if is_gt:
            ret['opacity'] = self.opacity.repeat(pts.size(0), 1)
            ret['shs'] = torch.cat((semantics, flows), dim=1)
        else:
            ret['opacity'] = torch.sigmoid(-semantics[..., -1])[..., None]
            ret['shs'] = torch.cat((semantics[..., :-1], flows), dim=1)

        ret['scaling'] = self.scaling.repeat(pts.size(0), 1)
        ret['rotation'] = self.rotation.repeat(pts.size(0), 1)

        return GaussianModel(**ret)


from mmdet.models import HEADS
@HEADS.register_module()
class GaussianHead(nn.Module):
    def __init__(self,
                 point_cloud_range=[-40, -40, -1, 40, 40, 5.4],
                 voxel_size=0.4,
                 resolution=(200, 200, 16),
                 out_dim=32,
                 num_classes=16,
                 loss_sem=None,
                 use_depth_loss=True,
                 ds_rate=4,
                 use_all_voxel=False,
                 loss_weight_cfg=None,
                 gaussian_cfg=None,
                 num_points_each_grid=1,
                 random_choice_ratio=1.0,
                 render_future=[True],
                 ):
        super().__init__()

        self.num_classes = num_classes
        X, Y, Z = resolution

        grid = gen_grid(Z, Y, X, point_cloud_range, voxel_size, num=num_points_each_grid).float()
        norm_grid = normalize_coords(grid, pc_range=point_cloud_range)

        self.register_buffer('query_points', grid) # (H, W, D, 3)
        self.register_buffer('normalized_points', norm_grid)  # (H, W, D, 3)
        #self.register_buffer('centers', gen_grid(Z, Y, X, point_cloud_range, voxel_size, num=1).float())  # (H, W, D, 3)

        self.feat2gau = Feats2Gau(gaussian_cfg)

        self.out_dim = out_dim
        self.device = get_device()
        self.loss_sem = build_loss(loss_sem)
        self.loss_corr = RangeNormalizedL1Loss()

        self.use_depth_loss = use_depth_loss
        #self.loss_depth = silog_loss()
        self.loss_depth = l1_loss
        self.loss_flow = l1_loss
        self.loss_sem_l1 = l1_loss

        self.ds_rate = ds_rate
        self.num_points_each_grid = num_points_each_grid

        self.use_all_voxel = use_all_voxel

        self.loss_sem_weight = loss_weight_cfg.get('loss_sem_weight', 1.0)
        self.loss_depth_weight = loss_weight_cfg.get('loss_depth_weight', 1.0)
        self.random_choice_ratio = random_choice_ratio

        self.znear = 0.1
        self.zfar = 100.0

        self.render_future = render_future
         

    def sample_feature(self, semantics, flows, gt_3d, gt_flow, sample_mask=None):
        if sample_mask.sum() == 0:
            no_empty_mask_pred = gt_3d != 255
            no_empty_mask_gt = gt_3d < self.num_classes
        else:
            no_empty_mask_pred = (gt_3d != 255) & sample_mask
            no_empty_mask_gt = (gt_3d < self.num_classes) & sample_mask

        pts = self.query_points[no_empty_mask_pred].reshape(-1, 3)
        semantics = semantics.permute(1, 2, 3, 0)[no_empty_mask_pred]
        flows = flows.permute(1, 2, 3, 0)[no_empty_mask_pred]

        sem_gt = F.one_hot(gt_3d[no_empty_mask_gt].long(), num_classes=self.num_classes).float()
        pts_gt = self.query_points[no_empty_mask_gt].reshape(-1, 3)
        flow_gt = gt_flow[no_empty_mask_gt]
         
        return pts, semantics, flows, sem_gt, pts_gt, flow_gt


    def forward_single_batch(self, semantics, flows, c2ws, intrinsics, gt_3d=None, gt_flow=None, sample_mask=None, flow_loss_weight=0.0):

        pts, semantics, flows, sem_gt, pts_gt, flow_gt = self.sample_feature(semantics, flows, gt_3d, gt_flow, sample_mask)

        sem_preds, flow_preds, depth_preds = [], [], []
        sem_gts,   flow_gts,   depth_gts   = [], [], []
        state_list = [True] if flow_loss_weight == 0 else self.render_future
        
        for state in state_list:
            gs = self.feat2gau(pts, semantics, flows, current=state)
            gs_gt = self.feat2gau(pts_gt, sem_gt, flow_gt, current=state, is_gt=True)

            sem_p, flow_p, depth_p = self.forward_rendering(gs, c2ws, intrinsics)
            sem_g, flow_g, depth_g = self.forward_rendering(gs_gt, c2ws, intrinsics)

            sem_preds.append(sem_p)
            flow_preds.append(flow_p)
            depth_preds.append(depth_p)

            sem_gts.append(sem_g)
            flow_gts.append(flow_g)
            depth_gts.append(depth_g)

        sem_preds = torch.stack(sem_preds)
        flow_preds = torch.stack(flow_preds)
        depth_preds = torch.stack(depth_preds)

        sem_gts = torch.stack(sem_gts)
        flow_gts = torch.stack(flow_gts)
        depth_gts = torch.stack(depth_gts)

        return sem_preds, flow_preds, depth_preds, sem_gts, flow_gts, depth_gts
    
    def forward_rendering(self, gs, c2ws, intrinsics):
        c2ws = rearrange(c2ws, 't v h w -> (t v) h w') # [V, 4, 4]
        w2cs = transform_c2w(c2w=c2ws) # [V, 4, 4]
        cks = rearrange(intrinsics, 't v h w -> (t v) h w') # [V, 3, 3]
        
        means3D = gs.xyz
        opacity = gs.opacity
     
        scales = gs.scaling
        rotations = gs.rotation
        sem_flow_logits = gs.shs
        scale_modifier = 1.0 
        scales_modified = scales * scale_modifier

        sem_flow_depth, _, _ = gsplat.rasterization(
                means=means3D,
                quats=rotations,
                scales=scales_modified,
                opacities=opacity.squeeze(),
                colors=sem_flow_logits,
                viewmats=w2cs,
                Ks=cks,
                width=1600 // self.ds_rate,
                height=900 // self.ds_rate,
                near_plane=self.znear,
                far_plane=self.zfar,
                # backgrounds=backgrounds,
                render_mode='RGB+D',
            )
        
        sem = sem_flow_depth[..., :self.num_classes]
        flow = sem_flow_depth[..., self.num_classes: self.num_classes + 2]
        depth = sem_flow_depth[..., self.num_classes + 2:]
    
        return sem, flow, depth
        

    def loss_2d(self, render_result, gaussian_label, flow_loss_weight=None, move_idx=None):
        loss_ = dict()

        semantics, flow, depth = render_result['semantics'], render_result['flows'], render_result['depths']
        semantics_gt, flow_gt, depth_gt = render_result['semantics_gt'], render_result['flows_gt'], render_result['depths_gt']
        
        semantics = semantics.reshape(-1, self.num_classes)
        depth = depth.reshape(-1)
        flow = flow.reshape(-1, 2)

        semantics_gt = semantics_gt.reshape(-1, self.num_classes).detach()
        depth_gt = depth_gt.reshape(-1).detach()
        flow_gt = flow_gt.reshape(-1, 2).detach()
        fore_mask = (depth_gt > 0.1) 
        
        loss_sem = self.loss_sem(semantics[fore_mask], semantics_gt[fore_mask]) * self.loss_sem_weight

        if self.use_depth_loss:
            #label_depths[label_depths > 80] = 0.
            loss_depth = self.loss_depth(depth, depth_gt)
            loss_['loss_gau_depth'] = loss_depth * self.loss_depth_weight

            norm_weight = torch.norm(flow_gt, dim=-1)[..., None] + 0.1
            avg_factor = norm_weight.sum()
            loss_flow = self.loss_flow(flow, flow_gt,weight=norm_weight, avg_factor=avg_factor)
            loss_['loss_gau_flow'] = loss_flow * flow_loss_weight

        loss_['loss_gau_sem'] = loss_sem
         
        return loss_

    def loss_3d(self, semantics, gt_3d):
        loss_ = dict()
        empty_mask = gt_3d == self.num_classes
        opacity = torch.sigmoid( - semantics.permute(0, 2, 3, 4, 1)[..., -1])

        loss_op = opacity[empty_mask].sum() / ((empty_mask).sum())

        loss_['loss_op'] = loss_op

        return loss_

    def forward(self, semantics, flows, gaussian_label, gt_3d=None, gt_flow=None, flow_loss_weight=-1.0):
        c2w, intrinsic, sample_mask = gaussian_label

        batch_size, frame_num = c2w.shape[0], c2w.shape[1]

        semantics_list = []
        flow_list = []
        depth_list = []

        semantics_gt_list = []
        flow_gt_list = []
        depth_gt_list = []
   
        for b in range(batch_size):

            sem_2d, flow_2d, depth_2d, sem_2d_gt, flow_2d_gt, depth_2d_gt = self.forward_single_batch(semantics[b],
                    flows[b], c2w[b], intrinsic[b], gt_3d=gt_3d[b], gt_flow=gt_flow[b], 
                    sample_mask=sample_mask[b], flow_loss_weight=flow_loss_weight)

            semantics_list.append(sem_2d)
            flow_list.append(flow_2d)
            depth_list.append(depth_2d)

            semantics_gt_list.append(sem_2d_gt)
            flow_gt_list.append(flow_2d_gt)
            depth_gt_list.append(depth_2d_gt)

        sem_pred = torch.stack(semantics_list)
        flow_pred = torch.stack(flow_list)
        depth_pred = torch.stack(depth_list)

        sem_gt = torch.stack(semantics_gt_list)
        flow_gt = torch.stack(flow_gt_list)
        depth_gt = torch.stack(depth_gt_list)

        render_results = {}
        render_results['semantics'] = sem_pred
        render_results['flows'] = flow_pred
        render_results['depths'] = depth_pred

        render_results['semantics_gt'] = sem_gt
        render_results['flows_gt'] = flow_gt
        render_results['depths_gt'] = depth_gt


        return [render_results]