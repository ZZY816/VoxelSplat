import torch
import torch.nn.functional as F
import torch.nn as nn
from mmcv.runner import force_fp32
import os
from mmdet3d.ops.bev_pool_v2.bev_pool import TRTBEVPoolv2
from mmdet.models import DETECTORS
from mmdet3d.models import builder
# from mmdet3d.models.detectors import CenterPoint
from mmdet3d.models.builder import build_head, build_neck
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from collections import defaultdict
from mmcv.runner import get_dist_info
from mmdet.core import reduce_mean
import mmcv
from mmdet3d.datasets.utils import nuscenes_get_rt_matrix
from mmdet.models.detectors import BaseDetector
from mmcv.runner import auto_fp16

def generate_forward_transformation_matrix(bda, img_meta_dict=None):
    b = bda.size(0)
    hom_res = torch.eye(4)[None].repeat(b, 1, 1).to(bda.device)
    for i in range(b):
        hom_res[i, :3, :3] = bda[i]
    return hom_res


@DETECTORS.register_module()
class FBOCC(BaseDetector):

    def __init__(self, 
                 img_backbone=None,
                 img_neck=None,
                 # BEVDet components
                 forward_projection=None,
                 img_bev_encoder_backbone=None,
                 img_bev_encoder_neck=None,
                 # BEVFormer components
                 backward_projection=None,
                 # FB-BEV components
                 frpn=None,
                 # depth_net
                 depth_net=None,
                 # occupancy head
                 occupancy_head=None,
                 occupancy_flow_head=None,
                 pts_bbox_head=None,
                 # other settings.
                 use_depth_supervision=False,
                 readd=False,
                 fix_void=False,
                 occupancy_save_path=None,
                 do_history=True,
                 interpolation_mode='bilinear',
                 history_cat_num=16,
                 history_cat_conv_out_channels=None,
                 single_bev_num_channels=80,
                 flow_loss_weight=0.5,
                 train_cfg=None,
                 test_cfg=None,
                  **kwargs):
        super(FBOCC, self).__init__(**kwargs)
        self.fix_void = fix_void
        
        if img_backbone:
            self.img_backbone = builder.build_backbone(img_backbone)
        if img_neck is not None:
            self.img_neck = builder.build_neck(img_neck)
      
        # BEVDet init
        self.forward_projection = builder.build_neck(forward_projection) if forward_projection else None
        self.img_bev_encoder_backbone = builder.build_backbone(img_bev_encoder_backbone) if img_bev_encoder_backbone else None
        self.img_bev_encoder_neck = builder.build_neck(img_bev_encoder_neck) if img_bev_encoder_neck else None

        # BEVFormer init
        self.backward_projection = builder.build_head(backward_projection) if backward_projection else None
    
        # FB-BEV init
        if not self.forward_projection: assert not frpn, 'frpn relies on LSS'
        self.frpn = builder.build_head(frpn) if frpn else None

        # Depth Net
        self.depth_net = builder.build_head(depth_net) if depth_net else None

        # Occupancy Head
        self.occupancy_head = builder.build_head(occupancy_head) if occupancy_head else None
        self.occupancy_flow_head = builder.build_head(occupancy_flow_head) if occupancy_flow_head else None

        
        self.readd = readd # fuse voxel features and bev features
        
        self.use_depth_supervision = use_depth_supervision
        
        self.occupancy_save_path = occupancy_save_path # for saving data\for submitting to test server

        # Deal with history
        self.single_bev_num_channels = single_bev_num_channels
        self.do_history = do_history
        self.interpolation_mode = interpolation_mode
        self.history_cat_num = history_cat_num
        self.history_cam_sweep_freq = 0.5 # seconds between each frame
        history_cat_conv_out_channels = (history_cat_conv_out_channels 
                                         if history_cat_conv_out_channels is not None 
                                         else self.single_bev_num_channels)
        ## Embed each sample with its relative temporal offset with current timestep
        conv = nn.Conv2d if self.forward_projection.nx[-1] == 1 else nn.Conv3d
        self.history_keyframe_time_conv = nn.Sequential(
             conv(self.single_bev_num_channels + 1,
                     self.single_bev_num_channels,
                     kernel_size=1,
                     padding=0,
                     stride=1),
             nn.SyncBatchNorm(self.single_bev_num_channels),
             nn.ReLU(inplace=True))
        ## Then concatenate and send them through an MLP.
        self.history_keyframe_cat_conv = nn.Sequential(
            conv(self.single_bev_num_channels * (self.history_cat_num + 1),
                    history_cat_conv_out_channels,
                    kernel_size=1,
                    padding=0,
                    stride=1),
            nn.SyncBatchNorm(history_cat_conv_out_channels),
            nn.ReLU(inplace=True))
        self.history_sweep_time = None
        self.history_bev = None
        self.history_bev_before_encoder = None
        self.history_seq_ids = None
        self.history_forward_augs = None
        self.count = 0
        self.flow_loss_weight = flow_loss_weight

    @property
    def with_img_neck(self):
        """bool: Whether the detector has a neck in image branch."""
        return hasattr(self, 'img_neck') and self.img_neck is not None

    def with_specific_component(self, component_name):
        """Whether the model owns a specific component"""
        return getattr(self, component_name, None) is not None
    
    def image_encoder(self, img):
        imgs = img
        B, N, C, imH, imW = imgs.shape
        imgs = imgs.view(B * N, C, imH, imW)
      
        x = self.img_backbone(imgs)

        if self.with_img_neck:
            x = self.img_neck(x)
            if type(x) in [list, tuple]:
                x = x[0]
        else:
            x = x[0]
        _, output_dim, ouput_H, output_W = x.shape
        x = x.view(B, N, output_dim, ouput_H, output_W)

        return x

    @force_fp32()
    def bev_encoder(self, x):
        if self.with_specific_component('img_bev_encoder_backbone'):
            x = self.img_bev_encoder_backbone(x)
        
        if self.with_specific_component('img_bev_encoder_neck'):
            x = self.img_bev_encoder_neck(x)
        
        if type(x) not in [list, tuple]:
             x = [x]

        return x
        
    @force_fp32()
    def fuse_history(self, curr_bev, img_metas, bda): # align features with 3d shift

        voxel_feat = True  if len(curr_bev.shape) == 5 else False
        if voxel_feat:
            curr_bev = curr_bev.permute(0, 1, 4, 2, 3) # n, c, z, h, w
        
        seq_ids = torch.LongTensor([
            single_img_metas['sequence_group_idx'] 
            for single_img_metas in img_metas]).to(curr_bev.device)
        start_of_sequence = torch.BoolTensor([
            single_img_metas['start_of_sequence'] 
            for single_img_metas in img_metas]).to(curr_bev.device)
        forward_augs = generate_forward_transformation_matrix(bda)

        curr_to_prev_ego_rt = torch.stack([
            single_img_metas['curr_to_prev_ego_rt']
            for single_img_metas in img_metas]).to(curr_bev)

        ## Deal with first batch
        if self.history_bev is None:
            self.history_bev = curr_bev.clone()
            self.history_seq_ids = seq_ids.clone()
            self.history_forward_augs = forward_augs.clone()

            # Repeat the first frame feature to be history
            if voxel_feat:
                self.history_bev = curr_bev.repeat(1, self.history_cat_num, 1, 1, 1) 
            else:
                self.history_bev = curr_bev.repeat(1, self.history_cat_num, 1, 1)
            # All 0s, representing current timestep.
            self.history_sweep_time = curr_bev.new_zeros(curr_bev.shape[0], self.history_cat_num)


        self.history_bev = self.history_bev.detach()

        assert self.history_bev.dtype == torch.float32

        ## Deal with the new sequences
        # First, sanity check. For every non-start of sequence, history id and seq id should be same.

        assert (self.history_seq_ids != seq_ids)[~start_of_sequence].sum() == 0, \
                "{}, {}, {}".format(self.history_seq_ids, seq_ids, start_of_sequence)

        ## Replace all the new sequences' positions in history with the curr_bev information
        self.history_sweep_time += 1 # new timestep, everything in history gets pushed back one.
        if start_of_sequence.sum()>0:
            if voxel_feat:    
                self.history_bev[start_of_sequence] = curr_bev[start_of_sequence].repeat(1, self.history_cat_num, 1, 1, 1)
            else:
                self.history_bev[start_of_sequence] = curr_bev[start_of_sequence].repeat(1, self.history_cat_num, 1, 1)
            
            self.history_sweep_time[start_of_sequence] = 0 # zero the new sequence timestep starts
            self.history_seq_ids[start_of_sequence] = seq_ids[start_of_sequence]
            self.history_forward_augs[start_of_sequence] = forward_augs[start_of_sequence]


        ## Get grid idxs & grid2bev first.
        if voxel_feat:
            n, c_, z, h, w = curr_bev.shape

        # Generate grid
        xs = torch.linspace(0, w - 1, w, dtype=curr_bev.dtype, device=curr_bev.device).view(1, w, 1).expand(h, w, z)
        ys = torch.linspace(0, h - 1, h, dtype=curr_bev.dtype, device=curr_bev.device).view(h, 1, 1).expand(h, w, z)
        zs = torch.linspace(0, z - 1, z, dtype=curr_bev.dtype, device=curr_bev.device).view(1, 1, z).expand(h, w, z)
        grid = torch.stack(
            (xs, ys, zs, torch.ones_like(xs)), -1).view(1, h, w, z, 4).expand(n, h, w, z, 4).view(n, h, w, z, 4, 1)

        # This converts BEV indices to meters
        # IMPORTANT: the feat2bev[0, 3] is changed from feat2bev[0, 2] because previous was 2D rotation
        # which has 2-th index as the hom index. Now, with 3D hom, 3-th is hom
        feat2bev = torch.zeros((4,4),dtype=grid.dtype).to(grid)
        feat2bev[0, 0] = self.forward_projection.dx[0]
        feat2bev[1, 1] = self.forward_projection.dx[1]
        feat2bev[2, 2] = self.forward_projection.dx[2]
        feat2bev[0, 3] = self.forward_projection.bx[0] - self.forward_projection.dx[0] / 2.
        feat2bev[1, 3] = self.forward_projection.bx[1] - self.forward_projection.dx[1] / 2.
        feat2bev[2, 3] = self.forward_projection.bx[2] - self.forward_projection.dx[2] / 2.
        # feat2bev[2, 2] = 1
        feat2bev[3, 3] = 1
        feat2bev = feat2bev.view(1,4,4)

        ## Get flow for grid sampling.
        # The flow is as follows. Starting from grid locations in curr bev, transform to BEV XY11,
        # backward of current augmentations, curr lidar to prev lidar, forward of previous augmentations,
        # transform to previous grid locations.
        rt_flow = (torch.inverse(feat2bev) @ self.history_forward_augs @ curr_to_prev_ego_rt
                   @ torch.inverse(forward_augs) @ feat2bev)

        grid = rt_flow.view(n, 1, 1, 1, 4, 4) @ grid

        # normalize and sample
        normalize_factor = torch.tensor([w - 1.0, h - 1.0, z - 1.0], dtype=curr_bev.dtype, device=curr_bev.device)
        grid = grid[:,:,:,:, :3,0] / normalize_factor.view(1, 1, 1, 1, 3) * 2.0 - 1.0
        

        tmp_bev = self.history_bev
        if voxel_feat: 
            n, mc, z, h, w = tmp_bev.shape
            tmp_bev = tmp_bev.reshape(n, mc, z, h, w)
        sampled_history_bev = F.grid_sample(tmp_bev, grid.to(curr_bev.dtype).permute(0, 3, 1, 2, 4), align_corners=True, mode=self.interpolation_mode)

        ## Update history
        # Add in current frame to features & timestep
        self.history_sweep_time = torch.cat(
            [self.history_sweep_time.new_zeros(self.history_sweep_time.shape[0], 1), self.history_sweep_time],
            dim=1) # B x (1 + T)

        if voxel_feat:
            sampled_history_bev = sampled_history_bev.reshape(n, mc, z, h, w)
            curr_bev = curr_bev.reshape(n, c_, z, h, w)
        feats_cat = torch.cat([curr_bev, sampled_history_bev], dim=1) # B x (1 + T) * 80 x H x W or B x (1 + T) * 80 xZ x H x W 

        # Reshape and concatenate features and timestep
        feats_to_return = feats_cat.reshape(
                feats_cat.shape[0], self.history_cat_num + 1, self.single_bev_num_channels, *feats_cat.shape[2:]) # B x (1 + T) x 80 x H x W
        if voxel_feat:
            feats_to_return = torch.cat(
            [feats_to_return, self.history_sweep_time[:, :, None, None, None, None].repeat(
                1, 1, 1, *feats_to_return.shape[3:]) * self.history_cam_sweep_freq
            ], dim=2) # B x (1 + T) x 81 x Z x H x W
        else:
            feats_to_return = torch.cat(
            [feats_to_return, self.history_sweep_time[:, :, None, None, None].repeat(
                1, 1, 1, feats_to_return.shape[3], feats_to_return.shape[4]) * self.history_cam_sweep_freq
            ], dim=2) # B x (1 + T) x 81 x H x W

        # Time conv
        feats_to_return = self.history_keyframe_time_conv(
            feats_to_return.reshape(-1, *feats_to_return.shape[2:])).reshape(
                feats_to_return.shape[0], feats_to_return.shape[1], -1, *feats_to_return.shape[3:]) # B x (1 + T) x 80 xZ x H x W

        # Cat keyframes & conv
        feats_to_return = self.history_keyframe_cat_conv(
            feats_to_return.reshape(
                feats_to_return.shape[0], -1, *feats_to_return.shape[3:])) # B x C x H x W or B x C x Z x H x W
        
        self.history_bev = feats_cat[:, :-self.single_bev_num_channels, ...].detach().clone()
        self.history_sweep_time = self.history_sweep_time[:, :-1]
        self.history_forward_augs = forward_augs.clone()
        if voxel_feat:
            feats_to_return = feats_to_return.permute(0, 1, 3, 4, 2)
        if not self.do_history:
            self.history_bev = None
        return feats_to_return.clone()


    def extract_img_bev_feat(self, img, img_metas, **kwargs):
        """Extract features of images."""

        return_map = {}

        context = self.image_encoder(img[0])

        cam_params = img[1:7]
        if self.with_specific_component('depth_net'):
            mlp_input = self.depth_net.get_mlp_input(*cam_params)
            context, depth = self.depth_net(context, mlp_input)
            return_map['depth'] = depth
            return_map['context'] = context
        else:
            context=None
            depth=None

        if self.with_specific_component('forward_projection'):
            bev_feat = self.forward_projection(cam_params, context, depth, **kwargs)
            return_map['cam_params'] = cam_params
        else:
            bev_feat = None

        if self.with_specific_component('frpn'): # not used in FB-OCC
            bev_mask_logit = self.frpn(bev_feat)
            bev_mask = bev_mask_logit.sigmoid() > self.frpn.mask_thre
            
            if bev_mask.requires_grad: # during training phase
                gt_bev_mask = kwargs['gt_bev_mask'].to(torch.bool)
                bev_mask = gt_bev_mask | bev_mask
            return_map['bev_mask_logit'] = bev_mask_logit    
        else:
            bev_mask = None

        if self.with_specific_component('backward_projection'):

            bev_feat_refined = self.backward_projection([context],
                                        img_metas,
                                        lss_bev=bev_feat.mean(-1),
                                        cam_params=cam_params,
                                        bev_mask=bev_mask,
                                        gt_bboxes_3d=None, # debug
                                        pred_img_depth=depth)  
                                        
            if self.readd:
                bev_feat = bev_feat_refined[..., None] + bev_feat
            else:
                bev_feat = bev_feat_refined

        # Fuse History
        bev_feat = self.fuse_history(bev_feat, img_metas, img[6])

        bev_feat = self.bev_encoder(bev_feat)

        return_map['img_bev_feat'] = bev_feat

        return return_map

    def extract_lidar_bev_feat(self, pts, img_feats, img_metas):
        """Extract features of points."""

        voxels, num_points, coors = self.voxelize(pts)

        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors)
        batch_size = coors[-1, 0] + 1
        bev_feat = self.pts_middle_encoder(voxel_features, coors, batch_size)
        bev_feat = self.bev_encoder(bev_feat)
        return bev_feat

    def extract_feat(self, points, img, img_metas, **kwargs):
        """Extract features from images and points."""
        results={}
        if img is not None and self.with_specific_component('image_encoder'):
            results.update(self.extract_img_bev_feat(img, img_metas, **kwargs))
        if points is not None and self.with_specific_component('pts_voxel_encoder'):
            results['lidar_bev_feat'] = self.extract_lidar_bev_feat(points, img, img_metas)

        return results

    @auto_fp16(apply_to=('img', 'points'))
    def forward(self, return_loss=True, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.

        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)


    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      gt_occupancy_flow=None,
                      # gaussian_labels=None,
                      **kwargs):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """
        results= self.extract_feat(
            points, img=img_inputs, img_metas=img_metas, **kwargs)
        losses = dict()

        if self.with_specific_component('occupancy_head'):
            if self.do_history:
                flow_loss_weight = self.flow_loss_weight
            else:
                flow_loss_weight = 0.0
            gt_flow = kwargs['flow'] if 'flow' in kwargs else None
          
            if 'gaussian_labels' in kwargs:
                losses_occupancy = self.occupancy_head.forward_train(results['img_bev_feat'], results=results,
                                                                 gt_occupancy=kwargs['gt_occupancy'],
                                                                 gt_flow=gt_flow, gt_gaussian=kwargs['gaussian_labels'],
                                                                 flow_loss_weight=flow_loss_weight)
            else:
                losses_occupancy = self.occupancy_head.forward_train(results['img_bev_feat'], results=results,
                                                                     gt_occupancy=kwargs['gt_occupancy'],
                                                                     gt_flow=gt_flow,
                                                                     flow_loss_weight=flow_loss_weight)
                                                                     
            losses.update(losses_occupancy)

        if self.with_specific_component('occupancy_flow_head'):
            if self.do_history:
                flow_loss_weight = 1.0
            else:
                flow_loss_weight = 1.0
            losses_flow = self.occupancy_flow_head.forward_train(results['img_bev_feat'], results=results,
                                                                 gt_occ=kwargs['gt_occupancy'],
                                                                 gt_flow=kwargs['flow'], flow_loss_weight=flow_loss_weight)
            losses.update(losses_flow)


        if self.use_depth_supervision and self.with_specific_component('depth_net'):
            loss_depth = self.depth_net.get_depth_loss(kwargs['gt_depth'], results['depth'])
            losses.update(loss_depth)
       
        return losses

    def forward_test(self,
                     points=None,
                     img_metas=None,
                     img_inputs=None,
                     **kwargs):
        """
        Args:
            points (list[torch.Tensor]): the outer list indicates test-time
                augmentations and inner torch.Tensor should have a shape NxC,
                which contains all points in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch
            img (list[torch.Tensor], optional): the outer
                list indicates test-time augmentations and inner
                torch.Tensor should have a shape NxCxHxW, which contains
                all images in the batch. Defaults to None.
        """
        self.do_history = True
        if img_inputs is not None:
            for var, name in [(img_inputs, 'img_inputs'),
                          (img_metas, 'img_metas')]:
                if not isinstance(var, list) :
                    raise TypeError('{} must be a list, but got {}'.format(
                        name, type(var)))        
            num_augs = len(img_inputs)
            if num_augs != len(img_metas):
                raise ValueError(
                    'num of augmentations ({}) != num of image meta ({})'.format(
                        len(img_inputs), len(img_metas)))

             
            if num_augs==1 and not img_metas[0][0].get('tta_config', dict(dist_tta=False))['dist_tta']:
                return self.simple_test(points, img_metas[0], img_inputs[0],
                                    **kwargs)
            else:
                return self.aug_test(points, img_metas, img_inputs, **kwargs)
        
        elif points is not None:
            img_inputs = [img_inputs] if img_inputs is None else img_inputs
            points = [points] if points is None else points
            return self.simple_test(points[0], img_metas[0], img_inputs[0],
                                    **kwargs)
        
    def aug_test(self,points,
                    img_metas,
                    img_inputs=None,
                    visible_mask=[None],
                    **kwargs):
        """Test function without augmentaiton."""
        assert False
        return None

    def simple_test(self,
                    points,
                    img_metas,
                    img=None,
                    rescale=False,
                    visible_mask=[None],
                    return_raw_occ=False,
                    **kwargs):
        """Test function without augmentaiton."""
        results = self.extract_feat(
            points, img=img, img_metas=img_metas, **kwargs)

        bbox_list = [dict() for _ in range(len(img_metas))]
          
        bbox_pts = [None for _ in range(len(img_metas))]

        if self.with_specific_component('occupancy_flow_head'):
            pred_flow = self.occupancy_flow_head(results['img_bev_feat'], results=results, **kwargs)['output_flow'][0]

            pred_flow = pred_flow.permute(0, 2, 3, 4, 1)[0]
            # convert to CVPR2023 Format
            pred_flow = pred_flow.permute(3, 2, 0, 1)
            pred_flow = torch.flip(pred_flow, [2])
            pred_flow = torch.rot90(pred_flow, -1, [2, 3])
            pred_flow= pred_flow.permute(2, 3, 1, 0)
            pred_flow = pred_flow.cpu().numpy()
        else:
            pred_flow = None

        if self.with_specific_component('occupancy_head'):
            out = self.occupancy_head(results['img_bev_feat'], results=results, **kwargs)
            
            depth = None
            semantic = None

            pred_occupancy = out['output_voxels'][0].permute(0, 2, 3, 4, 1)[0] # bncdhw->bnwhdc  bncdhw->dhwc
            if 'output_flow' in out:
                pred_flow = out['output_flow'][0]
            else:
                pred_flow = None

            if self.fix_void:
                pred_occupancy = pred_occupancy[..., 1:]

            if False:
                opacity = torch.sigmoid(- pred_occupancy[..., -1]).cpu().numpy()
                pred_occupancy_category[opacity > 0.4] = pred_occupancy[..., :-1].argmax(-1)[opacity > 0.4]
                #     #pred_occupancy_category[opacity <= 0.5] = 16
            else:
                opacity = None
            pred_occupancy = pred_occupancy.softmax(-1)

            if pred_flow is not None:
                pred_flow = pred_flow.permute(0, 2, 3, 4, 1)[0]
                pred_flow = pred_flow.cpu().numpy()

            if return_raw_occ:
                pred_occupancy_category = pred_occupancy
            else:
                pred_occupancy_category = pred_occupancy.argmax(-1)

            # For test server
            if self.occupancy_save_path is not None:
                    scene_name = img_metas[0]['scene_name']
                    sample_token = img_metas[0]['sample_idx']
                    # mask_camera = visible_mask[0][0]
                    # masked_pred_occupancy = pred_occupancy[mask_camera].cpu().numpy()
                    save_pred_occupancy = pred_occupancy.argmax(-1).cpu().numpy()
                    save_path = os.path.join(self.occupancy_save_path, 'occupancy_pred', f'{sample_token}.npz')
                    np.savez_compressed(save_path, save_pred_occupancy.astype(np.uint8)) 

            pred_occupancy_category= pred_occupancy_category.cpu().numpy()


            if pred_flow is not None:
                move_num = (pred_occupancy_category <= 7).sum()
                if move_num > 0:
                    pred_flow = pred_flow[pred_occupancy_category <= 7]
                else:
                    pred_flow = 0

        else:
            pred_occupancy_category =  None
        
        if results.get('bev_mask_logit', None) is not None:
            pred_bev_mask = results['bev_mask_logit'].sigmoid() > 0.5
            iou = IOU(pred_bev_mask.reshape(1, -1), kwargs['gt_bev_mask'][0].reshape(1, -1)).cpu().numpy()
        else:
            iou = None

        assert len(img_metas) == 1
        for i, result_dict in enumerate(bbox_list):
            result_dict['pts_bbox'] = bbox_pts[i]
            result_dict['iou'] = iou
            result_dict['pred_occupancy'] = pred_occupancy_category
            result_dict['index'] = img_metas[0]['index']
            result_dict['pred_flow'] = pred_flow
            if opacity is not None:
                result_dict['pred_opacity'] = opacity
            # if (semantics_2d is not None) & (depths_2d is not None):
            #     result_dict['semantics_2d'] = semantics_2d
            #     result_dict['depths_2d'] = depths_2d



        return bbox_list


    def forward_dummy(self,
                      points=None,
                      img_metas=None,
                      img_inputs=None,
                      **kwargs):
        results = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas, **kwargs)
        assert self.with_pts_bbox
        outs = self.pts_bbox_head(results['img_bev_feat'])
        return outs
