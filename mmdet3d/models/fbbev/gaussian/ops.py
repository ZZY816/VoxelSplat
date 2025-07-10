from torch.cuda.amp import custom_bwd, custom_fwd
from torch.autograd import Function
import torch
import os
import numpy as np
import math
import torch.nn as nn
import itertools

class _TruncExp(Function):  # pylint: disable=abstract-method
    # Implementation from torch-ngp:
    # https://github.com/ashawkey/torch-ngp/blob/93b08a0d4ec1cc6e69d85df7f0acdfb99603b628/activation.py
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, x):  # pylint: disable=arguments-differ
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    @custom_bwd
    def backward(ctx, g):  # pylint: disable=arguments-differ
        x = ctx.saved_tensors[0]
        return g * torch.exp(torch.clamp(x, max=15))


trunc_exp = _TruncExp.apply


inverse_sigmoid = lambda x: np.log(x / (1 - x))

def transform_c2w(c2w):
    """
    Args:
        c2w: [v, 4, 4] tensor, camera-to-world matrices
    Returns:
        w2c: [v, 4, 4] tensor, world-to-camera matrices after flipping y and z
    """
    # 定义 y-z 交换矩阵

    flip_yz = torch.eye(4, device=c2w.device)
    flip_yz[1, 1] = -1
    flip_yz[2, 2] = -1
    c2w= torch.matmul(c2w, flip_yz)

    # 再求逆
    w2c = torch.linalg.inv(c2w)
    # w2c = w2c.permute(0, 2, 1)

    return w2c


def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def intrinsic_to_fov(intrinsic, w, h):
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    fov_x = 2 * torch.arctan2(w, 2 * fx)
    fov_y = 2 * torch.arctan2(h, 2 * fy)
    return fov_x, fov_y

def get_rank():
    # SLURM_PROCID can be set even if SLURM is not managing the multiprocessing,
    # therefore LOCAL_RANK needs to be checked first
    rank_keys = ("RANK", "LOCAL_RANK", "SLURM_PROCID", "JSM_NAMESPACE_RANK")
    for key in rank_keys:
        rank = os.environ.get(key)
        if rank is not None:
            return int(rank)
    return 0

def get_device():
    return torch.device(f"cuda:{get_rank()}")


class Camera:
    def __init__(self, w2c, intrinsic, FoVx, FoVy, height=900, width=1600, trans=np.array([0.0, 0.0, 0.0]), scale=1.0) -> None:
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.height = height
        self.width = width
        self.world_view_transform = w2c.transpose(0, 1)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).to(w2c.device)
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

    @staticmethod
    def from_c2w(c2w, intrinsic, height=900, width=1600):
        w2c = torch.inverse(c2w)
        FoVx, FoVy = intrinsic_to_fov(intrinsic, w=torch.tensor(width, device=w2c.device), h=torch.tensor(height, device=w2c.device))
        return Camera(w2c=w2c, intrinsic=intrinsic, FoVx=FoVx, FoVy=FoVy, height=height, width=width)

def gridcloud3d(B, Z, Y, X, device='cpu'):
    # we want to sample for each location in the grid
    grid_z, grid_y, grid_x = meshgrid3d(B, Z, Y, X, device=device)
    x = torch.reshape(grid_x, [B, -1])
    y = torch.reshape(grid_y, [B, -1])
    z = torch.reshape(grid_z, [B, -1])

    # pdb.set_trace()
    # these are B x N
    xyz = torch.stack([x, y, z], dim=2)
    # here is stack in order with xyz
    # this is B x N x 3

    # pdb.set_trace()
    return xyz

def meshgrid3d(B, Z, Y, X, stack=False, device='cuda'):
    # returns a meshgrid sized B x Z x Y x X

    grid_z = torch.linspace(0.0, Z - 1, Z, device=device)
    grid_z = torch.reshape(grid_z, [1, Z, 1, 1])
    grid_z = grid_z.repeat(B, 1, Y, X)

    grid_y = torch.linspace(0.0, Y - 1, Y, device=device)
    grid_y = torch.reshape(grid_y, [1, 1, Y, 1])
    grid_y = grid_y.repeat(B, Z, 1, X)

    grid_x = torch.linspace(0.0, X - 1, X, device=device)
    grid_x = torch.reshape(grid_x, [1, 1, 1, X])
    grid_x = grid_x.repeat(B, Z, Y, 1)
    # here repeat is in the order with ZYX

    if stack:
        # note we stack in xyz order
        # (see https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.grid_sample)
        grid = torch.stack([grid_x, grid_y, grid_z], dim=-1)
        return grid
    else:
        return grid_z, grid_y, grid_x


def normalize_coords(points, pc_range=None):
    """
    points: Tensor of shape (N, 3), where N is the number of points.
    point_cloud_range: The 3D range of the point cloud as [xmin, ymin, zmin, xmax, ymax, zmax].
    voxel_size: Voxel size in each dimension.
    """
    min_coords = pc_range[:3]  # [-40, -40, -1]
    max_coords = pc_range[3:]  # [40, 40, 5.4]

    # Normalize to [0, 1]
    norm_points = (points - torch.tensor(min_coords)) / (torch.tensor(max_coords) - torch.tensor(min_coords))
    # Now scale to [-1, 1] for F.grid_sample
    grid_points = norm_points * 2 - 1

    return grid_points

def gen_grid(Z, Y, X, pc_range, voxel_size, num=1):
    xyz = gridcloud3d(1, Z, Y, X, device='cpu')
    xyz_min = np.array(pc_range[:3])
    xyz_max = np.array(pc_range[3:])
    occ_size = np.array([X, Y, Z])
    xyz = xyz / occ_size * (xyz_max - xyz_min) + xyz_min + 0.5 * voxel_size
    xyz = xyz.reshape(Z, Y, X, 3).permute(2, 1, 0, 3)

    if num != 1:
        xyz = xyz.unsqueeze(-2).repeat(1, 1, 1, 8, 1)
        offset = torch.tensor(list(itertools.product([-0.1, 0.1], repeat=3)))[None, None, None, ...]
        xyz = xyz + offset
    return xyz


class silog_loss(nn.Module):
    def __init__(self, variance_focus=0.85):
        super(silog_loss, self).__init__()
        self.variance_focus = variance_focus

    def forward(self, depth_est, depth_gt, eps=1e-7):

        d = torch.log(depth_est + eps) - torch.log(depth_gt + eps)

        return torch.sqrt((d ** 2).mean() - self.variance_focus * (d.mean() ** 2))

class CorrelationLoss(nn.Module):
    def __init__(self):
        super(CorrelationLoss, self).__init__()

    def forward(self, x, y):
        # x and y are expected to be of shape [batch_size, 16]
        x_mean = torch.mean(x, dim=-1, keepdim=True)
        y_mean = torch.mean(y, dim=-1, keepdim=True)

        x_centered = x - x_mean
        y_centered = y - y_mean

        # Compute the covariance between x and y
        covariance = torch.sum(x_centered * y_centered, dim=-1)

        # Compute the standard deviations
        x_std = torch.sqrt(torch.sum(x_centered ** 2, dim=-1))
        y_std = torch.sqrt(torch.sum(y_centered ** 2, dim=-1))

        # Pearson correlation coefficient for each vector
        correlation = covariance / (x_std * y_std + 1e-6)  # Add epsilon to avoid division by zero
        print(correlation.shape)
        # Return a loss that is 1 - correlation, so higher correlation means lower loss
        return 1 - correlation.mean()


class RangeNormalizedL1Loss(nn.Module):
    def __init__(self, eps=1e-6):
        super(RangeNormalizedL1Loss, self).__init__()
        self.eps = eps
        self.l1_loss = nn.L1Loss()

    def forward(self, x, y):
        # Step 1: Perform range normalization on the last dimension
        x_min = x.min(dim=-1, keepdim=True)[0]
        x_max = x.max(dim=-1, keepdim=True)[0]
        x_normalized = (x - x_min) / (x_max - x_min + self.eps)  # Normalize x

        y_min = y.min(dim=-1, keepdim=True)[0]
        y_max = y.max(dim=-1, keepdim=True)[0]
        y_normalized = (y - y_min) / (y_max - y_min + self.eps)  # Normalize y

        # Step 2: Compute the L1 loss between the normalized tensors
        loss = self.l1_loss(x_normalized, y_normalized)
        return loss

