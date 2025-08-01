
## NuScenes
Download nuScenes V1.0 full dataset data.
For Occupancy Prediction Task, you need to download Occupancy and flow annotation from [OpenOcc v1.2](https://drive.google.com/drive/folders/1lpqjXZRKEvNHFhsxTf0MOE13AZ3q4bTq) as gts.
The [Visibility Masks](https://drive.google.com/file/d/10jB08Z6MLT3JxkmQfxgPVNq5Fu4lHs_h/view) generated from AdaOcc are also necessary.


**Prepare nuScenes data**

We genetate custom annotation files which are different from original BEVDet's
```
python tools/create_data_bevdet.py
```

**Prepare Checkpoints**

The checkpoints can be download from the readme page or using the follow commands:

Resnet50: 
```
wget https://github.com/zhiqi-li/storage/releases/download/v1.0/r50_256x705_depth_pretrain.pth
```

EVA-VIT (optional): 
```
wget https://github.com/exiawsh/storage/releases/download/v1.0/eva02_L_coco_det_sys_o365_remapped.pth
```

Intern-XL (optional) (manual download is recommended): 
```
pip install gdown
gdown 'https://drive.google.com/uc?id=1YQxwgIGHRKvBSI8RvNmJ61d_fUZR3pcb'
```


**Folder structure**
```
VoxelSplat
├── mmdet3d/
├── tools/
├── configs/
├── ckpts/
│   ├── r50_256x705_depth_pretrain.pth
|   ├── eva02_L_coco_det_sys_o365_remapped.pth
|   ├── a-13-a_ep12.pth
├── data/
│   ├── nuscenes/
│   │   ├── gts/  # ln -s occupancy-flow gts OpenOcc to this location
|   |   ├── openocc_v2_ray_mask/  # visibility mask
│   │   ├── maps/
│   │   ├── samples/
│   │   ├── sweeps/
│   │   ├── v1.0-test/
|   |   ├── v1.0-trainval/
|   |   ├── my_gts.gz  # for evaluation
|   |   ├── bevdetv2-nuscenes_infos_val.pkl
|   |   ├── bevdetv2-nuscenes_infos_train.pkl
```