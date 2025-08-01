# Step-by-step installation instructions


**a. Create a conda virtual environment and activate it.**
```shell
conda create -n voxelsplat python=3.10 -y
conda activate voxelsplat
```

**b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/).**
```shell
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 --index-url https://download.pytorch.org/whl/cu118
# Recommended torch>=1.13
```

**c. Install mmcv and mmdetection.**
```shell
pip intall mmengine==0.10.7
pip install mmcv-full==1.6.0
pip install mmdet==2.28.2
```

**d. Install Gsplat for rendering.**
```shell
pip install git+https://github.com/nerfstudio-project/gsplat.git@v1.4.0
```

**e. Install VoxelSplat from source code.**
```shell
git clone https://github.com/ZZY816/VoxelSplat.git
cd VoxelSplat
pip install -r requirements.txt
pip install -e .
```

**f. Use InternImage as backbone. (optional)**
```shell
cd ops 
pip install -v -e .
```

**h. Prepare pretrained models.**
```shell
cd FB-BEV
mkdir ckpts

cd ckpts & wget TODO.pth
```


