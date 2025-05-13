# PointVector and X-3D

## Install
```
conda create -n openpoints -y python=3.7 numpy=1.20 numba
conda activate openpoints

# please always double check installation for pytorch and torch-scatter from the official documentation
conda install -y pytorch=1.10.1 torchvision cudatoolkit=11.3 -c pytorch -c nvidia
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.10.1+cu113.html

pip install -r requirements.txt

# install cpp extensions, the pointnet++ library
cd openpoints/cpp/pointnet2_batch
python setup.py install
cd ../
```

## Dataset Pre-processing
```
python process_dataset --dataset ed_homo_lumo_5w --npoint 1024
```

## Train
### Pointvector
```
CUDA_VISIBLE_DEVICES=0 python ./examples/regression/main.py --cfg ./cfgs/lumo_homo/pointvector-s.yaml --seed 2025 --npoint 2048
```
### X-3D
```
CUDA_VISIBLE_DEVICES=0 python ./examples/regression/main.py --cfg ./cfgs/lumo_homo/pointmetabase-s-x-3d.yaml --seed 2025 --npoint 2048
```


## Acknowledgment
This repository is built on reusing codes of [OpenPoints](https://github.com/guochengqian/openpoints)， [PointNeXt](https://github.com/guochengqian/PointNeXt) and [PointMetaBase](https://github.com/linhaojia13/PointMetaBase.git)