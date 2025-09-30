# EDBench GNN Baseline

This repository provides baseline implementations for GNN-based models on the EDBench benchmark across two major tasks: **Retrieval**, and **Generation**.

---

## 🔧 Installation

Set up the conda environment:

```
conda env create -f environment.yml -n EDBench_GNN
conda activate EDBench_GNN
```
To run the Retrieval Task, you also need to install the point cloud environment. Refer to the setup guide in the [PointCloud_code](../quantum_property_prediction_tasks/PointCloud_code) repo.

## 📁 Dataset Preparation
For all tasks:

Download and extract the dataset archives.

Move the extracted contents into the corresponding task folder's raw/ directory.

### 🔍 Retrieval Task
Preprocessing
```
cd Retrieval
python ./geoformer/datasets/EDpredict.py --data_path /path/to/downloaded_data
```
Training
```
python train.py --conf ./examples/ED_predict_geo_x3d_lr5e-5/code_seed2024_scaffold_pointmetabase-s-x-3d_Equiformer/augmented_file.yml
```
### 🧬 Generation Task
Preprocessing
```
cd Generation
python ./geoformer/datasets/EDpredict.py --data_path /path/to/downloaded_data
```
Training
```
python train.py --conf ./examples/ED_predict_EGNN/code_seed2026_scaffold/ed_prediction_5w.yml
```
## 📚 Acknowledgments

This repository reuses or builds upon code and ideas from the following open-source projects:

- [EquiformerV2](https://github.com/atomicarchitects/equiformer_v2): *EquiformerV2: Improved Equivariant Transformer for Scaling to Higher-Degree Representations*
- [E(n)-GNN](https://github.com/lucidrains/egnn-pytorch):*E(n)-Equivariant Graph Neural Networks*
- [Geoformer](https://github.com/microsoft/AI2BMD/tree/Geoformer): *Geometric Transformer with Interatomic Positional Encoding*


## 📌 Notes
Make sure all data files are placed under the correct raw/ folder for each task.

You can customize training behavior using the .yml files in the examples/ directory.