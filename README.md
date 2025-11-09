# 🐋 EDBench: Large-Scale Electron Density Data for Molecular Modeling

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
<a href="https://github.com/HongxinXiang/EDBench/blob/main/LICENSE"><img alt="GitHub" src="https://img.shields.io/github/license/HongxinXiang/EDBench?style=flat-square"></a>
<img alt="GitHub last commit" src="https://img.shields.io/github/last-commit/HongxinXiang/EDBench?style=flat-square">
<a href="" target='_blank'><img src="https://visitor-badge.laobi.icu/badge?page_id=HongxinXiang.EDBench-X&left_color=gray&right_color=orange"></a>

<a href='https://hongxinxiang.github.io/projects/EDBench'><img src='https://img.shields.io/badge/Project-Page-pink?style=flat&logo=rescript&logoColor=pink'></a>
<a href='https://arxiv.org/abs/2505.09262'><img src='https://img.shields.io/badge/Arxiv-2505.09262-A42C25?style=flat&logo=arXiv&logoColor=A42C25'></a>
<a href='https://www.arxiv.org/pdf/2505.09262.pdf'><img src='https://img.shields.io/badge/Paper-PDF-yellow?style=flat&logo=arXiv&logoColor=yellow'></a>
<a href='https://dataverse.harvard.edu/dataverse/EDBench'><img src='https://img.shields.io/badge/Data-Dataverse-green?style=flat&logo=icloud&logoColor=green'></a>

---


## 📁 Project Directory / Table of Contents

- [📢 News](#-news)  
- [🧪 1. Summary](#-1-summary)  
- [🧬 2. EDBench Database](#-2-edbench-database)  
- [🧪 3. Benchmark Tasks](#-3-benchmark-tasks)  
  - [🔮 3.1 Prediction Tasks](#-31-prediction-tasks)  
  - [🔁 3.2 Retrieval Task](#-32-retrieval-task)  
  - [🧬 3.3 Generation Task](#-33-generation-task)  
  - [📂 3.4 Dataset File Format](#-34-dataset-file-format)  
- [🚀 4. Running Benchmarks](#-4-running-benchmarks)  
  - [⚛️ 4.1 Quantum Prediction Tasks](#-41-quantum-prediction-tasks)  
  - [⚡ 4.2 Retrieval and Generation Tasks](#-42-retrieval-and-generation-tasks)  
- [📬 Contact](#-contact)  
- [📘 License](#-license)


## TODO

- [ ] Release the full ED dataset (>3M)
- [x] Release model checkpoints
- [x] Release model code for quantum property prediction task
- [x] Release model code for retrieval between molecule and ED task
- [x] Release model code for ED generation task
- [x] Release benchmark dataset

Note: The full dataset has not been released yet. Due to the increased data volume, we are currently coordinating with data storage providers to handle and host the large-scale data efficiently. Once this is completed, the full dataset will be made publicly available.

## 📢 News

- **[2025/09/19]** 🎉 Paper was accepted by _NeurIPS 2025_ !

- **[2025/05/13]** Uploaded code of prediction tasks with X-3D and PointVector.

- **[2025/05/10]** Repository initialized!

---



## 🧪 1. Summary

Most existing molecular machine learning force fields (MLFFs) focus on atom- or molecule-level properties like energy and forces, while overlooking the foundational role of electron density (ED), denoted as $\rho(r)$. According to the Hohenberg–Kohn theorem, ED uniquely determines all ground-state properties of many-body quantum systems. However, ED is expensive to compute via first-principles methods such as Density Functional Theory (DFT), limiting its large-scale use in MLFFs.

**EDBench** 🐋 addresses this gap by providing a large-scale, high-quality dataset of electron densities for over **3.3 million molecules**, based on the PCQM4Mv2 standard. To benchmark electronic-scale learning, we introduce a suite of ED-centric tasks covering:

- **Prediction** of quantum chemical properties
- **Retrieval** across structure and ED modalities
- **Generation** of ED from molecular structures

We demonstrate that ML models can learn from ED with high accuracy and also generate high-quality ED, dramatically reducing DFT costs. All data and benchmarks will be made publicly available to support ED-driven research in **drug discovery** and **materials science**.

> 📄 **Citation**
>
> ```bibtex
> @inproceedings{xiang2025edbench,
>   title         = {EDBench: Large-Scale Electron Density Data for Molecular Modeling},
>   author        = {Hongxin xiang and Ke Li and Mingquan Liu and Zhixiang Cheng and Bin Yao and Wenjie Du and Jun Xia and Li Zeng and Xin Jin and Xiangxiang Zeng},
>   booktitle     = {The Thirty-ninth Annual Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
>   year          = {2025},
>   url           = {https://openreview.net/forum?id=pAd7qVrYPG}
> }
> 
> @misc{xiang2025edbenchlargescaleelectrondensity,
>   title         = {EDBench: Large-Scale Electron Density Data for Molecular Modeling},
>   author        = {Hongxin Xiang and Ke Li and Mingquan Liu and Zhixiang Cheng and Bin Yao and Wenjie Du and Jun Xia and Li Zeng and Xin Jin and Xiangxiang Zeng},
>   year          = {2025},
>   eprint        = {2505.09262},
>   archivePrefix = {arXiv},
>   primaryClass  = {physics.chem-ph},
>   url           = {https://arxiv.org/abs/2505.09262}
> }
> ```


<p align="center">
  <img src="/docs/images/overview.png" width="600">
</p>

---



## 🧬 2. EDBench Database

Built on PCQM4Mv2, the EDBench dataset contains accurate DFT-computed EDs for **3.3M+ molecules**, enabling deep learning at the electronic scale.

| Component   | Description                                            | Link                                                         |
| ----------- | ------------------------------------------------------ | ------------------------------------------------------------ |
| 📂 Full data | DFT-computed electron density data for 3.3M+ molecules | [[Baidu Drive](https://pan.baidu.com/s/1VNoN8Uwo5XCyzk7z19X7Rg?pwd=yxy4)] |



---



## 🧪 3. Benchmark Tasks

We design a suite of benchmark tasks centered on electron density (ED):

### 🔮 3.1 Prediction Tasks

Predict quantum chemical properties from ED representations.

<details>
<summary>📂 Click to expand the directory structure</summary>

```bash
{benchmark_root}
{benchmark root}
├── ed_energy_5w
│   ├── raw
│   │   ├── ed_energy_5w.csv
│   │   ├── readme.md
│   │   └── psi4_grid0.4_cubes
│   │       └── {mol_index}
│   │           ├── Mol1_Dt.cube
│   │           ├── timer.dat
│   │           ├── Mol1.sdf
│   │           ├── Mol1_ESP.cube
│   │           └── {mol_index}_Psi4.out
│   └── processed
│       └── mol_EDthresh{thresh}_data.pkl
├── ed_homo_lumo_5w
│   ├── raw
│   │   ├── ed_homo_lumo_5w.csv
│   │   ├── readme.md
│   │   └── psi4_grid0.4_cubes
│   └── processed
│       └── mol_EDthresh{thresh}_data.pkl
├── ed_multipole_moments_5w
│   ├── raw
│   │   ├── ed_multipole_moments_5w.csv
│   │   ├── readme.md
│   │   └── psi4_grid0.4_cubes
│   └── processed
│       └── mol_EDthresh{thresh}_data.pkl
└── ed_open_shell_5w
    ├── raw
    │   ├── ed_open_shell_5w.csv
    │   ├── readme.md
    │   └── psi4_grid0.4_cubes
    └── processed
        └── mol_EDthresh{thresh}_data.pkl
```
</details>

| Dataset  | Dir Name              | Link                                                                                         | Description                                                  |
|----------|-----------------------|----------------------------------------------------------------------------------------------|--------------------------------------------------------------|
| ED5-EC   | `ed_energy_5w`        | [Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/YAFDHG) | 6 energy components (DF-RKS Final Energy, Nuclear Repulsion Energy, One-Electron Energy,  Two-Electron Energy, DFT Exchange-Correlation Energy, Total Energy) |
| ED5-OE   | `ed_homo_lumo_5w`     | [Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/WVQMJP) | 7 orbital energies (HOMO-2, HOMO-1, HOMO-0, LUMO+0, LUMO+1, LUMO+2, LUMO+3) |
| ED5-MM   | `ed_multipole_moments_5w` | [Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/UHLZOE)                                                                                | 4 multipole moment (Dipole X, Dipole Y, Dipole Z, Magnitude) |
| ED5-OCS  | `ed_open_shell_5w`    | [Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/XFIK9H)                                                                                | Binary classification of open-/closed-shell systems          |

---

### 🔁 3.2 Retrieval Task

Cross-modal retrieval between molecular structures (MS) and electron densities (ED).

<details>
<summary>📂 Click to expand the directory structure</summary>

```bash
{benchmark_root}
├── ed_retrieval_5w/
│   ├── raw/
│   │   ├── ed_retrieval_5w.csv
│   │   ├── readme.md
│   │   └── psi4_grid0.4_cubes/
│   └── processed/
│       └── mol_EDthresh{thresh}_data.pkl
```
</details>

| Dataset | Dir Name          | Link                                                         | Description                    |
| ------- | ----------------- | ------------------------------------------------------------ | ------------------------------ |
| ED5-MER | `ed_retrieval_5w` | [Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/CCSJLC) | Cross-modal retrieval: MS ↔ ED |

---

### 🧬 3.3 Generation Task

Generate ED representations from molecular structures.

<details>
<summary>📂 Click to expand the directory structure</summary>

```bash
{benchmark_root}
├── ed_prediction_5w/
│   ├── raw/
│   │   ├── ed_prediction_5w.csv
│   │   ├── readme.md
│   │   └── psi4_grid0.4_cubes/
│   └── processed/
│       └── mol_EDthresh{thresh}_data.pkl
```
</details>

| Dataset  | Dir Name           | Link | Description                             |
|----------|--------------------|------|-----------------------------------------|
| ED5-EDP  | `ed_prediction_5w` | [Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/SLYBD1) | Predict ED from molecular structures    |

---

### 📂 3.4 Dataset File Format

Each `raw/` directory includes a `.csv` summary file describing each molecule.

#### 📌 Common Columns

- `index`: Molecule index  
- `smiles`: Original SMILES  
- `canonical_smiles`: Canonicalized SMILES  
- `scaffold_split`: Scaffold-based split (80% train / 10% valid / 10% test)  
- `random_split`: Random split (80% train / 10% valid / 10% test)

#### 🧾 Task-Specific Columns

- **Prediction**:  
  - `label`: Ground-truth values (space-separated if multi-task)  
- **Retrieval**:  
  - `negative_index`: Space-separated indices of 10 negative samples

---



## 🚀 4. Running Benchmarks

### ⚛️ 4.1 Quantum Prediction Tasks

**Description:**  

This task focuses on predicting quantum properties of molecules using point-cloud representations of electronic density. It is designed to evaluate how accurately models can capture molecular electronic behavior and properties.

| Task                     | Datasets                                                     | Code                                                         | Checkpoint                                                 |
| ------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ---------------------------------------------------------- |
| Quantum Prediction Tasks | **ED5-EC**: 6 energy components<br />**ED5-OE**: 7 orbital energies<br />**ED5-MM**: 4 multipole moment<br />**ED5-OCS**: open-/closed-shell classification | [Code 📂](./src/quantum_property_prediction_tasks/PointCloud_code) | [Download](https://1drv.ms/f/c/53030532e7d1aed6/EmnQzcCAcs9HtxG47y8yuj8Bh-gPY_rNc6EugPd80A4v5w?e=PZp9Do) |

---

### ⚡ 4.2 Retrieval and Generation Tasks

**Description:**  

This task involves two objectives:  

1. Retrieval between molecular structures and their electronic densities.  
2. Predicting electronic densities from molecular conformations.  

These tasks assess the model’s ability to connect molecular structures with their electron distributions and to generate accurate electronic density predictions.

| Task                           | Description                                                  | Code                                          | Checkpoint                                                   |
| ------------------------------ | ------------------------------------------------------------ | --------------------------------------------- | ------------------------------------------------------------ |
| Retrieval and Generation Tasks | **ED5-MER**: Retrieval between molecular structures and ED<br/>**ED5-EDP**: Electronic density prediction | [Code 📂](./src/retrival_and_generation_tasks) | [Download](https://1drv.ms/f/c/53030532e7d1aed6/EubMp-Y5cGJDokOu1XJbKbsBqvOmvoR_-8v82FH9SI5K_g?e=KesJNu) |



---





## 📬 Contact

Feel free to open an issue or pull request for questions or contributions. For academic inquiries, contact the authors upon paper publication.

---



## 📘 License

Released for research use under an open-source MIT license.
