# 🐋 EDBench: Large-Scale Electron Density Data for Machine Learning in Molecular Modeling

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/) 
![Static Badge](https://img.shields.io/badge/Dataverse-green?logo=docsdotrs&label=Data&link=https%3A%2F%2Fdataverse.harvard.edu%2Fdataverse%2FEDBench)
<a href="https://github.com/HongxinXiang/EDBench/blob/main/LICENSE">
    <img alt="GitHub" src="https://img.shields.io/github/license/HongxinXiang/EDBench?style=flat-square">
</a>
<img alt="GitHub last commit" src="https://img.shields.io/github/last-commit/HongxinXiang/EDBench?style=flat-square">


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
  - [⚛️ 4.1 ED-based Methods](#-41-ed-based-methods)  
- [📬 Contact](#-contact)  
- [📘 License](#-license)



## 📢 News

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
> _Citation coming soon. Please stay tuned!_

<p align="center">
  <img src="/docs/images/overview.png" width="600">
</p>

---



## 🧬 2. EDBench Database

Built on PCQM4Mv2, the EDBench dataset contains accurate DFT-computed EDs for **3.3M+ molecules**, enabling deep learning at the electronic scale.

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

| Dataset  | Dir Name              | Link       | Description                                                  |
|----------|-----------------------|------------|--------------------------------------------------------------|
| ED5-EC   | `ed_energy_5w`        | [Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/YAFDHG) | 6 energy components (DF-RKS Final Energy, Nuclear Repulsion Energy, One-Electron Energy,  Two-Electron Energy, DFT Exchange-Correlation Energy, Total Energy) |
| ED5-OE   | `ed_homo_lumo_5w`     | *(Coming)* | 7 orbital energies (HOMO-2, HOMO-1, HOMO-0, LUMO+0, LUMO+1, LUMO+2, LUMO+3) |
| ED5-MM   | `ed_multipole_moments_5w` | *(Coming)* | 4 multipole moment (Dipole X, Dipole Y, Dipole Z, Magnitude) |
| ED5-OCS  | `ed_open_shell_5w`    | *(Coming)* | Binary classification of open-/closed-shell systems          |

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

| Dataset  | Dir Name           | Link       | Description                                    |
|----------|--------------------|------------|------------------------------------------------|
| ED5-MER  | `ed_retrieval_5w`  | *(Coming)* | Cross-modal retrieval: MS ↔ ED                |

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

| Dataset  | Dir Name           | Link       | Description                             |
|----------|--------------------|------------|-----------------------------------------|
| ED5-EDP  | `ed_prediction_5w` | *(Coming)* | Predict ED from molecular structures    |

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

### ⚛️ 4.1 Prediction Tasks

The code and detailed instructions for running prediction tasks can be found [in this 📂directory](./src/prediction_tasks/PointCloud_code).



---



## 📬 Contact

Feel free to open an issue or pull request for questions or contributions. For academic inquiries, contact the authors upon paper publication.

---



## 📘 License

Released for research use under an open-source license (to be finalized upon publication).
