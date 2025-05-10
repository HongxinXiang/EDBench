# EDBench🐋: Large-Scale Electron Density Data for Machine Learning Applications in Molecular Modeling

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)



## News!

- [2025/05/10] Initialize the repository.



## Summary

Existing molecular machine learning force fields (MLFF) focus on the learning of atoms, molecules and simple quantum chemical properties (such as energy and force), but ignore the importance of electron density (ED) $\rho(r)$ in accurately understanding molecular force fields (MFF). ED describes the probability of finding electrons at specific locations around atoms or molecules, which uniquely determines all ground state properties (such as energy, molecular structure, etc.) of interacting multi-particle systems according to the Hohenberg-Kohn theorem. However, the calculation of ED relies on the time-consuming first-principles density functional theory (DFT), which leads to the lack of large-scale ED data and limits its application in MLFFs. 

In this paper, we introduce EDBench 🐋, a large-scale, high-quality dataset of electron densities (ED) designed to advance machine learning research at the electronic scale. Built upon the PCQM4Mv2 standard, EDBench provides accurate ED data for over 3.3 million molecules. To comprehensively evaluate the ability of models to understand and utilize electronic information, we design a suite of ED-centric benchmark tasks spanning prediction, retrieval, and generation. Our evaluation on several state-of-the-art methods demonstrates that learning from ED is not only feasible but also achieves high accuracy. Moreover, we show that machine learning method can efficiently generate ED with comparable precision while significantly reducing the computational cost relative to traditional DFT calculations. All data and benchmarks from EDBench will be freely available, laying a robust foundation for ED-driven drug discovery and materials science. All data and code will be released upon paper acceptance. 

You can find more information in our paper. If you are using EDBench in your research paper, please cite us as

```bibx

```



<img src='/docs/images/overview.png' width='600'>



## EDBench Database

Built upon the PCQM4Mv2, we propose a large-scale, high-quality dataset of electron densities (ED)  over 3.3 million molecules.



## Benchmarks

We designs a suite of ED-centric benchmark tasks spanning prediction of diverse quantum chemical properties, retrieval between molecular structures (MS) and ED, and generation of ED from MS.



### Prediction tasks

| Datasets | Links | Description |
| -------- | ----- | ----------- |
|          |       |             |
|          |       |             |
|          |       |             |
|          |       |             |



### Retrieval task



### Generation task









