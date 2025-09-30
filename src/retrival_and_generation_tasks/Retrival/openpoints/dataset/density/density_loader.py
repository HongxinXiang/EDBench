"""Modified from DeepGCN and DGCNN
Reference: https://github.com/lightaime/deep_gcns_torch/tree/master/examples/classification
"""
import os
import glob
import h5py
import pandas as pd
import numpy as np
import pickle
import logging
import ssl
import urllib
from pathlib import Path
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from torchvision.datasets.utils import extract_archive, check_integrity
from ..build import DATASETS

# def farthest_point_sample(point, npoint):
#     """
#     Input:
#         xyz: pointcloud data, [N, D]
#         npoint: number of samples
#     Return:
#         centroids: sampled pointcloud index, [npoint, D]
#     """
#     N, D = point.shape
#     xyz = point[:, :3]
#     centroids = np.zeros((npoint,))
#     distance = np.ones((N,)) * 1e10
#     farthest = np.random.randint(0, N)
#     for i in range(npoint):
#         centroids[i] = farthest
#         centroid = xyz[farthest, :]
#         dist = np.sum((xyz - centroid) ** 2, -1)
#         mask = dist < distance
#         distance[mask] = dist[mask]
#         farthest = np.argmax(distance, -1)
#     point = point[centroids.astype(np.int32)]
#     return point


# def load_data(dataset, partition, npoint):
#     point_data = pickle.load(open(f"/root/PointVector/{dataset}/processed/mol_EDthresh0_data.pkl", 'rb'))
#     label_data = pd.read_csv(f"/root/PointVector/{dataset}/raw/{dataset}.csv")

#     labels = label_data[label_data["scaffold_split"] == partition]["label"].tolist()
#     labels = [list(map(float, x.split(" "))) for x in labels]

#     labels = np.array(labels, dtype=np.float32)

#     idxs = label_data[label_data["scaffold_split"] == partition]["index"].tolist()
    
#     coords, densitys = [], []
    
#     for idx in idxs:
#         coord = point_data[f"{idx}"]["electronic_density"]["coords"]
#         d = point_data[f"{idx}"]["electronic_density"]["density"]
#         process_point = np.column_stack((coord, d))
        
#         point = farthest_point_sample(process_point, npoint)
        
#         coords.append(point)

#     # coords = [farthest_point_sample(point_data[f"{idx}"]["electronic_density"]["coords"], npoint) for idx in idxs]
#     # density = [point_data[f"{idx}"]["electronic_density"]["density"] for idx in idxs]

#     # coords = np.concatenate(coords, axis=0)
#     # density = np.concatenate(density, axis=0)

#     return coords, labels

def load_data(dataset, partition, npoint):
    point_data = pickle.load(open(f"/code/PointCloud/{dataset}/processed/{dataset}_{npoint}.pkl", 'rb'))
    coords = point_data[partition]["coords"]
    labels = point_data[partition]["labels"]
#     print(f"/code/PointCloud/{dataset}/processed/{dataset}_{npoint}.pkl", labels.shape)
    
    return coords, labels


@DATASETS.register_module()
class Density(Dataset):
    """
    This is the data loader for Density
    num_points: 1024 by default
    data_dir
    paritition: train or vaild or test
    """

    def __init__(self,
                 name="",
                 split='train',
                 num_points = 1024,
                 transform=None
                 ):
        self.name = name
        self.partition = split
        self.npoint = num_points
        self.data, self.label = load_data(self.name, self.partition, self.npoint)
        logging.info(f'==> sucessfully loaded {self.partition} data')
        self.transform = transform

    def __getitem__(self, item):
        
        pointcloud = self.data[item]
        label = self.label[item]

        data = {'pos': pointcloud[:, :3], 'x': pointcloud, 'y': label}
        if self.transform is not None:
            data = self.transform(data)

        return data

    def __len__(self):
        return len(self.data)


