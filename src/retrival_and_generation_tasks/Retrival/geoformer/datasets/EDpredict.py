import os
import os.path as osp
import sys
from typing import Callable, List, Optional
import pandas as pd
import numpy as np
import torch
from torch import Tensor
import pickle
import glob
import torch.nn.functional as F
from tqdm import tqdm
from collections import defaultdict
from rdkit import Chem, RDLogger
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem.rdchem import HybridizationType
from torch_scatter import scatter
from pathlib import Path
from torch_geometric.transforms import Compose

RDLogger.DisableLog('rdApp.*')  # type: ignore
from torch_geometric.data import (
    Data,
    InMemoryDataset,
)

types = {'H':0, 'C':1, 'N':2, 'O':3, 'F':4}
UNKNOWN_INDEX = len(types)  # 预留索引5表示未知原子

def get_atom_type_index(atom_symbol: str) -> int:
    return types.get(atom_symbol, UNKNOWN_INDEX)
    
bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}

HAR2EV = 27.211386246
KCALMOL2EV = 0.04336414

conversion = torch.tensor([
    1., 1., HAR2EV, HAR2EV, HAR2EV, 1., HAR2EV, HAR2EV, HAR2EV, HAR2EV, HAR2EV,
    1., KCALMOL2EV, KCALMOL2EV, KCALMOL2EV, KCALMOL2EV, 1., 1., 1.
])

atomrefs = {
    6: [0., 0., 0., 0., 0.],
    7: [
        -13.61312172, -1029.86312267, -1485.30251237, -2042.61123593,
        -2713.48485589
    ],
    8: [
        -13.5745904, -1029.82456413, -1485.26398105, -2042.5727046,
        -2713.44632457
    ],
    9: [
        -13.54887564, -1029.79887659, -1485.2382935, -2042.54701705,
        -2713.42063702
    ],
    10: [
        -13.90303183, -1030.25891228, -1485.71166277, -2043.01812778,
        -2713.88796536
    ],
    11: [0., 0., 0., 0., 0.],
}

def get_single_csv_path(directory):
        # 递归搜索所有子目录中的CSV文件
    csv_files = list(Path(directory).rglob("*.csv"))
    
    # 过滤出文件（排除目录本身）
    csv_files = [f for f in csv_files if f.is_file()]
    
    # 检查数量
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {directory} or its subdirectories")
    elif len(csv_files) > 1:
        raise ValueError(f"Multiple CSV files found. Expected only one.\nFound: {[str(f) for f in csv_files]}")
    
    return str(csv_files[0].resolve())  # 返回解析符号链接后的绝对路径

class EDpredict(InMemoryDataset):

    def __init__(
        self,
        root: str,
        thresh: float,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
    ) -> None:
        self.data_csv = get_single_csv_path(root)  # 保存为实例变量供process使用

        def transform_filter(data:Data):
            mask = data.y > thresh
            data.z = data.z[mask]
            data.pos = data.pos[mask]
            data.y = data.y[mask]
            data.node_type = data.node_type[mask]
            return data

        super().__init__(root, transform_filter, pre_transform, pre_filter)

        self.data, self.slices = torch.load(self.processed_paths[0])  # 确保process已生成数据

    def mean(self, target: int) -> float:
        y = torch.cat([self.get(i).y for i in range(len(self))], dim=0)
        return float(y[:, target].mean())

    def std(self, target: int) -> float:
        y = torch.cat([self.get(i).y for i in range(len(self))], dim=0)
        return float(y[:, target].std())

    def atomref(self, target: int) -> Optional[Tensor]:
        if target in atomrefs:
            out = torch.zeros(100)
            out[torch.tensor([1, 6, 7, 8, 9])] = torch.tensor(atomrefs[target])
            return out.view(-1, 1)
        return None
    
    @property
    def raw_file_names(self):
        return [self.data_csv, 'sdf/', 'mol2/']


    @property
    def processed_file_names(self) -> str:
        return ['data.pt', 'split.npz']

    def process(self) -> None:
        """
        ed_prediction_5w.csv, index,smiles,canonical_smiles,mol_cluster,scaffold_split,random_split

        """

        # 读取CSV文件
        thresh_hold = 0.0
        df = pd.read_csv(self.raw_paths[0])
        columns = df.columns.tolist()

        # 找到 self.raw_paths[0] 所在路径下的唯一一个pkl文件的路径为 pkl_file
        raw_dir = os.path.dirname(self.raw_paths[0])
        pkl_files = glob.glob(os.path.join(raw_dir, "*.pkl"))
        pkl_file = pkl_files[0] 
        with open(pkl_file, "rb") as file:  # 注意模式是 'rb'
            data_mol = pickle.load(file)

        data_list = []
        original_to_processed = {}  # 记录原始CSV行号到data_list索引的映射
        for i, row in tqdm(df.iterrows(), total=len(df), desc="处理分子"):
            index = str(row['index'])
            struct_mol = data_mol[index]['mol']
            struct_ed = data_mol[index]['electronic_density']
            pos_mol = struct_mol['coords']
            pos_mol = torch.tensor(pos_mol, dtype=torch.float)
            
            pos_ed = struct_ed['coords']
            pos_ed = torch.tensor(pos_ed, dtype=torch.float)
            z_mol = struct_mol['x']
            z_mol = torch.tensor(z_mol, dtype=torch.long)
            z_ed = [99] * len(pos_ed) # 99 for ed 98 for global nodes
            z_ed = torch.tensor(z_ed, dtype=torch.long)
            
            # do mask here
            # mask = struct_ed['density'] > thresh_hold
            # z_ed = z_ed[mask]
            # pos_ed = pos_ed[mask]

            z = torch.cat((z_mol,torch.tensor([98]),z_ed,torch.tensor([98])), dim=0)
            pos = torch.cat((pos_mol,torch.mean(pos_mol,dim=0,keepdim=True),pos_ed,torch.mean(pos_ed,dim=0,keepdim=True)), dim=0)
            node_type = [0] * len(pos_mol) + [0] + [1] * len(pos_ed) + [1]
            node_type = torch.tensor(node_type, dtype=torch.long)
            y = [2.0] * len(pos_mol) + [2.0] + list(struct_ed['density']) + [2.0]
            y = torch.tensor(y, dtype = torch.float)

            name = index
            data = Data(
                z=z,
                pos=pos,
                node_type = node_type,
                name=name,
                y = y,
                idx=i,
            )

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            original_to_processed[i] = len(data_list)  # 当前data_list长度即为新索引
            data_list.append(data)

        split = defaultdict(list)

        for orig_idx, row in df.iterrows():
            # 只处理成功转换的行
            if orig_idx not in original_to_processed:
                continue
            
            # 获取处理后的索引
            processed_idx = original_to_processed[orig_idx]
            
            # 处理scaffold_split
            if 'scaffold_split' in columns:
                scaffold_split = row['scaffold_split']
                if scaffold_split in ['train', 'valid', 'test']:
                    key = f'scaffold_{scaffold_split}'
                    split[key].append(processed_idx)
            
            # 处理random_split
            if 'random_split' in columns:
                random_split = row['random_split']
                if random_split in ['train', 'valid', 'test']:
                    key = f'random_{random_split}'
                    split[key].append(processed_idx)

        split = dict(split)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        split_np = {k: np.array(v) for k, v in split.items()}  # 若 v 是张量
        np.savez(self.processed_paths[1], **split_np)
    

def get_csv_paths(base_dir):
    csv_files = []
    csv_dirs = []  # 使用集合避免重复路径
    
    # 转换为绝对路径并遍历
    for root, _, files in os.walk(os.path.abspath(base_dir)):
        for file in files:
            # 检查文件扩展名是否为.csv（不区分大小写）
            if os.path.splitext(file)[1].lower() == '.csv':
                csv_files.append(os.path.join(root, file))
                csv_dirs.append(os.path.dirname(root))  # 添加目录路径
    return csv_files, list(csv_dirs)

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process CSV directories for EDpredict.")
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to the folder containing processed CSV directories')
    args = parser.parse_args()

    # 获取路径下所有csv文件的绝对路径，以及csv所在文件夹的所有绝对路径
    csv_files, csv_dirs = get_csv_paths(args.data_path)

    for csv_file, csv_dir in zip(csv_files, csv_dirs):
        os.system('rm -rf {}/*'.format(os.path.join(csv_dir, 'processed')))
        data = EDpredict(csv_dir, 0.9)
        split = np.load(data.processed_paths[1])
        print(data[0].z)
        print(data[0].node_type)
        print(data[0].y)
        print(data.index_select(split['random_train']))
        print(data.index_select(split['random_valid']))
        print(data.index_select(split['random_test']))
