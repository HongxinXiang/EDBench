from typing import Any, Dict, List
from torch_geometric.data import Data, Batch
import torch_geometric
import torch


class GeoformerDataCollator:
    def __init__(self, max_nodes=None) -> None:
        self.max_nodes = max_nodes

    @staticmethod
    def _pad_attn_bias(attn_bias: torch.Tensor, max_node: int) -> torch.Tensor:
        N = attn_bias.shape[0]
        if N <= max_node:
            attn_bias_padded = torch.zeros(
                [max_node, max_node], dtype=torch.float
            ).fill_(float("-inf"))
            attn_bias_padded[:N, :N] = attn_bias
            attn_bias_padded[N:, :N] = 0
        else:
            print(
                f"Warning: max_node {max_node} is too small to hold all nodes {N} in a batch"
            )
            print("Play truncation...")

        return attn_bias_padded

    @staticmethod
    def _pad_feats(feats: torch.Tensor, max_node: int) -> torch.Tensor:
        N, *_ = feats.shape
        if N <= max_node:
            feats_padded = torch.zeros([max_node, *_], dtype=feats.dtype)
            feats_padded[:N] = feats
        else:
            print(
                f"Warning: max_node {max_node} is too small to hold all nodes {N} in a batch"
            )
            print("Play truncation...")

        return feats_padded

    def _check_attn_bias(self, feat: dict):
        num_node = len(feat["z"])
        if "attn_bias" not in feat:
            return torch.zeros([num_node, num_node]).float()
        else:
            return torch.tensor(feat["attn_bias"]).float()

    def __call__(self, data_list: List[dict]) -> Dict[str, Any]:        

        batch = dict()
        for data in data_list:
            data.natoms = len(data.z)
        data['graph'] = Batch.from_data_list(data_list)
        batch["labels"] = data['graph'].y
        batch['name'] = [data.name for data in data_list]
        batch['graph'] = {}

        mask_point = data['graph']['z'] == 99
        pos_point = data['graph']['pos'][mask_point] # [bx2048,3]
        density = data['graph']['y'][mask_point] # [bx2048,1]
        batch['point'] = torch.cat((pos_point.reshape(-1,2048,3),density.reshape(-1,2048,1)),dim=-1)

        max_node = (
            max(feat["z"][(feat["z"] != 98) & (feat["z"] != 99)].shape[0] for feat in data_list)
            if self.max_nodes is None
            else self.max_nodes
        )

        batch['graph']["z"] = torch.stack(
            [self._pad_feats(feat["z"][(feat["z"] != 98) & (feat["z"] != 99)] , max_node) for feat in data_list]
        )
        batch['graph']["pos"] = torch.stack(
            [self._pad_feats(feat["pos"][(feat["z"] != 98) & (feat["z"] != 99)], max_node) for feat in data_list]
        )

        return batch
