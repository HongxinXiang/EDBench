from typing import Callable, Dict

import torch
from torch import nn
from .utils import GaussianRBF,CosineCutoff
from typing import Callable, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import KNNBatchEdgeConstructor
from torch_geometric.nn import radius_graph
from torch_geometric.nn import radius_graph
from .utils import Dense,shifted_softplus
from torch_scatter import scatter_add 
from torch_scatter import scatter_mean
from torch_cluster import knn_graph
import torch.nn.functional as F

__all__ = ["SchNet", "SchNetInteraction"]


class SchNetInteraction(nn.Module):
    r"""SchNet interaction block for modeling interactions of atomistic systems."""

    def __init__(
        self,
        n_atom_basis: int,
        n_rbf: int,
        n_filters: int,
        activation: Callable = shifted_softplus,
    ):
        """
        Args:
            n_atom_basis: number of features to describe atomic environments.
            n_rbf (int): number of radial basis functions.
            n_filters: number of filters used in continuous-filter convolution.
            activation: if None, no activation function is used.
        """
        super(SchNetInteraction, self).__init__()
        self.in2f = Dense(n_atom_basis, n_filters, bias=False, activation=None)
        self.f2out = nn.Sequential(
            Dense(n_filters, n_atom_basis, activation=activation),
            Dense(n_atom_basis, n_atom_basis, activation=None),
        )
        self.filter_network = nn.Sequential(
            Dense(n_rbf, n_filters, activation=activation), Dense(n_filters, n_filters)
        )

    def forward(
        self,
        x: torch.Tensor,
        f_ij: torch.Tensor,
        idx_i: torch.Tensor,
        idx_j: torch.Tensor,
        rcut_ij: torch.Tensor,
    ):
        """Compute interaction output.

        Args:
            x: input values
            Wij: filter
            idx_i: index of center atom i
            idx_j: index of neighbors j

        Returns:
            atom features after interaction
        """
        x = self.in2f(x)
        Wij = self.filter_network(f_ij)
        Wij = Wij * rcut_ij[:, None]

        # continuous-filter convolution
        x_j = x[idx_j]
        x_ij = x_j * Wij
        x = scatter_add(x_ij, idx_i, dim=0,dim_size=x.shape[0])

        x = self.f2out(x)
        return x


class SchNet(nn.Module):
    """SchNet architecture for learning representations of atomistic systems

    References:

    .. [#schnet1] Schütt, Arbabzadah, Chmiela, Müller, Tkatchenko:
       Quantum-chemical insights from deep tensor neural networks.
       Nature Communications, 8, 13890. 2017.
    .. [#schnet_transfer] Schütt, Kindermans, Sauceda, Chmiela, Tkatchenko, Müller:
       SchNet: A continuous-filter convolutional neural network for modeling quantum
       interactions.
       In Advances in Neural Information Processing Systems, pp. 992-1002. 2017.
    .. [#schnet3] Schütt, Sauceda, Kindermans, Tkatchenko, Müller:
       SchNet - a deep learning architecture for molceules and materials.
       The Journal of Chemical Physics 148 (24), 241722. 2018.

    """

    def __init__(
        self,
        n_atom_basis: int = 128,
        n_interactions: int = 6,
        radial_basis: nn.Module = GaussianRBF(n_rbf = 32, cutoff = 10, start = 0.0),
        cutoff_fn: Optional[Callable] = CosineCutoff(10),
        n_filters: int = 128,
        shared_interactions: bool = False,
        max_z: int = 100,
        activation: Callable = shifted_softplus,
        dropout: float = 0.3,
    ):
        """
        Args:
            n_atom_basis: number of features to describe atomic environments.
                This determines the size of each embedding vector; i.e. embeddings_dim.
            n_interactions: number of interaction blocks.
            radial_basis: layer for expanding interatomic distances in a basis set
            cutoff_fn: cutoff function
            n_filters: number of filters used in continuous-filter convolution
            shared_interactions: if True, share the weights across
                interaction blocks and filter-generating networks.
            max_z: maximal nuclear charge
            activation: activation function
        """
        super().__init__()
        self.n_atom_basis = n_atom_basis
        self.size = (self.n_atom_basis,)
        self.n_filters = n_filters or self.n_atom_basis
        self.radial_basis = radial_basis
        
        self.cutoff_fn = cutoff_fn
        self.cutoff = cutoff_fn.cutoff
        self.edge_embedding = nn.Embedding(4, self.radial_basis.n_rbf)
        # layers
        self.ele_embedding = nn.Embedding(100,self.n_atom_basis)
        self.res_embedding = nn.Embedding(100,self.n_atom_basis)
        self.pos_embedding = nn.Embedding(100,self.n_atom_basis)
        self.interactions = nn.ModuleList([
            SchNetInteraction(
                n_atom_basis=self.n_atom_basis,
                n_rbf=self.radial_basis.n_rbf,
                n_filters=self.n_filters,
                activation=activation,
            )
            for i in range(n_interactions)])
        self.embedding_out = nn.Linear(self.n_atom_basis, 1)
        self.edge_constructor = KNNBatchEdgeConstructor(
            k_neighbors=9,
            delete_self_loop=True,
            global_node_id_vocab=[98]
            )

    def forward(self, data):
        data = data['graph']
        z,pos,z_type,batch = data.z, data.pos, data.node_type, data.batch
        
        intra_edges, inter_edges, global_normal_edges, global_global_edges = self.edge_constructor(z, batch, z_type,X=pos)
        
        edge_index = torch.cat([intra_edges, inter_edges, global_global_edges, global_normal_edges], dim=1)
        
        edge_attr = torch.cat([
            torch.zeros_like(intra_edges[0]),
            torch.ones_like(inter_edges[0]),
            torch.ones_like(global_global_edges[0]) * 2,
            torch.ones_like(global_normal_edges[0]) * 3])
        
        idx_i, idx_j = edge_index
        n_atoms = z.shape[0]
        r_ij = pos[idx_i] - pos[idx_j]
        
        # compute atom and pair features
        x = self.ele_embedding(z)
        # x = self.embedding(z)
        d_ij = torch.norm(r_ij, dim=1)
        f_ij = self.radial_basis(d_ij) + self.edge_embedding(edge_attr)
        rcut_ij = self.cutoff_fn(d_ij)
        # compute interaction block to update atomic embeddings
        
        for interaction in self.interactions:
            
            v = interaction(x, f_ij, idx_i, idx_j, rcut_ij)
            
            x = x + v
            
        mask = z_type != 99 # 99 for ed nodes
        
        x = self.embedding_out(x)
        
#         x[mask] = 2.0
        
        return x,mask
