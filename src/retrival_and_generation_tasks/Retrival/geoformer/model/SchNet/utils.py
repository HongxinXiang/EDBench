import math
import torch
import torch.nn as nn
from torch_scatter import scatter_mean,scatter_sum
from torch.nn.init import xavier_uniform_
from torch.nn.init import zeros_
from typing import Callable, Dict, Optional
import torch.nn.functional as F

def shifted_softplus(x: torch.Tensor):
    r"""Compute shifted soft-plus activation function.

    .. math::
       y = \ln\left(1 + e^{-x}\right) - \ln(2)

    Args:
        x (torch.Tensor): input tensor.

    Returns:
        torch.Tensor: shifted soft-plus of input.

    """
    return F.softplus(x) - math.log(2.0)

def cosine_cutoff(input: torch.Tensor, cutoff: torch.Tensor):
    """ Behler-style cosine cutoff.

        .. math::
           f(r) = \begin{cases}
            0.5 \times \left[1 + \cos\left(\frac{\pi r}{r_\text{cutoff}}\right)\right]
              & r < r_\text{cutoff} \\
            0 & r \geqslant r_\text{cutoff} \\
            \end{cases}

        Args:
            cutoff (float, optional): cutoff radius.

        """

    # Compute values of cutoff function
    input_cut = 0.5 * (torch.cos(input * math.pi / cutoff) + 1.0)
    # Remove contributions beyond the cutoff radius
    input_cut *= (input < cutoff).float()
    return input_cut


class CosineCutoff(nn.Module):
    r""" Behler-style cosine cutoff module.

    .. math::
       f(r) = \begin{cases}
        0.5 \times \left[1 + \cos\left(\frac{\pi r}{r_\text{cutoff}}\right)\right]
          & r < r_\text{cutoff} \\
        0 & r \geqslant r_\text{cutoff} \\
        \end{cases}

    """

    def __init__(self, cutoff: float):
        """
        Args:
            cutoff (float, optional): cutoff radius.
        """
        super(CosineCutoff, self).__init__()
        self.register_buffer("cutoff", torch.FloatTensor([cutoff]))

    def forward(self, input: torch.Tensor):
        return cosine_cutoff(input, self.cutoff)

def gaussian_rbf(inputs: torch.Tensor, offsets: torch.Tensor, widths: torch.Tensor):
    coeff = -0.5 / torch.pow(widths, 2)
    diff = inputs[..., None] - offsets
    y = torch.exp(coeff * torch.pow(diff, 2))
    return y

class GaussianRBF(nn.Module):
    r"""Gaussian radial basis functions."""

    def __init__(
        self, n_rbf: int, cutoff: float, start: float = 0.0, trainable: bool = False
    ):
        """
        Args:
            n_rbf: total number of Gaussian functions, :math:`N_g`.
            cutoff: center of last Gaussian function, :math:`\mu_{N_g}`
            start: center of first Gaussian function, :math:`\mu_0`.
            trainable: If True, widths and offset of Gaussian functions
                are adjusted during training process.
        """
        super(GaussianRBF, self).__init__()
        self.n_rbf = n_rbf

        # compute offset and width of Gaussian functions
        offset = torch.linspace(start, cutoff, n_rbf)
        widths = torch.FloatTensor(
            torch.abs(offset[1] - offset[0]) * torch.ones_like(offset)
        )
        if trainable:
            self.widths = nn.Parameter(widths)
            self.offsets = nn.Parameter(offset)
        else:
            self.register_buffer("widths", widths)
            self.register_buffer("offsets", offset)

    def forward(self, inputs: torch.Tensor):
        return gaussian_rbf(inputs, self.offsets, self.widths)
    
    
def covert_res_type_to_index(res_type):
    count = 0
    res_index = []
    for i,t in enumerate(res_type):
        if i == 0:
            res_index.append(0)
        elif(t == res_type[i-1]):
            res_index.append(count)
        else:
            res_index.append(count + 1)
            count += 1
    return torch.tensor(res_index).to(res_type.device)
            
def Blocking(h, v, pos, x_res,res_index, batch):
    """
    Blocking the complexs into block-level
    Args:
        h:atom-level scalar features
        v:atom-level vector features
        pos:atom positions
        res_type: residule types of atoms
        batch: batch of atoms
    Return:
        H:block-level scalar features
        V:block-level vector features
        POS:block positions
        batch_block: batch of blocks
    """
    H = scatter_mean(h,res_index,dim=0)
    V = scatter_mean(v,res_index,dim=0)
    POS = scatter_mean(pos,res_index,dim=0)
    RES = scatter_mean(x_res,res_index,dim=0).long()
    bacth_block = scatter_mean(batch,res_index,dim=0).long()
    return H,V,POS,RES,bacth_block
    
def scatter_add(
    x: torch.Tensor, idx_i: torch.Tensor, dim_size: int, dim: int = 0
) -> torch.Tensor:
    """
    Sum over values with the same indices.

    Args:
        x: input values
        idx_i: index of center atom i
        dim_size: size of the dimension after reduction
        dim: the dimension to reduce

    Returns:
        reduced input

    """
    return _scatter_add(x, idx_i, dim_size, dim)

def _scatter_add(
    x: torch.Tensor, idx_i: torch.Tensor, dim_size: int, dim: int = 0
) -> torch.Tensor:
    shape = list(x.shape)
    shape[dim] = dim_size
    tmp = torch.zeros(shape, dtype=x.dtype, device=x.device)
    y = tmp.index_add(dim, idx_i, x)
    return y

class Dense(nn.Linear):
    r"""Fully connected linear layer with activation function.

    .. math::
       y = activation(x W^T + b)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        activation: Callable = None,
        weight_init: Callable = xavier_uniform_,
        bias_init: Callable = zeros_,
    ):
        """
        Args:
            in_features: number of input feature :math:`x`.
            out_features: umber of output features :math:`y`.
            bias: If False, the layer will not adapt bias :math:`b`.
            activation: if None, no activation function is used.
            weight_init: weight initializer from current weight.
            bias_init: bias initializer from current bias.
        """
        self.weight_init = weight_init
        self.bias_init = bias_init
        super(Dense, self).__init__(in_features, out_features, bias)

        self.activation = activation
        if self.activation is None:
            self.activation = nn.Identity()

    def reset_parameters(self):
        self.weight_init(self.weight)
        if self.bias is not None:
            self.bias_init(self.bias)

    def forward(self, input: torch.Tensor):
        y = F.linear(input, self.weight, self.bias)
        y = self.activation(y)
        return y

def VectorNorm(V):
    V_norm = torch.norm(V,dim=1,keepdim=True)
    V_norm_max = torch.max(V_norm,dim=2,keepdim=True).values
    V_norm_min = torch.min(V_norm,dim=2,keepdim=True).values
    return V / (V_norm + 1e-6) * (V_norm - V_norm_min)/(V_norm_max - V_norm_min + 1e-6)

def sequential_or(*tensors):
    res = tensors[0]
    for mat in tensors[1:]:
        res = torch.logical_or(res, mat)
    return res

def rot_z(gamma):
    return torch.tensor([[cos(gamma), -sin(gamma), 0], [sin(gamma), cos(gamma), 0], [0, 0, 1]], dtype=gamma.dtype)


def rot_y(beta):
    return torch.tensor([[cos(beta), 0, sin(beta)], [0, 1, 0], [-sin(beta), 0, cos(beta)]], dtype=beta.dtype)


def rot(alpha, beta, gamma):
    return rot_z(alpha) @ rot_y(beta) @ rot_z(gamma)


class Classifier(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.out = nn.Sequential(nn.Linear(2 * dim, dim), nn.ReLU(),
                                 nn.Linear(dim, 1), nn.Sigmoid())

    def forward(self, x1, x2):
        return self.out(torch.cat((x1, x2), dim=-1)).squeeze(-1)
    
def _knn_edges(dist, src_dst, k_neighbors, batch_info):
    '''
    :param dist: [Ef], given distance of edges
    :param src_dst: [Ef, 2], full possible edges represented in (src, dst)
    '''
    offsets, batch_id, max_n, gni2lni = batch_info

    k_neighbors = min(max_n, k_neighbors)

    BIGINT = 1e10  # assign a large distance to invalid edges
    N = batch_id.shape[0]

    dist_mat = torch.ones(N, max_n, device=dist.device, dtype=dist.dtype) * BIGINT  # [N, max_n]
    dist_mat[(src_dst[0], gni2lni[src_dst[1]])] = dist
    del dist
    dist_neighbors, dst = torch.topk(dist_mat, k_neighbors, dim=-1, largest=False)  # [N, topk]

    src = torch.arange(0, N, device=dst.device).unsqueeze(-1).repeat(1, k_neighbors)
    src, dst = src.flatten(), dst.flatten()
    dist_neighbors = dist_neighbors.flatten()
    is_valid = dist_neighbors < BIGINT
    src = src.masked_select(is_valid)
    dst = dst.masked_select(is_valid)

    dst = dst + offsets[batch_id[src]]  # mapping from local to global node index

    edges = torch.stack([src, dst])  # message passed from dst to src

    return edges  # [2, E]
    
class BatchEdgeConstructor:
    '''
    Construct intra-segment edges (intra_edges) and inter-segment edges (inter_edges) with O(Nn) complexity,
    where n is the largest number of nodes of one graph in the batch.
    Additionally consider global nodes: 
        global nodes will connect to all nodes in its segment (global_normal_edges)
        global nodes will connect to each other regardless of the segments they are in (global_global_edges)
    Additionally consider edges between adjacent nodes in the sequence in the same segment (seq_edges)
    '''

    def __init__(self, global_node_id_vocab=[999], delete_self_loop=True) -> None:
        self.global_node_id_vocab = global_node_id_vocab
        self.delete_self_loop = delete_self_loop

        # buffer
        self._reset_buffer()

    def _reset_buffer(self):
        self.row = None
        self.col = None
        self.row_global = None
        self.col_global = None
        self.row_seg = None
        self.col_seg = None
        self.offsets = None
        self.max_n = None
        self.gni2lni = None
        self.not_global_edges = None
        # torch.cuda.empty_cache()

    def get_batch_edges(self, batch_id):

        # construct tensors to map between global / local node index
        lengths = scatter_sum(torch.ones_like(batch_id), batch_id)  # [bs]
        N, max_n = batch_id.shape[0], torch.max(lengths)
        offsets = F.pad(torch.cumsum(lengths, dim=0)[:-1], pad=(1, 0), value=0)  # [bs]
        # global node index to local index. lni2gni can be implemented as lni + offsets[batch_id]
        gni = torch.arange(N, device=batch_id.device)
        gni2lni = gni - offsets[batch_id]  # [N]

        # all possible edges (within the same graph)
        # same bid (get rid of self-loop and none edges)
        same_bid = torch.zeros(N, max_n, device=batch_id.device)
        same_bid[(gni, lengths[batch_id] - 1)] = 1
        same_bid = 1 - torch.cumsum(same_bid, dim=-1)
        # shift right and pad 1 to the left
        same_bid = F.pad(same_bid[:, :-1], pad=(1, 0), value=1)
        if self.delete_self_loop:
            same_bid[(gni, gni2lni)] = 0  # delete self loop
        row, col = torch.nonzero(same_bid).T  # [2, n_edge_all]
        col = col + offsets[batch_id[row]]  # mapping from local to global node index
        return (row, col), (offsets, max_n, gni2lni)

    def _prepare(self, S, batch_id, segment_ids) -> None:
        (row, col), (offsets, max_n, gni2lni) = self.get_batch_edges(batch_id)

        # not global edges
        if len(self.global_node_id_vocab):
            is_global = sequential_or(*[S == global_node_id for global_node_id in self.global_node_id_vocab]) # [N]
        else:
            is_global = torch.zeros_like(S, dtype=torch.bool)
        row_global, col_global = is_global[row], is_global[col]
        not_global_edges = torch.logical_not(torch.logical_or(row_global, col_global))
        
        # segment ids
        row_seg, col_seg = segment_ids[row], segment_ids[col]

        # add to buffer
        self.row, self.col = row, col
        self.offsets, self.max_n, self.gni2lni = offsets, max_n, gni2lni
        self.row_global, self.col_global = row_global, col_global
        self.not_global_edges = not_global_edges
        self.row_seg, self.col_seg = row_seg, col_seg

    def _construct_intra_edges(self, S, batch_id, segment_ids, **kwargs):
        row, col = self.row, self.col
        # all possible ctx edges: same seg, not global
        select_edges = torch.logical_and(self.row_seg == self.col_seg, self.not_global_edges)
        intra_all_row, intra_all_col = row[select_edges], col[select_edges]
        return torch.stack([intra_all_row, intra_all_col])

    def _construct_inter_edges(self, S, batch_id, segment_ids, **kwargs):
        row, col = self.row, self.col
        # all possible inter edges: not same seg, not global
        select_edges = torch.logical_and(self.row_seg != self.col_seg, self.not_global_edges)
        inter_all_row, inter_all_col = row[select_edges], col[select_edges]
        return torch.stack([inter_all_row, inter_all_col])

    def _construct_global_edges(self, S, batch_id, segment_ids, **kwargs):
        row, col = self.row, self.col
        # edges between global and normal nodes
        select_edges = torch.logical_and(self.row_seg == self.col_seg, torch.logical_not(self.not_global_edges))
        global_normal = torch.stack([row[select_edges], col[select_edges]])  # [2, nE]
        # edges between global and global nodes
        select_edges = torch.logical_and(self.row_global, self.col_global) # self-loop has been deleted
        global_global = torch.stack([row[select_edges], col[select_edges]])  # [2, nE]
        return global_normal, global_global

    @torch.no_grad()
    def __call__(self, S, batch_id, segment_ids, **kwargs):
        '''
        Memory efficient with complexity of O(Nn) where n is the largest number of nodes in the batch
        '''
        # prepare inputs
        self._prepare(S, batch_id, segment_ids)

        # intra-segment edges
        intra_edges = self._construct_intra_edges(S, batch_id, segment_ids, **kwargs)

        # inter-segment edges
        inter_edges = self._construct_inter_edges(S, batch_id, segment_ids, **kwargs)

        # edges between global nodes and normal/global nodes
        global_normal_edges, global_global_edges = self._construct_global_edges(S, batch_id, segment_ids, **kwargs)
        self._reset_buffer()

        return intra_edges, inter_edges, global_normal_edges, global_global_edges
    
class KNNBatchEdgeConstructor(BatchEdgeConstructor):
    def __init__(self, k_neighbors, global_message_passing=True, global_node_id_vocab=[999], delete_self_loop=True) -> None:
        super().__init__(global_node_id_vocab, delete_self_loop)
        self.k_neighbors = k_neighbors
        self.global_message_passing = global_message_passing

    def _construct_intra_edges(self, S, batch_id, segment_ids, **kwargs):
        all_intra_edges = super()._construct_intra_edges(S, batch_id, segment_ids)
        X = kwargs['X']
        # knn
        src_dst = all_intra_edges
        dist = torch.norm(X[src_dst[0]]-X[src_dst[1]],dim=-1)
        intra_edges = _knn_edges(
            dist, src_dst, self.k_neighbors,
            (self.offsets, batch_id, self.max_n, self.gni2lni))
        return intra_edges
    
    def _construct_inter_edges(self, S, batch_id, segment_ids, **kwargs):
        all_inter_edges = super()._construct_inter_edges(S, batch_id, segment_ids)
        X = kwargs['X']
        # knn
        src_dst = all_inter_edges
        dist = torch.norm(X[src_dst[0]]-X[src_dst[1]],dim=-1)
        inter_edges = _knn_edges(
            dist, src_dst, self.k_neighbors,
            (self.offsets, batch_id, self.max_n, self.gni2lni))
        return inter_edges
    
    def _construct_global_edges(self, S, batch_id, segment_ids, **kwargs):
        if self.global_message_passing:
            return super()._construct_global_edges(S, batch_id, segment_ids, **kwargs)
        else:
            return None, None
