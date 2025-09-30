from typing import Optional

import ase
import torch
from einops import rearrange, repeat
from torch import nn
from transformers import PreTrainedModel
import torch.nn.functional as F

from geoformer.datasets import QM9,ASD
from geoformer.model import modeling_priors
from geoformer.model.configuration_geoformer import GeoformerConfig
from geoformer.model.modeling_geoformer_layers import (CosineCutoff,
                                                       ExpNormalSmearing,
                                                       VecLayerNorm,
                                                       act_class_mapping)
from geoformer.model.equiformerv2 import EquiformerV2_OC20
from geoformer.model.EGNN import EGNN
from geoformer.model.SchNet import SchNet
from openpoints.models import build_model_from_cfg
from openpoints.utils import EasyConfig

class GeoformerMultiHeadAttention(nn.Module):
    def __init__(self, config, *args, **kwargs) -> None:
        super(GeoformerMultiHeadAttention, self).__init__(*args, **kwargs)

        self.embedding_dim = config.embedding_dim
        self.num_heads = config.num_attention_heads
        self.head_dim = config.embedding_dim // config.num_attention_heads
        if not (
            self.head_dim * config.num_attention_heads == self.embedding_dim
        ):
            raise AssertionError(
                "The embedding_dim must be divisible by num_heads."
            )

        self.act = act_class_mapping[config.activation_function]()
        self.cutoff = CosineCutoff(config.cutoff)

        self.dropout_module = nn.Dropout(
            p=config.attention_dropout, inplace=False
        )

        self.k_proj = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.q_proj = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.v_proj = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.dk_proj = nn.Linear(self.embedding_dim, self.embedding_dim)

        self.du_update_proj = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.du_norm = VecLayerNorm(
            self.embedding_dim, trainable=False, norm_type=config.norm_type
        )
        self.dihedral_proj = nn.Linear(
            self.embedding_dim, 2 * self.embedding_dim, bias=False
        )
        self.edge_attr_update = nn.Linear(
            self.embedding_dim, self.embedding_dim
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.k_proj.weight)
        self.k_proj.bias.data.fill_(0.0)
        nn.init.xavier_uniform_(self.q_proj.weight)
        self.q_proj.bias.data.fill_(0.0)
        nn.init.xavier_uniform_(self.v_proj.weight)
        self.v_proj.bias.data.fill_(0.0)
        nn.init.xavier_uniform_(self.dk_proj.weight)
        self.dk_proj.bias.data.fill_(0.0)
        nn.init.xavier_uniform_(self.du_update_proj.weight)
        self.du_update_proj.bias.data.fill_(0.0)
        nn.init.xavier_uniform_(self.dihedral_proj.weight)
        nn.init.xavier_uniform_(self.edge_attr_update.weight)
        self.edge_attr_update.bias.data.fill_(0.0)

    def forward(
        self,
        x: torch.Tensor,  # (B, N, F)
        vec: Optional[torch.Tensor],  # (B, N, N, 3)
        dist: Optional[torch.Tensor],  # (B, N, N)
        edge_attr: Optional[torch.Tensor],  # (B, N, N, F)
        key_padding_mask: Optional[torch.Tensor] = None,  # (B, N)
        **kwargs,
    ):
        q = rearrange(
            self.q_proj(x), "b n (h d) -> (b h) n d", h=self.num_heads
        )  # (BH, N, D)
        k = rearrange(
            self.k_proj(x), "b n (h d) -> (b h) n d", h=self.num_heads
        )  # (BH, N, D)
        v = rearrange(
            self.v_proj(x), "b n (h d) -> (b h) n d", h=self.num_heads
        )  # (BH, N, D)
        dk = rearrange(
            self.act(self.dk_proj(edge_attr)),
            "b n m (h d) -> (b h) n m d",
            h=self.num_heads,
        )  # (BH, N, N, D)

        attn_weights = ((q.unsqueeze(-2) * k.unsqueeze(-3)) * dk).sum(
            dim=-1
        )  # (BH, N, N)

        if key_padding_mask is not None:
            attn_weights = rearrange(
                attn_weights, "(b h) n m -> b h n m", h=self.num_heads
            )
            attn_weights = attn_weights.masked_fill(
                rearrange(key_padding_mask, "b n m -> b () n m"),
                0.0,
            )
            attn_weights = rearrange(attn_weights, "b h n m -> (b h) n m")

        attn_scale = repeat(
            self.cutoff(dist), "b n m -> b h n m", h=self.num_heads
        )  # (BH, N, N)
        attn_scale = rearrange(
            attn_scale, "b h n m -> (b h) n m", h=self.num_heads
        )  # (BH, N, N)
        attn_probs = self.act(attn_weights) * attn_scale  # (BH, N, N)

        attn_per_nodes = attn_probs.unsqueeze(-1) * v.unsqueeze(
            -3
        )  # (BH, N, N, D)
        attn_per_nodes = rearrange(
            attn_per_nodes, "(b h) n m d -> b n m (h d)", h=self.num_heads
        )  # (B, N, N, F)
        attn = attn_per_nodes.sum(dim=2)  # (B, N, F)

        du = (
            self.du_update_proj(attn_per_nodes)
            .masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
            .unsqueeze(-2)
            * vec.unsqueeze(-1)
        ).sum(
            dim=-3
        )  # (B, N, 3, F)
        du = self.du_norm(du)  # (B, N, 3, F)
        ws, wt = torch.split(
            self.dihedral_proj(du), self.embedding_dim, dim=-1
        )  # (B, N, 3, F)
        ipe = (wt.unsqueeze(1) * ws.unsqueeze(2)).sum(dim=-2)  # (B, N, N, F)
        ipe = self.act(self.edge_attr_update(edge_attr)) * ipe  # (B, N, N, F)

        return attn, ipe


class GeoformerAttnBlock(nn.Module):
    def __init__(self, config, *args, **kwargs) -> None:
        super(GeoformerAttnBlock, self).__init__(*args, **kwargs)

        self.embedding_dim = config.embedding_dim
        self.dropout_module = nn.Dropout(p=config.dropout, inplace=False)

        self.act = act_class_mapping[config.activation_function]()

        self.self_attn = GeoformerMultiHeadAttention(config)

        self.ffn = nn.Sequential(
            nn.Linear(self.embedding_dim, config.ffn_embedding_dim),
            self.act,
            nn.Dropout(p=config.activation_dropout, inplace=False),
            nn.Linear(config.ffn_embedding_dim, self.embedding_dim),
        )

        self.attn_layer_norm = nn.LayerNorm(self.embedding_dim)
        self.final_layer_norm = nn.LayerNorm(self.embedding_dim)

        self.reset_parameters()

    def reset_parameters(self):
        self.self_attn.reset_parameters()
        nn.init.xavier_uniform_(self.ffn[0].weight)
        self.ffn[0].bias.data.fill_(0.0)
        nn.init.xavier_uniform_(self.ffn[3].weight)
        self.ffn[3].bias.data.fill_(0.0)
        self.attn_layer_norm.reset_parameters()
        self.final_layer_norm.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,  # (B, N, F)
        vec: torch.Tensor,  # (B, N, N, 3)
        dist: torch.Tensor,  # (B, N, N)
        edge_attr: torch.Tensor,  # (B, N, N, ?)
        key_padding_mask: Optional[
            torch.Tensor
        ],  # [padding, cutoff] (B, N, N)
        **kwargs,
    ):
        # attention
        dx, dedge_attr = x, edge_attr
        x, edge_attr = self.self_attn(
            x=x,
            vec=vec,
            dist=dist,
            edge_attr=edge_attr,
            key_padding_mask=key_padding_mask,
        )

        x = self.dropout_module(x)
        x = x + dx
        x = self.attn_layer_norm(x)

        # ipe update
        edge_attr = edge_attr + dedge_attr

        # ffn
        dx = x
        x = self.ffn(x)
        x = self.dropout_module(x)
        x = x + dx

        x = self.final_layer_norm(x)

        return x, edge_attr


class GeoformerEncoder(nn.Module):
    def __init__(self, config, *args, **kwargs) -> None:
        super(GeoformerEncoder, self).__init__(*args, **kwargs)

        self.pad_token_id = config.pad_token_id
        self.embedding_dim = config.embedding_dim
        self.cutoff = config.cutoff

        self.embedding = nn.Embedding(
            config.max_z, self.embedding_dim, padding_idx=self.pad_token_id
        )
        self.distance_expansion = ExpNormalSmearing(
            cutoff=config.cutoff,
            num_rbf=config.num_rbf,
            trainable=config.rbf_trainable,
        )
        self.dist_proj = nn.Linear(config.num_rbf, self.embedding_dim)
        self.act = act_class_mapping[config.activation_function]()

        self.layers = nn.ModuleList(
            [GeoformerAttnBlock(config) for _ in range(config.num_layers)]
        )

        self.x_in_layernorm = nn.LayerNorm(self.embedding_dim)

        self.reset_parameters()

    def reset_parameters(self):
        self.embedding.reset_parameters()
        self.distance_expansion.reset_parameters()
        nn.init.xavier_uniform_(self.dist_proj.weight)
        self.dist_proj.bias.data.fill_(0.0)
        for layer in self.layers:
            layer.reset_parameters()
        self.x_in_layernorm.reset_parameters()

    def forward(
        self,
        z: torch.Tensor,  # (B, N)
        pos: torch.Tensor,  # (B, N, 3)
        **kwargs,
    ):
        B, N, *_ = z.shape
        # generate mask
        padding_mask = z == self.pad_token_id  # (B, N)
        pos_mask = ~(
            padding_mask.unsqueeze(1) | padding_mask.unsqueeze(2)
        )  # (B, N, N)
        dist = torch.norm(
            pos.unsqueeze(1) - pos.unsqueeze(2), dim=-1
        )  # (B, N, N)
        loop_mask = torch.eye(N, dtype=torch.bool, device=dist.device)
        loop_mask = repeat(loop_mask, "n m -> b n m", b=B)  # (B, N, N)
        dist = dist.masked_fill(loop_mask, 0.0)  # (B, N, N)
        adj_mask = (dist < self.cutoff) & pos_mask  # (B, N, N)
        loop_adj_mask = ~loop_mask & adj_mask  # (B, N, N)

        vec = (pos.unsqueeze(1) - pos.unsqueeze(2)) / (
            dist.unsqueeze(-1) + 1e-8
        )  # (B, N, N, 3)
        vec = vec.masked_fill(
            ~loop_adj_mask.unsqueeze(-1), 0.0
        )  # (B, N, N, 3)

        key_padding_mask = (
            (~adj_mask)
            .masked_fill(padding_mask.unsqueeze(-1), False)
            .masked_fill(padding_mask.unsqueeze(-2), True)
        )

        x = self.embedding(z)  # (B, N, F)
        x = self.x_in_layernorm(x)
        edge_attr = self.distance_expansion(dist)  # (B, N, N, num_rbf)
        edge_attr = self.act(self.dist_proj(edge_attr))  # (B, N, N, F)
        edge_attr = edge_attr.masked_fill(
            ~adj_mask.unsqueeze(-1), 0.0
        )  # (B, N, N, F)

        for layer in self.layers:
            x, edge_attr = layer(
                x=x,
                vec=vec,
                dist=dist,
                edge_attr=edge_attr,
                key_padding_mask=key_padding_mask,
            )

        return x, edge_attr


class GeoformerScalarDecoder(nn.Module):
    def __init__(self, config, *args, **kwargs) -> None:
        super(GeoformerScalarDecoder, self).__init__(*args, **kwargs)

        self.embedding_dim = config.embedding_dim
        self.num_classes = config.num_classes
        self.act = act_class_mapping[config.activation_function]()
        self.classifier = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim // 2),
            self.act,
            nn.Linear(self.embedding_dim // 2, self.num_classes),
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.classifier[0].weight)
        self.classifier[0].bias.data.fill_(0.0)
        nn.init.xavier_uniform_(self.classifier[2].weight)
        self.classifier[2].bias.data.fill_(0.0)

    def forward(
        self,
        x: torch.Tensor,  # (B, N, F)
        edge_attr: torch.Tensor,  # (B, N, N, F)
        **kwargs,
    ):
        return self.classifier(x) + edge_attr.sum() * 0


class GeoformerDipoleMomentDecoder(GeoformerScalarDecoder):
    def __init__(self, config, *args, **kwargs):
        super(GeoformerDipoleMomentDecoder, self).__init__(
            config, *args, **kwargs
        )
        atomic_mass = torch.from_numpy(ase.data.atomic_masses).float()
        self.register_buffer("atomic_mass", atomic_mass)

    def forward(
        self,
        x: torch.Tensor,  # (B, N, F)
        edge_attr: torch.Tensor,  # (B, N, N, F)
        **kwargs,
    ):
        x = self.classifier(x) + edge_attr.sum() * 0  # (B, N, 1)

        # Get center of mass.
        z = kwargs["z"]  # (B, N)
        pos = kwargs["pos"]  # (B, N, 3)
        padding_mask = kwargs["padding_mask"]  # (B, N)
        mass = (
            self.atomic_mass[z].masked_fill(padding_mask, 0.0).unsqueeze(-1)
        )  # (B, N, 1)
        c = torch.sum(mass * pos, dim=-2) / torch.sum(mass, dim=-2)
        x = x * (pos - c.unsqueeze(-2))
        return x  # (B, N, 3)


class GeoformerElectronicSpatialExtentDecoder(GeoformerScalarDecoder):
    def __init__(self, config, *args, **kwargs):
        super(GeoformerElectronicSpatialExtentDecoder, self).__init__(
            config, *args, **kwargs
        )
        atomic_mass = torch.from_numpy(ase.data.atomic_masses).float()
        self.register_buffer("atomic_mass", atomic_mass)

    def forward(
        self,
        x: torch.Tensor,  # (B, N, F)
        edge_attr: torch.Tensor,  # (B, N, N, F)
        **kwargs,
    ):
        x = self.classifier(x) + edge_attr.sum() * 0  # (B, N, 1)

        # Get center of mass.
        z = kwargs["z"]  # (B, N)
        pos = kwargs["pos"]  # (B, N, 3)
        padding_mask = kwargs["padding_mask"]  # (B, N)
        mass = (
            self.atomic_mass[z].masked_fill(padding_mask, 0.0).unsqueeze(-1)
        )  # (B, N, 1)
        c = torch.sum(mass * pos, dim=-2) / torch.sum(mass, dim=-2)
        x = torch.norm(pos - c.unsqueeze(-2), dim=-1, keepdim=True) ** 2 * x
        return x  # (B, N, 1)


class GeoformerModel(PreTrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        super(GeoformerModel, self).__init__(config, *inputs, **kwargs)

        self.geo_encoder = GeoformerEncoder(config)
        if config.decoder_type == "scalar":
            self.geo_decoder = GeoformerScalarDecoder(config)
        elif config.decoder_type == "dipole_moment":
            self.geo_decoder = GeoformerDipoleMomentDecoder(config)
        elif config.decoder_type == "electronic_spatial_extent":
            self.geo_decoder = GeoformerElectronicSpatialExtentDecoder(config)
        else:
            raise ValueError(f"Unknown decoder type: {config.decoder_type}")

        self.post_init()

    def init_weights(self):
        self.geo_encoder.reset_parameters()
        self.geo_decoder.reset_parameters()


class GeoformerForEnergyRegression(GeoformerModel):
    def __init__(self, config, *inputs, **kwargs):
        super(GeoformerForEnergyRegression, self).__init__(
            config, *inputs, **kwargs
        )

        self.config = config
        self.aggr = config.aggr
        self.pad_token_id = config.pad_token_id
        # self.prior_model = self._register_prior_model()
        self.prior_model = None
        mean = torch.scalar_tensor(0) if config.mean is None else config.mean
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean).float()
        self.register_buffer("mean", mean)
        std = torch.scalar_tensor(1) if config.std is None else config.std
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std).float()
        self.register_buffer("std", std)

    # def _register_prior_model(self):
    #     prior_model = None
    #     if self.config.prior_model is not None:
    #         assert hasattr(modeling_priors, self.config.prior_model), (
    #             f"Unknown prior model {self.config.prior_model}. "
    #             f"Available models are {', '.join(modeling_priors.__all__)}"
    #         )
    #         # initialize the prior model
    #         prior_model = getattr(modeling_priors, self.config.prior_model)(
    #             utils=ASD(
    #                 root=self.config.dataset_root
    #             )
    #         )
    #     return prior_model

    def forward(
        self,
        data,
        # z: torch.Tensor,  # (B, N)
        # pos: torch.Tensor,  # (B, N, 3)
        **kwargs,
    ):
        z, pos = data['z'], data['pos']

        x, edge_attr = self.geo_encoder(z=z, pos=pos)

        padding_mask = z == self.pad_token_id  # (B, N)

        # (B, N, 1) or (B, N, 3)
        x = self.geo_decoder(
            x=x, edge_attr=edge_attr, z=z, pos=pos, padding_mask=padding_mask
        )

        logits = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)  # (B, N, 1)

        if self.std is not None:
            logits = logits * self.std

        logits = (
            self.prior_model(logits, z)
            if self.prior_model is not None
            else logits
        )

        if self.aggr == "sum":
            logits = logits.sum(dim=1)  # (B, 1)
        elif self.aggr == "mean":
            logits = logits.sum(dim=1) / (~padding_mask).sum(dim=-1).unsqueeze(
                -1
            )  # (B, 1)
        else:
            NotImplementedError(f"Unknown aggregation method: {self.aggr}")

        if self.config.decoder_type == "dipole_moment":
            logits = torch.norm(logits, dim=-1, keepdim=True)

        if self.mean is not None:
            logits = logits + self.mean

        return logits
    
class Ranking(nn.Module):
    def __init__(self, config, *args, **kwargs) -> None:
        super(Ranking, self).__init__(*args, **kwargs)
        cfg = EasyConfig()
        cfg.load(config.point_model, recursive=True)
        if config.encoder_name == 'Equiformer':
            self.graph_encoder = EquiformerV2_OC20(num_classes = config.num_classes)
        else:
            model_config = GeoformerConfig(
                max_z=config.max_z,
                embedding_dim=config.embedding_dim,
                ffn_embedding_dim=config.ffn_embedding_dim,
                num_layers=config.num_layers,
                num_attention_heads=config.num_heads, 
                cutoff=config.cutoff,
                num_rbf=config.num_rbf,
                rbf_trainable=config.trainable_rbf,
                norm_type=config.norm_type,
                dropout=config.dropout,
                attention_dropout=config.attention_dropout,
                activation_dropout=config.activation_dropout,
                activation_function=config.activation_function,
                decoder_type=config.decoder_type,
                aggr=config.aggr,
                dataset_root=config.dataset_root,
                dataset_arg=config.dataset_arg,
                mean=config.mean,
                std=config.std,
                prior_model=config.prior_model,
                num_classes=config.num_classes,
                pad_token_id=config.pad_token_id,
            )

            self.graph_encoder = GeoformerForEnergyRegression(config=model_config)
        
        cfg.model.cls_args.num_classes = config.num_classes
        self.point_encoder = build_model_from_cfg(cfg.model)
        self.loss_func = SupConLoss()
#         self.reset_parameters()

#     def reset_parameters(self):
#         nn.init.xavier_uniform_(self.k_proj.weight)
#         self.k_proj.bias.data.fill_(0.0)
#         nn.init.xavier_uniform_(self.q_proj.weight)
#         self.q_proj.bias.data.fill_(0.0)
#         nn.init.xavier_uniform_(self.v_proj.weight)
#         self.v_proj.bias.data.fill_(0.0)
#         nn.init.xavier_uniform_(self.dk_proj.weight)
#         self.dk_proj.bias.data.fill_(0.0)
#         nn.init.xavier_uniform_(self.du_update_proj.weight)
#         self.du_update_proj.bias.data.fill_(0.0)
#         nn.init.xavier_uniform_(self.dihedral_proj.weight)
#         nn.init.xavier_uniform_(self.edge_attr_update.weight)
#         self.edge_attr_update.bias.data.fill_(0.0)

    def forward(
        self,data
    ):
        
        feats_graph = self.graph_encoder(data['graph'])
        data_point = {}
        data_point['pos'] = data['point'][:, :, :3].contiguous()
        data_point['x'] = data['point'][:, :, :4].transpose(1, 2).contiguous()
        feats_point = self.point_encoder.get_logits_loss(data_point)
#         print(feats_graph.shape,feats_point.shape)
        loss = self.loss_func(feats_graph,feats_point)
        return loss, feats_graph, feats_point
    
    
    
class SupConLoss(nn.Module):
    '''paper can download from https://arxiv.org/abs/2103.12953
    '''
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, *views, labels=None, mask=None):
        """If both `labels` and `mask` are None, it degenerates to SimCLR unsupervised loss: https://arxiv.org/pdf/2002.05709.pdf.
        Args:
            views: view1, view2, ..., their shape must be same.
            labels: ground truth of shape [bsz, 1].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        features = self.get_forward_features_from_multi_view(*views)  # shape [bsz, n_views, ...].

        device = (features.device if features.is_cuda else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...], at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T), self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()  # [负无穷, 0]

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        # loss = - mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

    def get_forward_features_from_multi_view(self, *views):
        '''

        :param views: the shape for each view must be same: [batch, h_dim]
        :return: features, which can be input to function forward().
        '''
        assert len(views) >= 2, "the number of view must >= 2."
        for idx in range(len(views)):
            assert views[0].shape == views[idx].shape
        features = []
        for idx in range(len(views)):
            features.append(F.normalize(views[idx], dim=1).unsqueeze(1))
        features = torch.cat(features, dim=1)
        return features


def create_model(config) -> GeoformerForEnergyRegression:
    model_config = GeoformerConfig(
        max_z=config.max_z,
        embedding_dim=config.embedding_dim,
        ffn_embedding_dim=config.ffn_embedding_dim,
        num_layers=config.num_layers,
        num_attention_heads=config.num_heads, 
        cutoff=config.cutoff,
        num_rbf=config.num_rbf,
        rbf_trainable=config.trainable_rbf,
        norm_type=config.norm_type,
        dropout=config.dropout,
        attention_dropout=config.attention_dropout,
        activation_dropout=config.activation_dropout,
        activation_function=config.activation_function,
        decoder_type=config.decoder_type,
        aggr=config.aggr,
        dataset_root=config.dataset_root,
        dataset_arg=config.dataset_arg,
        mean=config.mean,
        std=config.std,
        prior_model=config.prior_model,
        num_classes=config.num_classes,
        pad_token_id=config.pad_token_id,
    )
    
    return Ranking(config=config)
#     if config.encoder_name == 'EGNN':
#         return EGNN()
#     elif config.encoder_name == 'Equiformer':
#         return EquiformerV2_OC20(num_classes = config.num_classes)
#     elif config.encoder_name == 'Geoformer':
#         return GeoformerForEnergyRegression(config=model_config)
