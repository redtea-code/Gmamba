import torch
from torch.nn import (
    BatchNorm1d,
    Embedding,
    Linear,
    ModuleList,
    ReLU,
    Sequential,
)
from torch.optim.lr_scheduler import ReduceLROnPlateau

import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, global_add_pool
import inspect
from typing import Any, Dict, Optional

import torch.nn.functional as F
from torch import Tensor
from torch.nn import Dropout, Linear, Sequential

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import reset
from torch_geometric.nn.resolver import (
    activation_resolver,
    normalization_resolver,
)
from torch_geometric.typing import Adj
from torch_geometric.utils import to_dense_batch

# from mamba_ssm import Mamba

from torch_geometric.utils import degree, sort_edge_index

from cross_atten.mamba import MambaConfig, Mamba


def permute_within_batch(x, batch):
    # Enumerate over unique batch indices
    unique_batches = torch.unique(batch)

    # Initialize list to store permuted indices
    permuted_indices = []

    for batch_index in unique_batches:
        # Extract indices for the current batch
        indices_in_batch = (batch == batch_index).nonzero().squeeze()

        # Permute indices within the current batch
        permuted_indices_in_batch = indices_in_batch[torch.randperm(len(indices_in_batch))]

        # Append permuted indices to the list
        permuted_indices.append(permuted_indices_in_batch)

    # Concatenate permuted indices into a single tensor
    permuted_indices = torch.cat(permuted_indices)

    return permuted_indices


class GPSConv(torch.nn.Module):
    def __init__(
            self,
            channels: int,
            conv: Optional[MessagePassing],
            heads: int = 1,
            dropout: float = 0.0,
            attn_dropout: float = 0.0,
            act: str = 'relu',
            att_type: str = 'transformer',
            order_by_degree: bool = False,
            shuffle_ind: int = 0,
            d_state: int = 16,
            d_conv: int = 4,
            act_kwargs: Optional[Dict[str, Any]] = None,
            norm: Optional[str] = 'batch_norm',
            norm_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()

        self.channels = channels
        self.conv = conv
        self.heads = heads
        self.dropout = dropout
        self.att_type = att_type
        self.shuffle_ind = shuffle_ind
        self.order_by_degree = order_by_degree

        assert (self.order_by_degree == True and self.shuffle_ind == 0) or (
                self.order_by_degree == False), f'order_by_degree={self.order_by_degree} and shuffle_ind={self.shuffle_ind}'

        if self.att_type == 'mamba':
            config = MambaConfig(d_model=channels, n_layers=1, use_cuda=True)
            self.self_attn = Mamba(config)
            # self.self_attn = Mamba(
            #     d_model=channels,
            #     d_state=d_state,
            #     d_conv=d_conv,
            #     expand=1
            # )

        self.mlp = Sequential(
            Linear(channels, channels * 2),
            activation_resolver(act, **(act_kwargs or {})),
            Dropout(dropout),
            Linear(channels * 2, channels),
            Dropout(dropout),
        )

        norm_kwargs = norm_kwargs or {}
        self.norm1 = normalization_resolver(norm, channels, **norm_kwargs)
        self.norm2 = normalization_resolver(norm, channels, **norm_kwargs)
        self.norm3 = normalization_resolver(norm, channels, **norm_kwargs)

        self.norm_with_batch = False
        if self.norm1 is not None:
            signature = inspect.signature(self.norm1.forward)
            self.norm_with_batch = 'batch' in signature.parameters

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        if self.conv is not None:
            self.conv.reset_parameters()
        self.attn._reset_parameters()
        reset(self.mlp)
        if self.norm1 is not None:
            self.norm1.reset_parameters()
        if self.norm2 is not None:
            self.norm2.reset_parameters()
        if self.norm3 is not None:
            self.norm3.reset_parameters()

    def forward(
            self,
            x: Tensor,
            edge_index: Adj,
            batch: Optional[torch.Tensor] = None,
            **kwargs,
    ) -> Tensor:
        r"""Runs the forward pass of the module."""
        hs = []
        if self.conv is not None:  # Local MPNN.
            h = self.conv(x, edge_index, **kwargs)
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = h + x
            if self.norm1 is not None:
                if self.norm_with_batch:
                    h = self.norm1(h, batch=batch)
                else:
                    h = self.norm1(h)
            hs.append(h)
        # ------------ Mamba全局注意力计算 ------------
        if self.att_type == 'mamba':
            # 如果启用按节点度排序（提升重要节点的上下文访问）
            if self.order_by_degree:
                # 计算每个节点的度（基于边索引）
                deg = degree(edge_index[0], x.shape[0]).to(torch.long)
                # 按 [batch, degree] 排序节点（重要节点排在序列末尾）
                order_tensor = torch.stack([batch, deg], 1).T
                _, x = sort_edge_index(order_tensor, edge_attr=x)

            # 单次排列推理（默认模式）
            if self.shuffle_ind == 0:
                # 将稀疏图数据转换为密集批次格式（处理变长序列）
                h, mask = to_dense_batch(x, batch)
                # Mamba处理密集序列后恢复稀疏格式
                h = self.self_attn(h)[mask]
            # 多次排列平均（增强稳定性）
            else:
                mamba_arr = []
                for _ in range(self.shuffle_ind):
                    # 在批次内随机排列节点顺序（减少顺序偏差）
                    h_ind_perm = permute_within_batch(x, batch)
                    # 转换为密集格式并通过Mamba处理
                    h_i, mask = to_dense_batch(x[h_ind_perm], batch)
                    h_i = self.self_attn(h_i)[mask][h_ind_perm]  # 恢复原始顺序
                    mamba_arr.append(h_i)
                # 对多次排列结果取平均
                h = sum(mamba_arr) / self.shuffle_ind

        # ------------ 残差连接与归一化 ------------
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = h + x  # 残差连接：Mamba输出与原始输入相加
        if self.norm2 is not None:
            if self.norm_with_batch:
                h = self.norm2(h, batch=batch)
            else:
                h = self.norm2(h)
        hs.append(h)  # 保存全局Mamba的输出

        # ------------ 合并局部和全局输出 ------------
        out = sum(hs)  # 简单相加（或可自定义加权方式）

        # ------------ 最终MLP处理 ------------
        if self.norm3 is not None:
            if self.norm_with_batch:
                out = self.norm3(out, batch=batch)
            else:
                out = self.norm3(out)

        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.channels}, '
                f'conv={self.conv}, heads={self.heads})')


class Graphblock(torch.nn.Module):
    def __init__(self, channels: int, num_layers: int, model_type: str, shuffle_ind: int, d_state: int,
                 d_conv: int, order_by_degree: False, if_pool=False):
        super().__init__()

        self.model_type = model_type
        self.shuffle_ind = shuffle_ind
        self.order_by_degree = order_by_degree
        self.if_pool = if_pool
        self.convs = ModuleList()
        for _ in range(num_layers):
            nn = Sequential(
                Linear(channels, channels),
                ReLU(),
                Linear(channels, channels),
            )
            if self.model_type == 'gine':
                conv = GINEConv(nn)

            if self.model_type == 'mamba':
                # 输入先经过第二个参数(GINEConv(nn))再进入mamba
                conv = GPSConv(channels, GINEConv(nn), heads=4, attn_dropout=0.5,
                               att_type='mamba',
                               shuffle_ind=self.shuffle_ind,
                               order_by_degree=self.order_by_degree,
                               d_state=d_state, d_conv=d_conv)
            if self.model_type == "only_mamba":
                conv = GPSConv(channels, None, heads=4, attn_dropout=0.5,
                               att_type='mamba',
                               shuffle_ind=self.shuffle_ind,
                               order_by_degree=self.order_by_degree,
                               d_state=d_state, d_conv=d_conv)
            self.convs.append(conv)

    def forward(self, x,  edge_index, edge_attr, batch):

        for conv in self.convs:
            if self.model_type == 'gine':
                x = conv(x, edge_index, edge_attr=edge_attr)
            else:
                x = conv(x, edge_index, batch, edge_attr=edge_attr)
        if self.if_pool:
            x = global_add_pool(x, batch)
        return x

class GraphModel(torch.nn.Module):
    def __init__(self, channels: int, pe_dim: int, num_layers: int, model_type: str, shuffle_ind: int, d_state: int,
                 d_conv: int, order_by_degree: False, node_dim: int=48, edge_dim: int=1,if_pool=False):
        super().__init__()

        self.node_emb = Linear(node_dim, channels - pe_dim)
        self.pe_lin = Linear(20, pe_dim)
        self.pe_norm = BatchNorm1d(20)
        self.edge_emb = Linear(edge_dim, channels)

        self.if_pool = if_pool
        self.model_type = model_type
        self.shuffle_ind = shuffle_ind
        self.order_by_degree = order_by_degree

        self.convs = ModuleList()
        for _ in range(num_layers):
            nn = Sequential(
                Linear(channels, channels),
                ReLU(),
                Linear(channels, channels),
            )
            if self.model_type == 'gine':
                conv = GINEConv(nn)

            if self.model_type == 'mamba':
                # 输入先经过第二个参数(GINEConv(nn))再进入mamba
                conv = GPSConv(channels, GINEConv(nn), heads=4, attn_dropout=0.5,
                               att_type='mamba',
                               shuffle_ind=self.shuffle_ind,
                               order_by_degree=self.order_by_degree,
                               d_state=d_state, d_conv=d_conv)
            if self.model_type == "only_mamba":
                conv = GPSConv(channels, None, heads=4, attn_dropout=0.5,
                               att_type='mamba',
                               shuffle_ind=self.shuffle_ind,
                               order_by_degree=self.order_by_degree,
                               d_state=d_state, d_conv=d_conv)
            self.convs.append(conv)

    def forward(self, x, pe, edge_index, edge_attr, batch):
        x_pe = self.pe_norm(pe)
        x = torch.cat((self.node_emb(x.squeeze(-1)), self.pe_lin(x_pe)), 1)
        edge_attr = self.edge_emb(edge_attr)

        for conv in self.convs:
            if self.model_type == 'gine':
                x = conv(x, edge_index, edge_attr=edge_attr)
            else:
                x = conv(x, edge_index, batch, edge_attr=edge_attr)
        if self.if_pool:
            x = global_add_pool(x, batch)
        return x


if __name__ == '__main__':
    from torch_geometric.data import Data
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = GraphModel(channels=64, pe_dim=8, num_layers=10,
                       model_type='gine',
                       shuffle_ind=0, order_by_degree=True,
                       d_conv=4, d_state=16,
                       ).to(device)
    # x = torch.randint(0,27,(256, 48))  # 随机生成256个节点，每个节点有48个特征
    x = torch.rand(256,48).to(device)
    edge_index = torch.randint(0, 256, (2, 1860)).to(device)  # 随机生成1860条边，节点索引范围 [0, 255]
    # edge_attr = torch.randint(0,4,(1860, 5))  # 假设边属性是一个5维特征的张量（如果没有，可以设置为 None）
    edge_attr = torch.rand(1860, 1).to(device)
    batch = torch.zeros(x.size(0), dtype=torch.long).to(device)  # 所有256个节点属于一个图

    transform = T.AddRandomWalkPE(walk_length=20, attr_name='pe')
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)
    data = transform(data).to(device)  # pe(node,20)
    out = model(data.x, data.pe, data.edge_index, data.edge_attr,
                data.batch)
    print("ok")