import torch_geometric.transforms as T
# from torch import Tensor
# from torch.nn import Dropout, Linear, Sequential
#
# from torch_geometric.nn.conv import MessagePassing
# from torch_geometric.nn.inits import reset
# from mamba_ssm import Mamba
from torch_geometric.utils import degree, sort_edge_index
import torch
import torch.nn.functional as F
from torch import nn, einsum

from Graph_Mamba.CNN_BLOCK import CNN_Fusion
from Graph_Mamba.RESNET_3d import ResNet_3d, Bottleneck
from Graph_Mamba.graph_construct import build_graph_from_img
from Graph_Mamba.graph_mamba import GraphModel, Graphblock
from Graph_Mamba.src.test import GCondNet
# import sys;sys.path.append('./')
from cross_atten.sd_cross_atten import CrossAttention
from einops import rearrange, repeat
from cross_atten.corss_ft_transformer import Attention, FeedForward, GEGLU, NumericalEmbedder


class Graph_mamba_(nn.Module):
    def __init__(
            self,
            *,
            categories,
            num_continuous,
            dim,
            depth,
            heads,
            dim_out=1,
            num_special_tokens=2,
            cross_ff_multi=2,
            cross_ff_dropout=0.1,
    ):
        super().__init__()
        assert all(map(lambda n: n > 0, categories)), 'number of each category must be positive'
        assert len(categories) + num_continuous > 0, 'input shape must not be null'

        self.num_categories = len(categories)
        self.num_unique_categories = sum(categories)

        # create category embeddings table
        self.num_special_tokens = num_special_tokens
        total_tokens = self.num_unique_categories + num_special_tokens

        # for automatically offsetting unique category ids to the correct position in the categories embedding table
        if self.num_unique_categories > 0:
            categories_offset = F.pad(torch.tensor(list(categories)), (1, 0), value=num_special_tokens)
            categories_offset = categories_offset.cumsum(dim=-1)[:-1]
            self.register_buffer('categories_offset', categories_offset)

            self.categorical_embeds = nn.Embedding(total_tokens, dim)

        # continuous
        self.num_continuous = num_continuous
        if self.num_continuous > 0:
            self.numerical_embedder = NumericalEmbedder(dim, self.num_continuous)

        self.mri_conv = torch.nn.Conv3d(in_channels=1, out_channels=1, kernel_size=(16, 16, 8), stride=(16, 16, 8))
        self.pet_conv = torch.nn.Conv3d(in_channels=1, out_channels=1, kernel_size=(16, 16, 8), stride=(16, 16, 8))

        self.mri_mamba = GraphModel(channels=64, pe_dim=8, num_layers=10,
                                    model_type='mamba',
                                    shuffle_ind=0, order_by_degree=True,
                                    d_conv=4, d_state=16, node_dim=16, edge_dim=436
                                    )
        self.pet_mamba = GraphModel(channels=64, pe_dim=8, num_layers=10,
                                    model_type='mamba',
                                    shuffle_ind=0, order_by_degree=True,
                                    d_conv=4, d_state=16, node_dim=16, edge_dim=436
                                    )
        self.final_cross = CrossAttention(n_heads=heads, d_embed=dim, d_cross=128)
        self.final_feed = FeedForward(dim, mult=cross_ff_multi, dropout=cross_ff_dropout)

        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim_out)
        )

    def forward(self, x_categ, x_numer, image_condition=None):
        assert x_categ.shape[
                   -1] == self.num_categories, f'you must pass in {self.num_categories} values for your categories input'
        if image_condition != None:
            transform = T.AddRandomWalkPE(walk_length=20, attr_name='pe')

            mri_condition = image_condition[0]
            mri_condition = self.mri_conv(mri_condition)
            mri_condition = rearrange(mri_condition, 'b c h w d -> b c (h w) d').contiguous()
            mri_condition = transform(build_graph_from_img(mri_condition)).cuda()

            pet_condition = image_condition[1]
            pet_condition = self.pet_conv(pet_condition)
            pet_condition = rearrange(pet_condition, 'b c h w d -> b c (h w) d').contiguous()
            pet_condition = transform(build_graph_from_img(pet_condition)).cuda()

            mri_condition = self.mri_mamba(mri_condition.x, mri_condition.pe, mri_condition.edge_index,
                                           mri_condition.edge_attr,
                                           mri_condition.batch)
            pet_condition = self.pet_mamba(pet_condition.x, pet_condition.pe, pet_condition.edge_index,
                                           pet_condition.edge_attr,
                                           pet_condition.batch)  # (batch, channel=64)

            whole_condition = torch.cat([mri_condition, pet_condition], dim=1)

        xs = []
        if self.num_unique_categories > 0:
            x_categ = x_categ + self.categories_offset
            x_categ = self.categorical_embeds(x_categ)
            xs.append(x_categ)
        # add numerically embedded tokens
        if self.num_continuous > 0:
            x_numer = self.numerical_embedder(x_numer)
            xs.append(x_numer)
        x = torch.cat(xs, dim=1)
        x = torch.mean(x, dim=1, keepdims=True)  # (batch,1,dim=512)

        x = self.final_cross(x, whole_condition[:, None, :]) + x
        x = self.final_feed(x) + x

        x = x.squeeze(1)  # make less dimension to linear layer

        logits = self.to_logits(x)

        return logits


class Graph_mamba2_(nn.Module):
    """将特征提取的CNN换成了3Dresnet"""

    def __init__(
            self,
            *,
            categories,
            num_continuous,
            dim,
            depth,
            heads,
            dim_out=1,
            num_special_tokens=2,
            cross_ff_multi=2,
            cross_ff_dropout=0.1,
    ):
        super().__init__()
        assert all(map(lambda n: n > 0, categories)), 'number of each category must be positive'
        assert len(categories) + num_continuous > 0, 'input shape must not be null'

        self.num_categories = len(categories)
        self.num_unique_categories = sum(categories)

        self.num_special_tokens = num_special_tokens
        total_tokens = self.num_unique_categories + num_special_tokens
        if self.num_unique_categories > 0:
            categories_offset = F.pad(torch.tensor(list(categories)), (1, 0), value=num_special_tokens)
            categories_offset = categories_offset.cumsum(dim=-1)[:-1]
            self.register_buffer('categories_offset', categories_offset)

            self.categorical_embeds = nn.Embedding(total_tokens, dim)

        self.num_continuous = num_continuous
        if self.num_continuous > 0:
            self.numerical_embedder = NumericalEmbedder(dim, self.num_continuous)

        self.mri_conv = ResNet_3d(Bottleneck, [3, 4, 6, 3])
        self.pet_conv = ResNet_3d(Bottleneck, [3, 4, 6, 3])

        self.mri_mamba = GraphModel(channels=64, pe_dim=8, num_layers=10,
                                    model_type='gine',
                                    shuffle_ind=0, order_by_degree=True,
                                    d_conv=4, d_state=16, node_dim=108 * 16, edge_dim=1
                                    )
        self.pet_mamba = GraphModel(channels=64, pe_dim=8, num_layers=10,
                                    model_type='gine',
                                    shuffle_ind=0, order_by_degree=True,
                                    d_conv=4, d_state=16, node_dim=108 * 16, edge_dim=1
                                    )
        self.final_cross = CrossAttention(n_heads=heads, d_embed=dim, d_cross=128)
        self.final_feed = FeedForward(dim, mult=cross_ff_multi, dropout=cross_ff_dropout)

        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim_out)
        )

    def forward(self, x_categ, x_numer, image_condition=None):
        assert x_categ.shape[
                   -1] == self.num_categories, f'you must pass in {self.num_categories} values for your categories input'
        if image_condition != None:
            transform = T.AddRandomWalkPE(walk_length=20, attr_name='pe')

            mri_condition = image_condition[0]
            mri_condition = self.mri_conv(mri_condition)
            mri_condition = rearrange(mri_condition, 'b c h w d -> b c (h w) d').contiguous()
            mri_condition = transform(build_graph_from_img(mri_condition)).cuda()

            pet_condition = image_condition[1]
            pet_condition = self.pet_conv(pet_condition)
            pet_condition = rearrange(pet_condition, 'b c h w d -> b c (h w) d').contiguous()
            pet_condition = transform(build_graph_from_img(pet_condition)).cuda()

            mri_condition = self.mri_mamba(mri_condition.x, mri_condition.pe, mri_condition.edge_index,
                                           mri_condition.edge_attr,
                                           mri_condition.batch)
            pet_condition = self.pet_mamba(pet_condition.x, pet_condition.pe, pet_condition.edge_index,
                                           pet_condition.edge_attr,
                                           pet_condition.batch)  # (batch, channel=64)

            whole_condition = torch.cat([mri_condition, pet_condition], dim=1)

        xs = []
        if self.num_unique_categories > 0:
            x_categ = x_categ + self.categories_offset
            x_categ = self.categorical_embeds(x_categ)
            xs.append(x_categ)
        # add numerically embedded tokens
        if self.num_continuous > 0:
            x_numer = self.numerical_embedder(x_numer)
            xs.append(x_numer)
        x = torch.cat(xs, dim=1)
        x = torch.mean(x, dim=1, keepdims=True)  # (batch,1,dim=512)

        x = self.final_cross(x, whole_condition[:, None, :]) + x
        x = self.final_feed(x) + x

        x = x.squeeze(1)  # make less dimension to linear layer

        logits = self.to_logits(x)

        return logits


class Graph_mamba3_(nn.Module):
    """去除文本模态"""

    def __init__(
            self,
            *,
            categories,
            num_continuous,
            dim,
            depth,
            heads,
            dim_out=1,
            num_special_tokens=2,
            cross_ff_multi=2,
            cross_ff_dropout=0.1,
    ):
        super().__init__()
        assert all(map(lambda n: n > 0, categories)), 'number of each category must be positive'
        assert len(categories) + num_continuous > 0, 'input shape must not be null'

        self.mri_conv = torch.nn.Conv3d(in_channels=1, out_channels=1, kernel_size=(16, 16, 8), stride=(16, 16, 8))
        self.pet_conv = torch.nn.Conv3d(in_channels=1, out_channels=1, kernel_size=(16, 16, 8), stride=(16, 16, 8))

        self.mri_mamba = GraphModel(channels=64, pe_dim=8, num_layers=depth,
                                    model_type='gine',
                                    shuffle_ind=0, order_by_degree=True,
                                    d_conv=4, d_state=16, node_dim=1 * 16, edge_dim=1
                                    )
        self.pet_mamba = GraphModel(channels=64, pe_dim=8, num_layers=depth,
                                    model_type='gine',
                                    shuffle_ind=0, order_by_degree=True,
                                    d_conv=4, d_state=16, node_dim=1 * 16, edge_dim=1
                                    )

        self.final_feed = FeedForward(dim, mult=cross_ff_multi, dropout=cross_ff_dropout)

        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim_out)
        )

    def forward(self, x_categ, x_numer, image_condition=None):
        if image_condition != None:
            transform = T.AddRandomWalkPE(walk_length=20, attr_name='pe')

            mri_condition = image_condition[0]
            mri_condition = self.mri_conv(mri_condition)
            mri_condition = rearrange(mri_condition, 'b c h w d -> b c (h w) d').contiguous()
            mri_condition = transform(build_graph_from_img(mri_condition)).cuda()

            pet_condition = image_condition[1]
            pet_condition = self.pet_conv(pet_condition)
            pet_condition = rearrange(pet_condition, 'b c h w d -> b c (h w) d').contiguous()
            pet_condition = transform(build_graph_from_img(pet_condition)).cuda()

            mri_condition = self.mri_mamba(mri_condition.x, mri_condition.pe, mri_condition.edge_index,
                                           mri_condition.edge_attr,
                                           mri_condition.batch)
            pet_condition = self.pet_mamba(pet_condition.x, pet_condition.pe, pet_condition.edge_index,
                                           pet_condition.edge_attr,
                                           pet_condition.batch)  # (batch, channel=64)

            whole_condition = torch.cat([mri_condition, pet_condition], dim=1)

        x = self.final_feed(whole_condition) + whole_condition

        x = x.squeeze(1)  # make less dimension to linear layer

        logits = self.to_logits(x)

        return logits


class Graph_mamba4_(nn.Module):
    """使用GCondNet处理文本"""

    def __init__(
            self,
            *,
            categories,
            num_continuous,
            dim,
            depth,
            heads,
            dim_out=1,
            num_special_tokens=2,
            cross_ff_multi=2,
            cross_ff_dropout=0.1,
            **kwargs
    ):
        super().__init__()
        assert all(map(lambda n: n > 0, categories)), 'number of each category must be positive'
        assert len(categories) + num_continuous > 0, 'input shape must not be null'

        self.num_categories = len(categories)
        self.num_unique_categories = sum(categories)

        # create category embeddings table
        self.num_special_tokens = num_special_tokens
        total_tokens = self.num_unique_categories + num_special_tokens

        # for automatically offsetting unique category ids to the correct position in the categories embedding table
        if self.num_unique_categories > 0:
            categories_offset = F.pad(torch.tensor(list(categories)), (1, 0), value=num_special_tokens)
            categories_offset = categories_offset.cumsum(dim=-1)[:-1]
            self.register_buffer('categories_offset', categories_offset)
        # continuous
        self.num_continuous = num_continuous

        self.DNN = GCondNet(kwargs['X'], kwargs['graphs_dataset_all'], dim=[100, 256, 512], layer=2)

        self.mri_conv = torch.nn.Conv3d(in_channels=1, out_channels=1, kernel_size=(16, 16, 8), stride=(16, 16, 8))
        self.pet_conv = torch.nn.Conv3d(in_channels=1, out_channels=1, kernel_size=(16, 16, 8), stride=(16, 16, 8))

        self.mri_mamba = GraphModel(channels=64, pe_dim=8, num_layers=10,
                                    model_type='gine',
                                    shuffle_ind=0, order_by_degree=True,
                                    d_conv=4, d_state=16, node_dim=16, edge_dim=1
                                    )
        self.pet_mamba = GraphModel(channels=64, pe_dim=8, num_layers=10,
                                    model_type='gine',
                                    shuffle_ind=0, order_by_degree=True,
                                    d_conv=4, d_state=16, node_dim=16, edge_dim=1
                                    )
        # 融合策略
        self.final_cross = CrossAttention(n_heads=heads, d_embed=dim, d_cross=128)
        self.final_feed = FeedForward(dim, mult=cross_ff_multi, dropout=cross_ff_dropout)

        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim_out)
        )

    def forward(self, x_categ, x_numer, image_condition=None):
        assert x_categ.shape[
                   -1] == self.num_categories, f'you must pass in {self.num_categories} values for your categories input'
        # 处理MRI和PET模态，并行处理
        transform = T.AddRandomWalkPE(walk_length=20, attr_name='pe')
        mri_condition = image_condition[0]
        mri_condition = self.mri_conv(mri_condition)
        mri_condition = rearrange(mri_condition, 'b c h w d -> b c (h w) d').contiguous()
        mri_condition = transform(build_graph_from_img(mri_condition)).cuda()

        pet_condition = image_condition[1]
        pet_condition = self.pet_conv(pet_condition)
        pet_condition = rearrange(pet_condition, 'b c h w d -> b c (h w) d').contiguous()
        pet_condition = transform(build_graph_from_img(pet_condition)).cuda()

        mri_condition = self.mri_mamba(mri_condition.x, mri_condition.pe, mri_condition.edge_index,
                                       mri_condition.edge_attr,
                                       mri_condition.batch)
        pet_condition = self.pet_mamba(pet_condition.x, pet_condition.pe, pet_condition.edge_index,
                                       pet_condition.edge_attr,
                                       pet_condition.batch)  # (batch, channel=64)

        whole_condition = torch.cat([mri_condition, pet_condition], dim=1)

        # 处理文本模态 GCondNet
        xs = []
        if self.num_unique_categories > 0:
            x_categ = x_categ + self.categories_offset
            # x_categ = self.categorical_embeds(x_categ)
            xs.append(x_categ)
        # add numerically embedded tokens
        if self.num_continuous > 0:
            # x_numer = self.numerical_embedder(x_numer)
            xs.append(x_numer)
        x = torch.cat(xs, dim=1)
        # x = torch.mean(x, dim=1, keepdims=True)  # (batch,1,dim=512)
        x = self.DNN(x)

        # 多模态融合
        x = self.final_cross(x[:, None, :], whole_condition[:, None, :]) + x[:, None, :]
        x = self.final_feed(x) + x

        x = x.squeeze(1)  # make less dimension to linear layer

        logits = self.to_logits(x)

        return logits


class Graph_mamba5_(nn.Module):
    """1.使用GCondNet处理文本 2.增加了dropout 3.交叉融合 """

    def __init__(
            self,
            *,
            categories,
            num_continuous,
            dim,
            depth,
            heads,
            dim_out=1,
            num_special_tokens=2,
            cross_ff_multi=2,
            cross_ff_dropout=0.1,
            dim2=256,
            **kwargs
    ):
        super().__init__()
        assert all(map(lambda n: n > 0, categories)), 'number of each category must be positive'
        assert len(categories) + num_continuous > 0, 'input shape must not be null'

        self.num_categories = len(categories)
        self.num_unique_categories = sum(categories)

        # create category embeddings table
        self.num_special_tokens = num_special_tokens
        # total_tokens = self.num_unique_categories + num_special_tokens

        # for automatically offsetting unique category ids to the correct position in the categories embedding table
        if self.num_unique_categories > 0:
            categories_offset = F.pad(torch.tensor(list(categories)), (1, 0), value=num_special_tokens)
            categories_offset = categories_offset.cumsum(dim=-1)[:-1]
            self.register_buffer('categories_offset', categories_offset)
        # continuous
        self.num_continuous = num_continuous

        self.DNN = GCondNet(kwargs['X'], kwargs['graphs_dataset_all'], dim=[100, 256, 512], layer=2)

        self.mri_conv = torch.nn.Conv3d(in_channels=1, out_channels=1, kernel_size=(16, 16, 8), stride=(16, 16, 8))
        self.pet_conv = torch.nn.Conv3d(in_channels=1, out_channels=1, kernel_size=(16, 16, 8), stride=(16, 16, 8))

        self.mri_mamba = GraphModel(channels=dim2, pe_dim=8, num_layers=depth,
                                    model_type='mamba',
                                    shuffle_ind=0, order_by_degree=True,
                                    d_conv=4, d_state=16, node_dim=16, edge_dim=1
                                    )
        self.pet_mamba = GraphModel(channels=dim2, pe_dim=8, num_layers=depth,
                                    model_type='mamba',
                                    shuffle_ind=0, order_by_degree=True,
                                    d_conv=4, d_state=16, node_dim=16, edge_dim=1
                                    )
        self.dropout = nn.Dropout(0.2)
        # 融合策略
        self.final_cross = CrossAttention(n_heads=heads, d_embed=dim, d_cross=dim2 * 2)
        self.final_cross2 = CrossAttention(n_heads=heads, d_embed=dim2 * 2, d_cross=dim)

        self.final_feed = FeedForward(dim + dim2 * 2, mult=cross_ff_multi, dropout=cross_ff_dropout)

        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim_out)
        )

    def forward(self, x_categ, x_numer, image_condition=None):
        assert x_categ.shape[
                   -1] == self.num_categories, f'you must pass in {self.num_categories} values for your categories input'
        # 处理MRI和PET模态，并行处理
        transform = T.AddRandomWalkPE(walk_length=20, attr_name='pe')
        mri_condition = image_condition[0]
        mri_condition = self.mri_conv(mri_condition)
        mri_condition = rearrange(mri_condition, 'b c h w d -> b c (h w) d').contiguous()
        mri_condition = transform(build_graph_from_img(mri_condition)).cuda()

        pet_condition = image_condition[1]
        pet_condition = self.pet_conv(pet_condition)
        pet_condition = rearrange(pet_condition, 'b c h w d -> b c (h w) d').contiguous()
        pet_condition = transform(build_graph_from_img(pet_condition)).cuda()

        mri_condition = self.mri_mamba(mri_condition.x, mri_condition.pe, mri_condition.edge_index,
                                       mri_condition.edge_attr,
                                       mri_condition.batch)
        pet_condition = self.pet_mamba(pet_condition.x, pet_condition.pe, pet_condition.edge_index,
                                       pet_condition.edge_attr,
                                       pet_condition.batch)  # (batch, channel=64)

        whole_condition = torch.cat([mri_condition, pet_condition], dim=1)
        whole_condition = self.dropout(whole_condition)
        # 处理文本模态 GCondNet
        xs = []
        if self.num_unique_categories > 0:
            x_categ = x_categ + self.categories_offset
            # x_categ = self.categorical_embeds(x_categ)
            xs.append(x_categ)
        # add numerically embedded tokens
        if self.num_continuous > 0:
            # x_numer = self.numerical_embedder(x_numer)
            xs.append(x_numer)
        x = torch.cat(xs, dim=1)
        # x = torch.mean(x, dim=1, keepdims=True)  # (batch,1,dim=512)
        x = self.DNN(x)

        # 多模态融合
        res1 = self.final_cross(x[:, None, :], whole_condition[:, None, :]) + x[:, None, :]
        res2 = self.final_cross(whole_condition[:, None, :], x[:, None, :]) + whole_condition[:, None, :]
        final_x = torch.cat([res1, res2], dim=2)
        final_x = self.final_feed(final_x) + final_x

        final_x = final_x.squeeze(1)  # make less dimension to linear layer

        logits = self.to_logits(final_x)

        return logits


class Graph_mamba6_(nn.Module):
    """1.使用GCondNet处理文本 2.增加了dropout 3.图像融合 """

    def __init__(
            self,
            *,
            categories,
            num_continuous,
            dim,
            depth,
            heads,
            dim_out=1,
            num_special_tokens=2,
            cross_ff_multi=2,
            cross_ff_dropout=0.1,
            dim2=256,
            **kwargs
    ):
        super().__init__()
        assert all(map(lambda n: n > 0, categories)), 'number of each category must be positive'
        assert len(categories) + num_continuous > 0, 'input shape must not be null'

        self.num_categories = len(categories)
        self.num_unique_categories = sum(categories)

        # create category embeddings table
        self.num_special_tokens = num_special_tokens
        # total_tokens = self.num_unique_categories + num_special_tokens

        # for automatically offsetting unique category ids to the correct position in the categories embedding table
        if self.num_unique_categories > 0:
            categories_offset = F.pad(torch.tensor(list(categories)), (1, 0), value=num_special_tokens)
            categories_offset = categories_offset.cumsum(dim=-1)[:-1]
            self.register_buffer('categories_offset', categories_offset)
        # continuous
        self.num_continuous = num_continuous

        self.DNN = GCondNet(kwargs['X'], kwargs['graphs_dataset_all'], dim=[100, 256, 512], layer=2)

        self.mri_conv = torch.nn.Conv3d(in_channels=1, out_channels=1, kernel_size=(16, 16, 8), stride=(16, 16, 8))
        self.pet_conv = torch.nn.Conv3d(in_channels=1, out_channels=1, kernel_size=(16, 16, 8), stride=(16, 16, 8))

        self.mri_mamba = GraphModel(channels=dim2, pe_dim=8, num_layers=depth,
                                    model_type='mamba',
                                    shuffle_ind=0, order_by_degree=True,
                                    d_conv=4, d_state=16, node_dim=16, edge_dim=1
                                    )
        self.pet_mamba = GraphModel(channels=dim2, pe_dim=8, num_layers=depth,
                                    model_type='mamba',
                                    shuffle_ind=0, order_by_degree=True,
                                    d_conv=4, d_state=16, node_dim=16, edge_dim=1
                                    )
        self.dropout = nn.Dropout(0.2)
        # 融合策略
        self.final_cross = CrossAttention(n_heads=heads, d_embed=dim, d_cross=dim2 * 2)
        self.final_cross2 = CrossAttention(n_heads=heads, d_embed=dim2 * 2, d_cross=dim)

        self.final_feed = FeedForward(dim + dim2 * 2, mult=cross_ff_multi, dropout=cross_ff_dropout)

        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim_out)
        )

    def forward(self, x_categ, x_numer, image_condition=None):
        assert x_categ.shape[
                   -1] == self.num_categories, f'you must pass in {self.num_categories} values for your categories input'
        # 处理MRI和PET模态，并行处理
        transform = T.AddRandomWalkPE(walk_length=20, attr_name='pe')
        mri_condition = image_condition[0]
        mri_condition = self.mri_conv(mri_condition)
        mri_condition = rearrange(mri_condition, 'b c h w d -> b c (h w) d').contiguous()
        mri_condition = transform(build_graph_from_img(mri_condition)).cuda()

        pet_condition = image_condition[1]
        pet_condition = self.pet_conv(pet_condition)
        pet_condition = rearrange(pet_condition, 'b c h w d -> b c (h w) d').contiguous()
        pet_condition = transform(build_graph_from_img(pet_condition)).cuda()

        mri_condition = self.mri_mamba(mri_condition.x, mri_condition.pe, mri_condition.edge_index,
                                       mri_condition.edge_attr,
                                       mri_condition.batch)
        pet_condition = self.pet_mamba(pet_condition.x, pet_condition.pe, pet_condition.edge_index,
                                       pet_condition.edge_attr,
                                       pet_condition.batch)  # (batch, channel=64)

        whole_condition = torch.cat([mri_condition, pet_condition], dim=1)
        whole_condition = self.dropout(whole_condition)
        # 处理文本模态 GCondNet
        xs = []
        if self.num_unique_categories > 0:
            x_categ = x_categ + self.categories_offset
            # x_categ = self.categorical_embeds(x_categ)
            xs.append(x_categ)
        # add numerically embedded tokens
        if self.num_continuous > 0:
            # x_numer = self.numerical_embedder(x_numer)
            xs.append(x_numer)
        x = torch.cat(xs, dim=1)
        # x = torch.mean(x, dim=1, keepdims=True)  # (batch,1,dim=512)
        x = self.DNN(x)

        # 多模态融合
        res1 = self.final_cross(x[:, None, :], whole_condition[:, None, :]) + x[:, None, :]
        res2 = self.final_cross(whole_condition[:, None, :], x[:, None, :]) + whole_condition[:, None, :]
        final_x = torch.cat([res1, res2], dim=2)
        final_x = self.final_feed(final_x) + final_x

        final_x = final_x.squeeze(1)  # make less dimension to linear layer

        logits = self.to_logits(final_x)

        return logits


class Graph_mamba7_(nn.Module):
    """1.使用GCondNet处理文本 2.增加了dropout
    图mamba不如图conv，猜测是图mamba的结构问题
    (GCN-Gmamba)Xn --> (GCN Xn - Gmamba Xn)"""

    def __init__(
            self,
            *,
            categories,
            num_continuous,
            dim,
            depth,
            heads,
            dim_out=1,
            num_special_tokens=2,
            cross_ff_multi=2,
            cross_ff_dropout=0.1,
            dim2=256,
            **kwargs
    ):
        super().__init__()
        assert all(map(lambda n: n > 0, categories)), 'number of each category must be positive'
        assert len(categories) + num_continuous > 0, 'input shape must not be null'

        self.num_categories = len(categories)
        self.num_unique_categories = sum(categories)

        # create category embeddings table
        self.num_special_tokens = num_special_tokens
        total_tokens = self.num_unique_categories + num_special_tokens

        # for automatically offsetting unique category ids to the correct position in the categories embedding table
        if self.num_unique_categories > 0:
            categories_offset = F.pad(torch.tensor(list(categories)), (1, 0), value=num_special_tokens)
            categories_offset = categories_offset.cumsum(dim=-1)[:-1]
            self.register_buffer('categories_offset', categories_offset)
        # continuous
        self.num_continuous = num_continuous

        self.DNN = GCondNet(kwargs['X'], kwargs['graphs_dataset_all'], dim=[100, 256, 512], layer=2)

        self.mri_conv = torch.nn.Conv3d(in_channels=1, out_channels=1, kernel_size=(16, 16, 8), stride=(16, 16, 8))
        self.pet_conv = torch.nn.Conv3d(in_channels=1, out_channels=1, kernel_size=(16, 16, 8), stride=(16, 16, 8))

        self.mri_GCN = GraphModel(channels=dim2, pe_dim=8, num_layers=depth,
                                  model_type='Gine',
                                  shuffle_ind=0, order_by_degree=True,
                                  d_conv=4, d_state=16, node_dim=16, edge_dim=1
                                  )
        self.mri_mamba = Graphblock(channels=dim2, num_layers=depth,
                                    model_type='only_mamba',
                                    shuffle_ind=0, order_by_degree=True,
                                    d_conv=4, d_state=16, if_pool=True
                                    )
        self.pet_GCN = GraphModel(channels=dim2, pe_dim=8, num_layers=depth,
                                  model_type='Gine',
                                  shuffle_ind=0, order_by_degree=True,
                                  d_conv=4, d_state=16, node_dim=16, edge_dim=1
                                  )
        self.pet_mamba = Graphblock(channels=dim2, num_layers=depth,
                                    model_type='only_mamba',
                                    shuffle_ind=0, order_by_degree=True,
                                    d_conv=4, d_state=16, if_pool=True
                                    )
        self.dropout = nn.Dropout(0.2)
        # 融合策略
        self.final_cross = CrossAttention(n_heads=heads, d_embed=dim, d_cross=dim2 * 2)
        self.final_feed = FeedForward(dim, mult=cross_ff_multi, dropout=cross_ff_dropout)

        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim_out)
        )

    def forward(self, x_categ, x_numer, image_condition=None):
        assert x_categ.shape[
                   -1] == self.num_categories, f'you must pass in {self.num_categories} values for your categories input'
        # 处理MRI和PET模态，并行处理
        transform = T.AddRandomWalkPE(walk_length=20, attr_name='pe')
        mri_condition = image_condition[0]
        mri_condition = self.mri_conv(mri_condition)
        mri_condition = rearrange(mri_condition, 'b c h w d -> b c (h w) d').contiguous()
        mri_condition = transform(build_graph_from_img(mri_condition)).cuda()

        pet_condition = image_condition[1]
        pet_condition = self.pet_conv(pet_condition)
        pet_condition = rearrange(pet_condition, 'b c h w d -> b c (h w) d').contiguous()
        pet_condition = transform(build_graph_from_img(pet_condition)).cuda()

        mri_condition = self.mri_GCN(mri_condition.x, mri_condition.pe, mri_condition.edge_index,
                                     mri_condition.edge_attr,
                                     mri_condition.batch)
        mri_condition = self.mri_mamba(mri_condition.x, mri_condition.pe, mri_condition.edge_index,
                                       mri_condition.edge_attr,
                                       mri_condition.batch)
        pet_condition = self.pet_GCN(pet_condition.x, pet_condition.pe, pet_condition.edge_index,
                                     pet_condition.edge_attr,
                                     pet_condition.batch)
        pet_condition = self.pet_mamba(pet_condition.x, pet_condition.pe, pet_condition.edge_index,
                                       pet_condition.edge_attr,
                                       pet_condition.batch)  # (batch, channel=64)

        whole_condition = torch.cat([mri_condition, pet_condition], dim=1)
        whole_condition = self.dropout(whole_condition)
        # 处理文本模态 GCondNet
        xs = []
        if self.num_unique_categories > 0:
            x_categ = x_categ + self.categories_offset
            # x_categ = self.categorical_embeds(x_categ)
            xs.append(x_categ)
        # add numerically embedded tokens
        if self.num_continuous > 0:
            # x_numer = self.numerical_embedder(x_numer)
            xs.append(x_numer)
        x = torch.cat(xs, dim=1)
        # x = torch.mean(x, dim=1, keepdims=True)  # (batch,1,dim=512)
        x = self.DNN(x)

        # 多模态融合
        x = self.final_cross(x[:, None, :], whole_condition[:, None, :]) + x[:, None, :]
        x = self.final_feed(x) + x

        x = x.squeeze(1)  # make less dimension to linear layer

        logits = self.to_logits(x)

        return logits


class Graph_mamba_(nn.Module):
    """1.使用GCondNet处理文本 2.增加了dropout
    3.(GCN Xn - Gmamba Xn)
    4.CNN特征融合
    """

    def __init__(
            self,
            *,
            categories,
            num_continuous,
            dim,
            depth,
            heads,
            dim_out=1,
            num_special_tokens=2,
            cross_ff_multi=2,
            cross_ff_dropout=0.1,
            dim2=256,
            **kwargs
    ):
        super().__init__()
        assert all(map(lambda n: n > 0, categories)), 'number of each category must be positive'
        assert len(categories) + num_continuous > 0, 'input shape must not be null'

        self.num_categories = len(categories)
        self.num_unique_categories = sum(categories)

        # create category embeddings table
        self.num_special_tokens = num_special_tokens
        total_tokens = self.num_unique_categories + num_special_tokens

        # for automatically offsetting unique category ids to the correct position in the categories embedding table
        if self.num_unique_categories > 0:
            categories_offset = F.pad(torch.tensor(list(categories)), (1, 0), value=num_special_tokens)
            categories_offset = categories_offset.cumsum(dim=-1)[:-1]
            self.register_buffer('categories_offset', categories_offset)
        # continuous
        self.num_continuous = num_continuous

        self.DNN = GCondNet(kwargs['X'], kwargs['graphs_dataset_all'], dim=[100, 256, 512], layer=2)

        self.feature_layer = CNN_Fusion()

        self.mri_GCN = GraphModel(channels=dim2, pe_dim=8, num_layers=depth,
                                  model_type='gine',
                                  shuffle_ind=0, order_by_degree=True,
                                  d_conv=4, d_state=16, node_dim=64 * 16, edge_dim=1
                                  )
        self.mri_mamba = Graphblock(channels=dim2, num_layers=depth,
                                    model_type='only_mamba',
                                    shuffle_ind=0, order_by_degree=True,
                                    d_conv=4, d_state=16, if_pool=True
                                    )

        self.dropout = nn.Dropout(0.2)
        # 融合策略
        self.final_cross = CrossAttention(n_heads=heads, d_embed=dim, d_cross=dim2 )
        self.final_feed = FeedForward(dim, mult=cross_ff_multi, dropout=cross_ff_dropout)

        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim_out)
        )

    def forward(self, x_categ, x_numer, image_condition=None):
        assert x_categ.shape[
                   -1] == self.num_categories, f'you must pass in {self.num_categories} values for your categories input'
        # 处理MRI和PET模态，并行处理
        transform = T.AddRandomWalkPE(walk_length=20, attr_name='pe')
        mri_condition = image_condition[0]
        pet_condition = image_condition[1]

        whole_condition = self.feature_layer(mri_condition, pet_condition)
        #*64
        whole_condition = rearrange(whole_condition, 'b c h w d -> b c (h w) d').contiguous()
        whole_condition = transform(build_graph_from_img(whole_condition)).cuda()

        latent_condition = self.mri_GCN(whole_condition.x, whole_condition.pe, whole_condition.edge_index,
                                        whole_condition.edge_attr,
                                        whole_condition.batch)
        whole_condition = self.mri_mamba(latent_condition, whole_condition.edge_index,
                                         whole_condition.edge_attr,
                                         whole_condition.batch)

        whole_condition = self.dropout(whole_condition)
        # 处理文本模态 GCondNet
        xs = []
        if self.num_unique_categories > 0:
            x_categ = x_categ + self.categories_offset
            # x_categ = self.categorical_embeds(x_categ)
            xs.append(x_categ)
        # add numerically embedded tokens
        if self.num_continuous > 0:
            # x_numer = self.numerical_embedder(x_numer)
            xs.append(x_numer)
        x = torch.cat(xs, dim=1)

        x = self.DNN(x)

        # 多模态融合
        x = self.final_cross(x[:, None, :], whole_condition[:, None, :]) + x[:, None, :]
        x = self.final_feed(x) + x
        x = x.squeeze(1)  # make less dimension to linear layer

        logits = self.to_logits(x)

        return logits


if __name__ == '__main__':
    batch = 2
    mri = torch.rand(batch, 1, 160, 160, 96).cuda()
    pet = torch.rand(batch, 1, 160, 160, 96).cuda()

    x_cat = torch.randint(1, 3, (batch, 5)).cuda()
    x_num = torch.rand(batch, 28).cuda()

    model = Graph_mamba_(categories=[11, 2, 2, 4, 4],
                         num_continuous=28,
                         dim=128,
                         depth=1,
                         heads=8, ).cuda()
    y_hat = model(x_cat, x_num, [mri, pet])
    y = torch.rand(batch)
