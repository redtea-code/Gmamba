"""Channel-node multimodal graph model with sparse learned edges."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn


FULL_SOURCE_NAMES = ("mri", "pet_gen", "z_rec", "z_gen_mri", "z_gen_pet")


class Residual3DBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.norm1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False)
        self.norm2 = nn.BatchNorm3d(out_channels)
        self.skip = (nn.Sequential(nn.Conv3d(in_channels, out_channels, 1, stride, bias=False),
                                   nn.BatchNorm3d(out_channels))
                     if stride != 1 or in_channels != out_channels else nn.Identity())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = torch.relu(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return torch.relu(x + residual)


class LearnedSpatialDownsampler(nn.Module):
    """Learned MRI/PET downsampler from native image space to latent space."""

    def __init__(self, in_channels: int = 8, hidden_channels: int = 16, out_channels: int = 8):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv3d(in_channels, hidden_channels, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm3d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_channels, out_channels, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class Lightweight3DResNet(nn.Module):
    """Small trunk; the adapter determines whether input is image or latent."""
    def __init__(self, output_channels: int = 64, base_width: int = 8):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(base_width, base_width, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(base_width), nn.ReLU(inplace=True),
            nn.MaxPool3d(3, stride=2, padding=1),
        )
        self.blocks = nn.Sequential(
            Residual3DBlock(base_width, base_width),
            Residual3DBlock(base_width, base_width * 2, stride=2),
            Residual3DBlock(base_width * 2, base_width * 2),
            Residual3DBlock(base_width * 2, output_channels, stride=2),
            Residual3DBlock(output_channels, output_channels),
        )
        self.output_channels = output_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(self.stem(x))


class TopKLearnedGraph(nn.Module):
    def __init__(self, num_nodes: int, dim: int, top_k: int = 8, depth: int = 2, dropout: float = 0.1):
        super().__init__()
        if not 1 <= top_k <= num_nodes:
            raise ValueError(f"top_k must be in [1, {num_nodes}], got {top_k}")
        self.num_nodes, self.top_k = num_nodes, top_k
        self.edge_logits = nn.Parameter(torch.zeros(num_nodes, num_nodes))
        nn.init.normal_(self.edge_logits, std=0.02)
        self.weights = nn.ModuleList(nn.Linear(dim, dim, bias=False) for _ in range(depth))
        self.norms = nn.ModuleList(nn.LayerNorm(dim) for _ in range(depth))
        self.dropout = nn.Dropout(dropout)

    def learned_adjacency(self) -> torch.Tensor:
        logits = (self.edge_logits + self.edge_logits.transpose(0, 1)) / 2
        diagonal = torch.eye(self.num_nodes, device=logits.device, dtype=torch.bool)
        if self.top_k == 1:
            mask = diagonal
        else:
            nonself = logits.masked_fill(diagonal, torch.finfo(logits.dtype).min)
            _, indices = torch.topk(nonself, k=self.top_k - 1, dim=1)
            mask = torch.zeros_like(logits, dtype=torch.bool).scatter(1, indices, True)
            mask |= diagonal
        return torch.softmax(logits.masked_fill(~mask, torch.finfo(logits.dtype).min), dim=1)

    def forward(self, nodes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        adjacency = self.learned_adjacency()
        for weight, norm in zip(self.weights, self.norms):
            message = torch.einsum("ij,bjd->bid", adjacency, nodes)
            nodes = norm(nodes + self.dropout(torch.relu(weight(message))))
        return nodes, adjacency


class ChannelGraphClassifier(nn.Module):
    """Five-source channel-node classifier; subsets are supported by source_names."""
    def __init__(self, *, source_names: Sequence[str] = FULL_SOURCE_NAMES, categories: Sequence[int] = (),
                 num_continuous: int = 5, dim: int = 128, depth: int = 2, node_channels: int = 64,
                 pool_size: tuple[int, int, int] = (2, 2, 2), top_k: int = 8,
                 encoder_mode: str = "shared_encoder", base_width: int = 8, dropout: float = 0.2,
                 spatial_alignment: str = "adaptive_avg_pool",
                 alignment_target_size: tuple[int, int, int] | None = None):
        super().__init__()
        self.source_names = tuple(source_names)
        if not self.source_names or not set(self.source_names).issubset(FULL_SOURCE_NAMES):
            raise ValueError(f"source_names must be a non-empty subset of {FULL_SOURCE_NAMES}")
        if encoder_mode not in {"shared_encoder", "independent_encoder"}:
            raise ValueError(f"Unknown encoder_mode={encoder_mode}")
        if spatial_alignment not in {"adaptive_avg_pool", "learned_cnn"}:
            raise ValueError(f"Unknown spatial_alignment={spatial_alignment}")
        if alignment_target_size is None or len(alignment_target_size) != 3:
            raise ValueError("alignment_target_size must be a 3-tuple")
        if any(int(size) <= 0 for size in alignment_target_size):
            raise ValueError(f"alignment_target_size must be positive, got {alignment_target_size}")
        self.encoder_mode, self.node_channels = encoder_mode, node_channels
        self.spatial_alignment = spatial_alignment
        self.alignment_target_size = tuple(int(size) for size in alignment_target_size)
        self.pool = nn.AdaptiveAvgPool3d(pool_size)
        self.pool_size = tuple(pool_size)
        self.adapters = nn.ModuleDict({
            name: nn.Sequential(nn.Conv3d(1 if name in ("mri", "pet_gen") else 64, base_width, 1),
                                nn.BatchNorm3d(base_width), nn.ReLU(inplace=True))
            for name in self.source_names
        })
        if spatial_alignment == "learned_cnn":
            self.image_downsampler = LearnedSpatialDownsampler(base_width, base_width * 2, base_width)
        else:
            self.image_downsampler = nn.Identity()
        if encoder_mode == "shared_encoder":
            self.encoders = nn.ModuleDict({"shared": Lightweight3DResNet(node_channels, base_width)})
        else:
            self.encoders = nn.ModuleDict({name: Lightweight3DResNet(node_channels, base_width) for name in self.source_names})
        self.node_projection = nn.Linear(pool_size[0] * pool_size[1] * pool_size[2], dim)
        self.source_embedding = nn.Parameter(torch.zeros(len(self.source_names), 1, dim))
        nn.init.normal_(self.source_embedding, std=0.02)
        self.graph = TopKLearnedGraph(len(self.source_names) * node_channels, dim, top_k, depth, dropout)
        self.categories = tuple(int(size) for size in categories)
        self.num_continuous = int(num_continuous)
        clinical_dim = sum(self.categories) + self.num_continuous
        self.clinical = nn.Sequential(nn.Linear(clinical_dim, dim), nn.LayerNorm(dim), nn.GELU(), nn.Dropout(dropout))
        self.head = nn.Sequential(nn.Linear(dim * 2, dim), nn.LayerNorm(dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim, 1))

    def align_sources(self, sources: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Adapt and spatially align every source before the shared/independent trunk."""
        missing = set(self.source_names) - set(sources)
        if missing:
            raise ValueError(f"Missing source tensors: {sorted(missing)}")
        aligned = {}
        target = self.alignment_target_size
        for name in self.source_names:
            x = sources[name]
            expected = 1 if name in ("mri", "pet_gen") else 64
            if x.ndim != 5 or x.size(1) != expected:
                raise ValueError(f"{name} must have shape [B,{expected},D,W,H], got {tuple(x.shape)}")
            x = self.adapters[name](x)
            if name in ("mri", "pet_gen"):
                if self.spatial_alignment == "adaptive_avg_pool":
                    x = F.adaptive_avg_pool3d(x, target)
                else:
                    x = self.image_downsampler(x)
                if tuple(x.shape[-3:]) != target:
                    raise ValueError(f"{name} alignment produced {tuple(x.shape[-3:])}, expected {target}")
            elif tuple(x.shape[-3:]) != target:
                raise ValueError(f"{name} has spatial shape {tuple(x.shape[-3:])}, expected {target}")
            aligned[name] = x
        return aligned

    def encode_sources(self, sources: Mapping[str, torch.Tensor]) -> torch.Tensor:
        aligned = self.align_sources(sources)
        nodes = []
        for index, name in enumerate(self.source_names):
            encoder = self.encoders["shared"] if self.encoder_mode == "shared_encoder" else self.encoders[name]
            x = self.pool(encoder(aligned[name])).flatten(2)
            # Channel nodes: [B, node_channels, pooled_spatial_volume].
            x = self.node_projection(x) + self.source_embedding[index]
            nodes.append(x)
        return torch.cat(nodes, dim=1)

    def forward(self, x_categ: torch.Tensor, x_numer: torch.Tensor, sources: Mapping[str, torch.Tensor],
                return_details: bool = False):
        nodes = self.encode_sources(sources)
        nodes, adjacency = self.graph(nodes)
        graph_feature = nodes.mean(dim=1)
        if x_categ.ndim != 2 or x_categ.size(1) != len(self.categories):
            raise ValueError(f"x_categ must have shape [B,{len(self.categories)}], got {tuple(x_categ.shape)}")
        if x_numer.ndim != 2 or x_numer.size(1) != self.num_continuous:
            raise ValueError(f"x_numer must have shape [B,{self.num_continuous}], got {tuple(x_numer.shape)}")
        categorical = []
        for index, size in enumerate(self.categories):
            if torch.any((x_categ[:, index] < 0) | (x_categ[:, index] >= size)):
                raise ValueError(f"categorical feature {index} contains an out-of-range value")
            categorical.append(F.one_hot(x_categ[:, index].long(), num_classes=size).float())
        clinical_input = torch.cat((*categorical, x_numer.float()), dim=1)
        clinical = self.clinical(clinical_input)
        logits = self.head(torch.cat((graph_feature, clinical), dim=1))
        if return_details:
            return logits, {"nodes": nodes, "adjacency": adjacency}
        return logits


class FiveSourceChannelGraphClassifier(ChannelGraphClassifier):
    source_names = FULL_SOURCE_NAMES

    def __init__(self, **kwargs):
        super().__init__(source_names=self.source_names, **kwargs)


class MriPetChannelGraphClassifier(ChannelGraphClassifier):
    source_names = ("mri", "pet_gen")

    def __init__(self, **kwargs):
        super().__init__(source_names=self.source_names, **kwargs)
