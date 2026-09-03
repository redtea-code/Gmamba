import pytest
import torch

from graph_construct.channel_graph_learned_edge_model import ChannelGraphClassifier


def _sources(batch=2):
    return {
        "mri": torch.randn(batch, 1, 32, 40, 32),
        "pet_gen": torch.randn(batch, 1, 32, 40, 32),
        "z_rec": torch.randn(batch, 64, 8, 10, 8),
        "z_gen_mri": torch.randn(batch, 64, 8, 10, 8),
        "z_gen_pet": torch.randn(batch, 64, 8, 10, 8),
    }


def test_alignment_modes_make_common_encoder_inputs_and_320_nodes():
    for spatial_alignment in ("adaptive_avg_pool", "learned_cnn"):
        for encoder_mode in ("shared_encoder", "independent_encoder"):
            model = ChannelGraphClassifier(
                source_names=tuple(_sources()), categories=(), num_continuous=5,
                dim=128, depth=2, node_channels=64, pool_size=(2, 2, 2),
                top_k=8, encoder_mode=encoder_mode, base_width=8,
                spatial_alignment=spatial_alignment,
                alignment_target_size=(8, 10, 8),
            )
            aligned = model.align_sources(_sources())
            assert all(value.shape == (2, 8, 8, 10, 8) for value in aligned.values())
            logits, details = model(
                torch.empty(2, 0, dtype=torch.long), torch.randn(2, 5), _sources(),
                return_details=True,
            )
            assert logits.shape == (2, 1)
            assert details["nodes"].shape == (2, 320, 128)
            assert details["adjacency"].shape == (320, 320)
            assert torch.isfinite(logits).all()


def test_alignment_requires_latent_target_shape():
    with pytest.raises(ValueError, match="alignment_target_size"):
        ChannelGraphClassifier(source_names=tuple(_sources()), alignment_target_size=None)


def test_learned_edge_keeps_self_loops_and_top_k_neighbors():
    model = ChannelGraphClassifier(
        source_names=tuple(_sources()), categories=(), num_continuous=5, dim=16,
        depth=1, node_channels=64, pool_size=(2, 2, 2), top_k=8, base_width=8,
        alignment_target_size=(8, 10, 8),
    )
    adjacency = model.graph.learned_adjacency()
    assert torch.all(adjacency.diagonal() > 0)
    assert (adjacency > 0).sum(dim=1).eq(8).all()


def test_channel_graph_parser_accepts_c_group_focal_configuration():
    from run_ad_project_binary_channel_graph import build_parser

    args = build_parser().parse_args([
        "--dataset", "nacc", "--alpha", "0.75", "--gamma", "3.0", "--focal-loss"
    ])
    assert args.alpha == 0.75
    assert args.gamma == 3.0
    assert args.focal_loss is True
