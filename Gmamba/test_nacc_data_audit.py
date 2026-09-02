import numpy as np
import pandas as pd
import torch

from run_baseline_comparison import BinaryFocalLoss
from run_ad_project_binary_repro import prepare_clinical_features


def test_binary_focal_loss_gamma_zero_matches_weighted_bce():
    logits = torch.tensor([-1.0, 0.5, 2.0])
    targets = torch.tensor([0.0, 1.0, 1.0])
    alpha = 0.65
    actual = BinaryFocalLoss(alpha=alpha, gamma=0.0)(logits, targets)
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    weights = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    expected = (weights * bce).mean()
    assert torch.allclose(actual, expected)


def test_binary_focal_loss_is_finite_and_alpha_weights_positive_examples():
    logits = torch.tensor([-100.0, 100.0])
    targets = torch.tensor([1.0, 0.0])
    low = BinaryFocalLoss(alpha=0.5, gamma=2.0)(logits, targets)
    high = BinaryFocalLoss(alpha=0.75, gamma=2.0)(logits, targets)
    assert torch.isfinite(low)
    assert torch.isfinite(high)
    positive_logit = torch.tensor([-0.25])
    positive_target = torch.tensor([1.0])
    assert BinaryFocalLoss(alpha=0.75, gamma=1.0)(positive_logit, positive_target) > BinaryFocalLoss(alpha=0.5, gamma=1.0)(positive_logit, positive_target)


def test_nacc_clinical_features_exclude_naccmmse_and_retain_cdrsum():
    table = pd.DataFrame({
        "NACCID": ["NACC0001", "NACC0002"],
        "visit_date": ["2020-01-01", "2020-01-02"],
        "NACCMMSE": [-4, -4],
        "CDRSUM": [0.0, 1.0],
        "NACCGDS": [1.0, 2.0],
        "SEX": [1, 2],
        "AGE": [70, 71],
    })
    clinical = prepare_clinical_features(table, "nacc")
    columns = clinical["numeric_columns"] + clinical["categorical_columns"]
    assert "NACCMMSE" not in columns
    assert {"CDRSUM", "NACCGDS", "SEX", "AGE"}.issubset(columns)


def test_nacc_focal_runner_parser_accepts_approved_configuration():
    from run_nacc_resnet_focal import build_parser
    args = build_parser().parse_args(["--alpha", "0.65", "--gamma", "2.0", "--no-amp"])
    assert args.dataset == "nacc"
    assert args.model == "resnet"
    assert (args.alpha, args.gamma) == (0.65, 2.0)


def test_nacc_numeric_normalization_uses_training_rows_only():
    table = pd.DataFrame({
        "NACCID": ["NACC0001", "NACC0002", "NACC0003"],
        "date": ["2020-01-01"] * 3,
        "CDRSUM": [0.0, 10.0, 20.0],
        "NACCGDS": [1.0, 1.0, 1.0],
        "SEX": [1, 1, 1],
        "AGE": [70.0, 70.0, 70.0],
    })
    clinical = prepare_clinical_features(table, "nacc", fit_indices=[0, 1])
    assert clinical["normalization_fit_indices"] == [0, 1]
    assert np.isclose(clinical["numeric"][2, clinical["numeric_columns"].index("CDRSUM")], 3.0)
