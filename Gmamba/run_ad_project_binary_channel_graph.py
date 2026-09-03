"""Train and run inference for the binary ADNI/NACC datasets in AD_Project.

The original AFF-GMamba entry point expects a pretrained MRI-to-PET model and
an external classifier package that are not present in this checkout.  The
available AD_Project ``*_base`` datasets contain one normalized MRI per scan.
This runner keeps the checked-in ``Graph GCN`` classifier, uses the supplied
MRI-to-PET checkpoint for the second image condition, and records the full
setup in every run's metadata.  Files with ``cropped_norm`` are read as-is;
the raw MRI is not normalized or resized.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import re
import subprocess
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from einops import rearrange
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from torch.utils.data import DataLoader, Dataset

GENERATOR_SOURCE_ROOT = Path("/zjs/cjw/TMI/Exp151_ganmamba_t1mri_fdgpet")
if str(GENERATOR_SOURCE_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(GENERATOR_SOURCE_ROOT))

from graph_construct.channel_graph_learned_edge_model import ChannelGraphClassifier
from run_baseline_comparison import BinaryFocalLoss
from pytorch3dunet.unet3d.model import Residual_mid_UNet3D_vit
from conditioned_gmamba_vit import ConditionedNativeMriToPet
from graph_construct.cross_task_patch_graph_gcn_model import (
    CrossTaskPatchGraphGCNClassifier,
    MriMriEncoderPatchGraphGCNClassifier,
    MriPatchGraphGCNClassifier,
    MriPetMriEncoderPatchGraphGCNClassifier,
    MriPetPatchGraphGCNClassifier,
)


DEFAULTS = {
    "adni": {
        "data_root": "/zjs/AD_Project/ADNI/ADNI_base",
        "table_path": "/zjs/AD_Project/ADNI/ADNI_table_general.csv",
        "image_names": ("MRI_brain_mni152_cropped_norm.nii.gz",),
        "id_column": "PTID",
        "date_column": "EXAMDATE",
        "table_label_column": "LABEL",
        "max_date_days": 30,
    },
    "nacc": {
        "data_root": "/zjs/AD_Project/NACC/NACC_base",
        "table_path": "/zjs/AD_Project/NACC/3year2_general.csv",
        "image_names": ("MRI_N4_brain_mni152_cropped_norm.nii.gz",),
        "id_column": "NACCID",
        "date_column": "date",
        "table_label_column": None,
        "max_date_days": 365,
    },
}

ABLATION_CHOICES = ("full", "no_mri", "no_pet", "no_clinical")
FEATURE_MODE_CHOICES = (
    "mri_only",
    "mri_pet",
    "mri_mri_encoder",
    "mri_pet_mri_encoder",
    "mri_pet_latent_concat",
)
LATENT_FEATURE_FILES = {
    # Latest generator exports under ADNI/gen contain 64-channel maps with
    # the MRI filename prefixed to each feature file.
    "z_rec": ("MRI_brain_mni152_cropped_norm_image_downsample_latent.nii.gz", 64),
    "z_gen_mri": ("MRI_brain_mni152_cropped_norm_mri_encoder_latent.nii.gz", 64),
    "z_gen_pet": ("MRI_brain_mni152_cropped_norm_pre_pet_decoder_latent.nii.gz", 64),
}

LEGACY_LATENT_FEATURE_NAMES = {
    "z_rec": ("raw_mri_downsampled_latent.nii.gz", "*_image_downsample_latent.nii.gz"),
    "z_gen_mri": ("mri_encoder_latent.nii.gz", "*_mri_encoder_latent.nii.gz"),
    "z_gen_pet": ("pre_pet_decoder_latent.nii.gz", "*_pre_pet_decoder_latent.nii.gz"),
}

FEATURE_MODE_LATENT_SOURCES = {
    "mri_only": (),
    "mri_pet": (),
    "mri_mri_encoder": ("z_gen_mri",),
    "mri_pet_mri_encoder": ("z_gen_mri",),
    "mri_pet_latent_concat": tuple(LATENT_FEATURE_FILES),
}


class LegacyNativeMriToPet(nn.Module):
    """Run the fixed-size MRI-to-PET ViT on native cropped_norm volumes.

    The checkpoint was trained with a bottleneck ViT canvas of 320x120, while
    the native cropped_norm volumes produce a different bottleneck canvas.
    Only that intermediate 2D feature canvas is resized; the input MRI and the
    generated output stay at the native volume shape.
    """

    vit_canvas = (320, 120)

    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base_model = base_model

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if tuple(image.shape[-3:]) == (160, 160, 96):
            return self.base_model(image)

        encoder_features = []
        x = image
        for encoder in self.base_model.encoders:
            x = encoder(x)
            encoder_features.insert(0, x)
        encoder_features = encoder_features[1:]

        # Match Mid_UNet_vit.forward(), but adapt only the fixed ViT canvas.
        mid_input = rearrange(x, "b c (md1 md2) h w -> b c (h md1) (md2 w)", md1=8)
        source_canvas = tuple(mid_input.shape[-2:])
        vit_input = F.interpolate(
            mid_input,
            size=self.vit_canvas,
            mode="bilinear",
            align_corners=False,
        )
        mid_output = self.base_model.mid(vit_input)
        mid_output = F.interpolate(
            mid_output,
            size=source_canvas,
            mode="bilinear",
            align_corners=False,
        )
        x = rearrange(
            mid_output,
            "b c (h md1) (md2 w) -> b c (md1 md2) h w",
            md1=8,
            w=x.shape[-1],
        )

        for decoder, skip in zip(self.base_model.decoders, encoder_features):
            x = decoder(skip, x)
        x = self.base_model.final_conv(x)
        if not self.base_model.training and self.base_model.final_activation is not None:
            x = self.base_model.final_activation(x)
        return x


NativeMriToPet = ConditionedNativeMriToPet


def load_mri_to_pet(
    checkpoint_path: Path,
    device: torch.device,
    category_sizes: tuple[int, ...] = (),
    num_continuous: int = 0,
) -> nn.Module:
    base_model = Residual_mid_UNet3D_vit(
        1,
        1,
        is_segmentation=False,
        f_maps=(64, 128, 256),
    )
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    if not isinstance(state, dict):
        raise TypeError(f"Expected a state_dict in {checkpoint_path}, got {type(state)!r}")
    if state and all(str(key).startswith("module.") for key in state):
        state = {str(key)[len("module."):]: value for key, value in state.items()}
    if any(str(key).startswith("base_model.") for key in state):
        generator = ConditionedNativeMriToPet(
            base_model=base_model,
            mri_aae_checkpoint=Path(
                "/zjs/MRI2PET/MRI2PET/baseline/Exp151_ganmamba_t1mri_fdgpet/checkpoints/mri_aae_best.pth"
            ),
            category_sizes=category_sizes,
            num_continuous=num_continuous,
            freeze_aae_encoder=True,
            aae_latent_channels=4,
            table_token_dim=128,
        )
        generator.load_state_dict(state, strict=True)
    else:
        base_model.load_state_dict(state, strict=True)
        generator = LegacyNativeMriToPet(base_model).to(device).eval()
    for parameter in generator.parameters():
        parameter.requires_grad_(False)
    return generator


class AblatedGraphGCN(nn.Module):
    """Remove one learned modality representation while keeping the backbone fixed."""

    def __init__(self, base_model: nn.Module, ablation: str):
        super().__init__()
        if ablation not in ABLATION_CHOICES or ablation == "full":
            raise ValueError(f"Unsupported ablation wrapper: {ablation}")
        self.base_model = base_model
        self.ablation = ablation
        self._hooks = []
        if ablation == "no_mri":
            self._hooks.append(self.base_model.graph_encoder.register_forward_hook(self._zero_output))
        elif ablation == "no_pet":
            self._hooks.append(self.base_model.graph_encoder.register_forward_hook(self._zero_output))
        elif ablation == "no_clinical":
            for name in ("categorical_embeds", "numerical_embedder"):
                module = getattr(self.base_model, name, None)
                if module is not None:
                    self._hooks.append(module.register_forward_hook(self._zero_output))

    @staticmethod
    def _zero_output(module: nn.Module, inputs: tuple[Any, ...], output: Any) -> Any:
        if not isinstance(output, torch.Tensor):
            raise TypeError(f"Expected tensor output from {module.__class__.__name__}, got {type(output)!r}")
        return torch.zeros_like(output)

    def forward(self, x_categ: torch.Tensor, x_numer: torch.Tensor, image_condition: list[torch.Tensor]) -> torch.Tensor:
        return self.base_model(x_categ, x_numer, image_condition)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(jsonable(value), indent=2, ensure_ascii=True) + "\n")


def parse_scan_folder(name: str) -> tuple[str, pd.Timestamp, int] | None:
    date_match = re.match(r"^(?P<subject>.+?)-(?P<date>\d{4}_\d{2}_\d{2})", name)
    label_match = re.search(r"-(?P<label>[01])$", name)
    if date_match is None or label_match is None:
        return None
    scan_date = pd.to_datetime(date_match.group("date"), format="%Y_%m_%d", errors="coerce")
    if pd.isna(scan_date):
        return None
    return date_match.group("subject"), scan_date, int(label_match.group("label"))


def find_image(folder: Path, image_names: Iterable[str]) -> Path | None:
    for image_name in image_names:
        candidate = folder / image_name
        if candidate.is_file():
            return candidate
    candidates = sorted(
        p for p in folder.glob("*.nii.gz") if "cropped_norm" in p.name and "stats" not in p.name
    )
    return candidates[0] if candidates else None


def find_generated_pet(
    generated_pet_root: Path,
    data_root: Path,
    image_folder: Path,
) -> Path | None:
    """Locate the already-generated PET while preserving the MRI tree layout."""
    relative_folder = image_folder.relative_to(data_root)
    roots = (
        generated_pet_root / relative_folder,
        generated_pet_root / data_root.name / relative_folder,
    )
    for root in roots:
        candidates = [root / "PET_generated.nii.gz"]
        candidates.extend(sorted(root.glob("*_generated_pet.nii.gz")))
        for path in candidates:
            if path.is_file():
                return path
    return None


def find_generated_feature(
    generated_pet_root: Path,
    data_root: Path,
    image_folder: Path,
    filename: str,
) -> Path | None:
    relative_folder = image_folder.relative_to(data_root)
    roots = (
        generated_pet_root / relative_folder,
        generated_pet_root / data_root.name / relative_folder,
    )
    aliases = (filename,) + LEGACY_LATENT_FEATURE_NAMES.get(
        next((name for name, (canonical, _channels) in LATENT_FEATURE_FILES.items() if canonical == filename), ""),
        (),
    )
    for root in roots:
        candidates = [root / name for name in aliases if "*" not in name]
        for pattern in aliases:
            if "*" in pattern:
                candidates.extend(sorted(root.glob(pattern)))
        for path in candidates:
            if path.is_file():
                return path
    return None


def prepare_clinical_features(table: pd.DataFrame, dataset: str) -> dict[str, Any]:
    """Mirror table/deal_table.py for the two numeric AD_Project tables."""
    cfg = DEFAULTS[dataset]
    table = table.copy().reset_index(drop=True)
    info_columns = [cfg["id_column"], cfg["date_column"]]
    if dataset == "adni":
        info_columns.append("LABEL")
    # date_diff is a matching helper produced during table preparation, not a
    # clinical predictor.  Exclude it from both ADNI and NACC feature vectors.
    info_columns.extend(
        column for column in table.columns if str(column).strip().lower() == "date_diff"
    )
    feature_columns = [column for column in table.columns if column not in info_columns]
    categorical_columns: list[str] = []
    numeric_columns: list[str] = []
    encoded_categorical: dict[str, np.ndarray] = {}
    encoded_numeric: dict[str, np.ndarray] = {}
    category_sizes: list[int] = []

    for column in feature_columns:
        series = table[column]
        # This is the same intent as discovery_mix(): only object columns that
        # actually contain text are treated as categorical features.
        has_text = series.dtype == object and series.astype(str).str.contains(r"[A-Za-z]", regex=True).any()
        if has_text:
            categorical_columns.append(column)
            values = series.fillna("NA").astype(str)
            levels = {value: index for index, value in enumerate(sorted(values.unique()))}
            encoded_categorical[column] = values.map(levels).to_numpy(dtype=np.int64)
            category_sizes.append(len(levels))
        else:
            numeric_columns.append(column)
            values = pd.to_numeric(series, errors="coerce").fillna(0).to_numpy(dtype=np.float64)
            encoded_numeric[column] = values

    if numeric_columns:
        numeric_matrix = np.column_stack([encoded_numeric[column] for column in numeric_columns])
        mean = numeric_matrix.mean(axis=0)
        std = numeric_matrix.std(axis=0)
        std[std == 0] = 1.0
        numeric_matrix = ((numeric_matrix - mean) / std).astype(np.float32)
    else:
        numeric_matrix = np.empty((len(table), 0), dtype=np.float32)

    if categorical_columns:
        categorical_matrix = np.column_stack(
            [encoded_categorical[column] for column in categorical_columns]
        ).astype(np.int64)
    else:
        categorical_matrix = np.empty((len(table), 0), dtype=np.int64)

    return {
        "table": table,
        "categorical": categorical_matrix,
        "numeric": numeric_matrix,
        "category_sizes": tuple(category_sizes),
        "categorical_columns": categorical_columns,
        "numeric_columns": numeric_columns,
    }


def match_clinical_row(
    table: pd.DataFrame,
    dataset: str,
    subject_id: str,
    scan_date: pd.Timestamp,
    label: int,
) -> tuple[int, int, int | None]:
    cfg = DEFAULTS[dataset]
    subject_mask = table[cfg["id_column"]].astype(str).str.strip() == subject_id
    candidates = table.loc[subject_mask].copy()
    if candidates.empty:
        raise ValueError(f"No clinical row for {subject_id} ({dataset})")
    candidates["_date"] = pd.to_datetime(candidates[cfg["date_column"]], errors="coerce")
    candidates = candidates[candidates["_date"].notna()].copy()
    if candidates.empty:
        raise ValueError(f"No valid clinical date for {subject_id} ({dataset})")
    candidates["_diff"] = (candidates["_date"] - scan_date).abs().dt.days.astype(int)

    # ADNI's original loader first requires the table label to agree with the
    # suffix in the image folder, then selects the closest visit within 30d.
    label_column = cfg["table_label_column"]
    if label_column is not None:
        same_label = candidates[pd.to_numeric(candidates[label_column], errors="coerce") == label]
        if not same_label.empty:
            candidates = same_label

    chosen = candidates.sort_values(["_diff"]).iloc[0]
    diff_days = int(chosen["_diff"])
    if diff_days > cfg["max_date_days"]:
        raise ValueError(
            f"Clinical match for {subject_id} is {diff_days} days away "
            f"(limit={cfg['max_date_days']})"
        )
    table_index = int(chosen.name)
    table_label = None
    if label_column is not None and pd.notna(chosen[label_column]):
        table_label = int(chosen[label_column])
    return table_index, diff_days, table_label


def discover_records(
    dataset: str,
    data_root: Path,
    table_path: Path,
    generated_pet_root: Path | None = None,
    include_latent_features: bool = False,
    latent_sources: tuple[str, ...] | None = None,
    max_samples: int | None = None,
    min_scan_date: pd.Timestamp | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cfg = DEFAULTS[dataset]
    table = pd.read_csv(table_path)
    clinical = prepare_clinical_features(table, dataset)
    records: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []

    for folder in sorted(p for p in data_root.iterdir() if p.is_dir()):
        parsed = parse_scan_folder(folder.name)
        if parsed is None:
            skipped.append({"folder": folder.name, "reason": "unrecognized_folder_name"})
            continue
        subject_id, scan_date, label = parsed
        if min_scan_date is not None and scan_date < min_scan_date:
            continue
        image_path = find_image(folder, cfg["image_names"])
        if image_path is None:
            skipped.append({"folder": folder.name, "reason": "normalized_image_missing"})
            continue
        generated_pet_path: Path | None = None
        if generated_pet_root is not None:
            generated_pet_path = find_generated_pet(generated_pet_root, data_root, folder)
            if generated_pet_path is None:
                skipped.append({"folder": folder.name, "reason": "generated_pet_missing"})
                continue
        latent_paths: dict[str, str] = {}
        if include_latent_features:
            if generated_pet_root is None:
                raise ValueError("Latent feature mode requires --generated-pet-root")
            missing_latent = None
            requested_latent_sources = latent_sources or tuple(LATENT_FEATURE_FILES)
            for source_name in requested_latent_sources:
                filename, _channels = LATENT_FEATURE_FILES[source_name]
                path = find_generated_feature(generated_pet_root, data_root, folder, filename)
                if path is None:
                    missing_latent = source_name
                    break
                latent_paths[source_name] = str(path)
            if missing_latent is not None:
                skipped.append({
                    "folder": folder.name,
                    "reason": f"latent_feature_missing_{missing_latent}",
                })
                continue
        try:
            table_index, diff_days, table_label = match_clinical_row(
                clinical["table"], dataset, subject_id, scan_date, label
            )
        except ValueError as exc:
            skipped.append({"folder": folder.name, "reason": str(exc)})
            continue
        records.append(
            {
                "name": folder.name,
                "image_path": str(image_path),
                "subject_id": subject_id,
                "scan_date": scan_date.strftime("%Y-%m-%d"),
                "label": label,
                "generated_pet_path": str(generated_pet_path) if generated_pet_path else None,
                **latent_paths,
                "table_index": table_index,
                "table_date_diff_days": diff_days,
                "table_label": table_label,
                "cate_x": clinical["categorical"][table_index],
                "conti_x": clinical["numeric"][table_index],
            }
        )

    if max_samples is not None and max_samples < len(records):
        # Keep a deterministic representation of both classes for smoke runs.
        selected: list[dict[str, Any]] = []
        by_label = {0: [r for r in records if r["label"] == 0], 1: [r for r in records if r["label"] == 1]}
        while len(selected) < max_samples and (by_label[0] or by_label[1]):
            for label in (0, 1):
                if by_label[label] and len(selected) < max_samples:
                    selected.append(by_label[label].pop(0))
        records = selected

    if not records:
        raise RuntimeError(f"No usable {dataset} records found under {data_root}")
    metadata = {
        "dataset": dataset,
        "data_root": str(data_root),
        "table_path": str(table_path),
        "generated_pet_root": str(generated_pet_root) if generated_pet_root else None,
        "pet_source": "existing_generated_pet" if generated_pet_root else "mri_to_pet_inference",
        "latent_feature_files": {
            name: LATENT_FEATURE_FILES[name][0]
            for name in (latent_sources or tuple(LATENT_FEATURE_FILES))
        } if include_latent_features else {},
        "latent_feature_channels": {
            name: LATENT_FEATURE_FILES[name][1]
            for name in (latent_sources or tuple(LATENT_FEATURE_FILES))
        } if include_latent_features else {},
        "total_discovered": len(records),
        "skipped": skipped,
        "category_sizes": clinical["category_sizes"],
        "categorical_columns": clinical["categorical_columns"],
        "numeric_columns": clinical["numeric_columns"],
        "num_categories": len(clinical["category_sizes"]),
        "num_continuous": len(clinical["numeric_columns"]),
        "min_scan_date": min_scan_date.strftime("%Y-%m-%d") if min_scan_date is not None else None,
    }
    return records, metadata


def split_records(records: list[dict[str, Any]], seed: int) -> dict[str, list[int]]:
    labels = np.asarray([record["label"] for record in records], dtype=np.int64)
    groups = np.asarray([record["subject_id"] for record in records])
    n_groups = len(np.unique(groups))
    if n_groups < 3:
        raise RuntimeError("At least three subjects are required for grouped train/val/test splits")

    n_splits = min(5, n_groups)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = list(splitter.split(np.zeros(len(records)), labels, groups))
    overall_positive = labels.mean()

    def choose_fold(candidates: list[tuple[np.ndarray, np.ndarray]], desired_fraction: float) -> tuple[np.ndarray, np.ndarray]:
        def score(pair: tuple[np.ndarray, np.ndarray]) -> float:
            _, held_out = pair
            held_labels = labels[held_out]
            held_rate = held_labels.mean() if len(held_labels) else overall_positive
            return abs(len(held_out) / len(labels) - desired_fraction) + abs(held_rate - overall_positive)

        return min(candidates, key=score)

    trainval, test = choose_fold(folds, 0.2)
    remaining_groups = groups[trainval]
    remaining_labels = labels[trainval]
    inner_splits = min(5, len(np.unique(remaining_groups)))
    inner = StratifiedGroupKFold(n_splits=inner_splits, shuffle=True, random_state=seed + 1)
    inner_folds = list(inner.split(np.zeros(len(trainval)), remaining_labels, remaining_groups))

    def inner_score(pair: tuple[np.ndarray, np.ndarray]) -> float:
        _, held_out = pair
        held_labels = remaining_labels[held_out]
        held_rate = held_labels.mean() if len(held_labels) else overall_positive
        return abs(len(held_out) / len(records) - 0.15) + abs(held_rate - overall_positive)

    inner_train, inner_val = min(inner_folds, key=inner_score)
    train = trainval[inner_train]
    val = trainval[inner_val]
    result = {"train": train.tolist(), "val": val.tolist(), "test": test.tolist()}
    group_sets = {key: {groups[index] for index in value} for key, value in result.items()}
    assert not (group_sets["train"] & group_sets["val"])
    assert not (group_sets["train"] & group_sets["test"])
    assert not (group_sets["val"] & group_sets["test"])
    return result


class BinaryMRIDataset(Dataset):
    def __init__(
        self,
        records: list[dict[str, Any]],
        indices: list[int],
        load_pet: bool = True,
        load_latents: bool = False,
        latent_sources: tuple[str, ...] | None = None,
    ):
        self.records = records
        self.indices = indices
        self.load_pet = load_pet
        self.load_latents = load_latents
        self.latent_sources = latent_sources or tuple(LATENT_FEATURE_FILES)

    @staticmethod
    def _load_latent(path: str, channels: int) -> torch.Tensor:
        array = np.asarray(nib.load(path).dataobj, dtype=np.float32).copy()
        if array.ndim != 4:
            raise ValueError(f"Expected a 4D latent NIfTI, got {array.shape} for {path}")
        if array.shape[-1] == channels:
            array = np.moveaxis(array, -1, 0)
        elif array.shape[0] != channels:
            raise ValueError(
                f"Expected {channels} latent channels in the first or last dimension, "
                f"got {array.shape} for {path}"
            )
        return torch.from_numpy(np.ascontiguousarray(array))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[self.indices[index]]
        image = np.asarray(nib.load(record["image_path"]).dataobj, dtype=np.float32).copy()
        if image.ndim == 4 and image.shape[-1] == 1:
            image = image[..., 0]
        if image.ndim != 3:
            raise ValueError(f"Expected a 3D MRI volume, got {image.shape} for {record['image_path']}")
        image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
        sample = {
            "image": image_tensor.squeeze(0),
            "cate_x": torch.as_tensor(record["cate_x"], dtype=torch.long),
            "conti_x": torch.as_tensor(record["conti_x"], dtype=torch.float32),
            "label": torch.tensor(record["label"], dtype=torch.float32),
            "name": record["name"],
            "subject_id": record["subject_id"],
        }
        if self.load_pet:
            generated_pet_path = record.get("generated_pet_path")
            if generated_pet_path:
                pet = np.asarray(nib.load(generated_pet_path).dataobj, dtype=np.float32).copy()
                if pet.ndim == 4 and pet.shape[-1] == 1:
                    pet = pet[..., 0]
                if pet.ndim != 3:
                    raise ValueError(
                        f"Expected a 3D generated PET volume, got {pet.shape} for {generated_pet_path}"
                    )
                if tuple(pet.shape) != tuple(image.shape):
                    raise ValueError(
                        "MRI and generated PET must have identical native shapes: "
                        f"mri={image.shape}, pet={pet.shape}, case={record['name']}"
                    )
                sample["pet"] = torch.from_numpy(pet).unsqueeze(0)
            else:
                pet_path = record.get("pet_cache_path")
                if not pet_path:
                    raise RuntimeError("PET cache path is missing; run MRI-to-PET precomputation first")
                sample["pet"] = torch.load(pet_path, map_location="cpu", weights_only=True).float()
        if self.load_latents:
            for source_name in self.latent_sources:
                _filename, channels = LATENT_FEATURE_FILES[source_name]
                latent_path = record.get(source_name)
                if not latent_path:
                    raise RuntimeError(f"Latent path is missing for source={source_name}, case={record['name']}")
                sample[source_name] = self._load_latent(latent_path, channels)
        return sample


def nvidia_status() -> dict[int, tuple[int, int]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.check_output(command, text=True, stderr=subprocess.STDOUT)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("Cannot verify GPU idleness with nvidia-smi") from exc
    status: dict[int, tuple[int, int]] = {}
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 3:
            status[int(fields[0])] = (int(fields[1]), int(fields[2]))
    return status


def assert_requested_gpu_idle(device: torch.device, max_memory_mb: int = 128, max_utilization: int = 5) -> None:
    if device.type != "cuda":
        return
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible and visible not in {"NoDevFiles", "-1"}:
        physical = [int(item) for item in visible.split(",") if item.strip().isdigit()]
        physical_index = physical[device.index or 0] if physical else device.index or 0
    else:
        physical_index = device.index or 0
    status = nvidia_status()
    memory, utilization = status.get(physical_index, (10**9, 100))
    print(f"GPU check: physical={physical_index}, memory={memory} MiB, utilization={utilization}%")
    if memory > max_memory_mb or utilization > max_utilization:
        raise RuntimeError(
            f"GPU {physical_index} is not idle (memory={memory} MiB, utilization={utilization}%). "
            "Refusing to start because this runner only uses idle GPUs."
        )


def move_batch(
    batch: dict[str, Any],
    device: torch.device,
    feature_mode: str,
) -> dict[str, torch.Tensor]:
    moved = {
        "image": batch["image"].to(device, non_blocking=True),
        "cate_x": batch["cate_x"].to(device, non_blocking=True),
        "conti_x": batch["conti_x"].to(device, non_blocking=True),
        "label": batch["label"].to(device, non_blocking=True),
    }
    if "pet" in batch:
        moved["pet"] = batch["pet"].to(device, non_blocking=True)
    for source_name in FEATURE_MODE_LATENT_SOURCES[feature_mode]:
        moved[source_name] = batch[source_name].to(device, non_blocking=True)
    return moved


def build_channel_graph_sources(moved: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Map dataset batch tensors to the five channel-graph source names."""
    return {
        "mri": moved["image"],
        "pet_gen": moved["pet"],
        "z_rec": moved["z_rec"],
        "z_gen_mri": moved["z_gen_mri"],
        "z_gen_pet": moved["z_gen_pet"],
    }


def build_image_condition(
    moved: dict[str, torch.Tensor],
    feature_mode: str,
) -> Any:
    if feature_mode == "mri_only":
        return MriPatchGraphGCNClassifier.build_image_condition(
            mri=moved["image"],
            pet_gen=moved.get("pet"),
        )
    if feature_mode == "mri_pet_latent_concat":
        return CrossTaskPatchGraphGCNClassifier.build_image_condition(
            mri=moved["image"],
            pet_gen=moved["pet"],
            z_rec=moved["z_rec"],
            z_gen_mri=moved["z_gen_mri"],
            z_gen_pet=moved["z_gen_pet"],
        )
    if feature_mode == "mri_pet_mri_encoder":
        return MriPetMriEncoderPatchGraphGCNClassifier.build_image_condition(
            mri=moved["image"],
            pet_gen=moved["pet"],
            z_gen_mri=moved["z_gen_mri"],
        )
    if feature_mode == "mri_mri_encoder":
        return MriMriEncoderPatchGraphGCNClassifier.build_image_condition(
            mri=moved["image"],
            pet_gen=None,
            z_gen_mri=moved["z_gen_mri"],
        )
    return [moved["image"], moved["pet"]]


def safe_metric(function, *args, **kwargs) -> float:
    try:
        value = function(*args, **kwargs)
        return float(value)
    except (ValueError, ZeroDivisionError):
        return float("nan")


def calculate_metrics(labels: np.ndarray, probabilities: np.ndarray, losses: list[float]) -> dict[str, Any]:
    predictions = (probabilities >= 0.5).astype(np.int64)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1]).tolist()
    metrics = {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "accuracy": safe_metric(accuracy_score, labels, predictions),
        "balanced_accuracy": safe_metric(balanced_accuracy_score, labels, predictions),
        "recall_macro": safe_metric(recall_score, labels, predictions, average="macro", zero_division=0),
        "precision_macro": safe_metric(precision_score, labels, predictions, average="macro", zero_division=0),
        "f1_macro": safe_metric(f1_score, labels, predictions, average="macro", zero_division=0),
        "mcc": safe_metric(matthews_corrcoef, labels, predictions),
        "auc": safe_metric(roc_auc_score, labels, probabilities),
        "n": int(len(labels)),
        "class_counts": {"0": int((labels == 0).sum()), "1": int((labels == 1).sum())},
        "confusion_matrix": matrix,
    }
    # Keep the descriptive fields for backwards compatibility and expose the
    # paper-style aliases used by the result tables.
    metrics.update(
        {
            "ACC": metrics["accuracy"],
            "REC": metrics["recall_macro"],
            "PRE": metrics["precision_macro"],
            "AUC": metrics["auc"],
            "F1": metrics["f1_macro"],
        }
    )
    return metrics


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    amp: bool,
    prediction_path: Path | None = None,
    feature_mode: str = "mri_pet",
) -> dict[str, Any]:
    model.eval()
    all_labels: list[np.ndarray] = []
    all_probabilities: list[np.ndarray] = []
    losses: list[float] = []
    rows: list[dict[str, Any]] = []
    autocast_enabled = amp and device.type == "cuda"
    for batch in loader:
        moved = move_batch(batch, device, feature_mode)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=autocast_enabled):
            logits = model(
                moved["cate_x"],
                moved["conti_x"],
                build_channel_graph_sources(moved) if feature_mode == "mri_pet_latent_concat" else build_image_condition(moved, feature_mode),
            )
            loss = criterion(logits.squeeze(1), moved["label"])
        probabilities = torch.sigmoid(logits.squeeze(1)).float().cpu().numpy()
        labels = moved["label"].long().cpu().numpy()
        all_labels.append(labels)
        all_probabilities.append(probabilities)
        losses.append(float(loss.detach().cpu()))
        names = list(batch["name"])
        subjects = list(batch["subject_id"])
        for name, subject, target, probability in zip(names, subjects, labels, probabilities):
            rows.append(
                {
                    "name": name,
                    "subject_id": subject,
                    "label": int(target),
                    "probability": float(probability),
                    "prediction": int(probability >= 0.5),
                }
            )

    labels = np.concatenate(all_labels) if all_labels else np.empty(0, dtype=np.int64)
    probabilities = np.concatenate(all_probabilities) if all_probabilities else np.empty(0, dtype=np.float32)
    metrics = calculate_metrics(labels, probabilities, losses)
    if prediction_path is not None:
        pd.DataFrame(rows).to_csv(prediction_path, index=False)
    return metrics


def make_loader(
    records: list[dict[str, Any]],
    indices: list[int],
    batch_size: int,
    shuffle: bool,
    workers: int,
    device: torch.device,
    load_pet: bool = True,
    load_latents: bool = False,
    latent_sources: tuple[str, ...] | None = None,
) -> DataLoader:
    dataset = BinaryMRIDataset(
        records,
        indices,
        load_pet=load_pet,
        load_latents=load_latents,
        latent_sources=latent_sources,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )


def precompute_pet_cache(
    records: list[dict[str, Any]],
    cache_dir: Path,
    generator: nn.Module,
    device: torch.device,
    workers: int,
    amp: bool,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    missing = [index for index, record in enumerate(records) if not Path(record["pet_cache_path"]).is_file()]
    if not missing:
        print(f"MRI-to-PET cache: reusing {len(records)} volumes from {cache_dir}")
        return

    loader = make_loader(
        records,
        missing,
        batch_size=1,
        shuffle=False,
        workers=workers,
        device=device,
        load_pet=False,
    )
    name_to_path = {record["name"]: Path(record["pet_cache_path"]) for record in records}
    autocast_enabled = amp and device.type == "cuda"
    generator.eval()
    print(f"MRI-to-PET cache: generating {len(missing)} native volumes")
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader, start=1):
            image = batch["image"].to(device, non_blocking=True)
            cate_x = batch["cate_x"].to(device, non_blocking=True)
            conti_x = batch["conti_x"].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=autocast_enabled):
                if isinstance(generator, ConditionedNativeMriToPet):
                    pet = generator(image, cate_x, conti_x)
                else:
                    pet = generator(image)
            for name, volume in zip(list(batch["name"]), pet):
                path = name_to_path[name]
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(volume.detach().float().cpu(), path)
            if batch_index == 1 or batch_index % 25 == 0 or batch_index == len(missing):
                print(f"MRI-to-PET cache: {batch_index}/{len(missing)}")


def save_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, metrics: dict[str, Any], run_config: dict[str, Any]) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "metrics": deepcopy(metrics),
            "run_config": jsonable(run_config),
        },
        path,
    )


def train(args: argparse.Namespace) -> None:
    dataset_name = args.dataset.lower()
    default = DEFAULTS[dataset_name]
    data_root = Path(args.data_root or default["data_root"])
    table_path = Path(args.table_path or default["table_path"])
    generated_pet_root = Path(args.generated_pet_root) if args.generated_pet_root else None
    feature_mode = args.feature_mode
    use_latent_features = feature_mode in ("mri_mri_encoder", "mri_pet_mri_encoder", "mri_pet_latent_concat")
    use_pet = feature_mode not in ("mri_only", "mri_mri_encoder")
    latent_sources = FEATURE_MODE_LATENT_SOURCES[feature_mode]
    if use_latent_features and generated_pet_root is None:
        raise ValueError("Latent feature modes require --generated-pet-root")
    if use_latent_features and args.ablation != "full":
        raise ValueError("Latent graph classification currently supports --ablation full only")
    output_dir = Path(args.output_dir or f"runs/ad_project_binary/{dataset_name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    min_scan_date = pd.to_datetime(args.min_scan_date) if args.min_scan_date else None

    if args.num_threads > 0:
        torch.set_num_threads(args.num_threads)
        torch.set_num_interop_threads(1)
    set_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        assert_requested_gpu_idle(device)
        torch.cuda.set_device(device.index if device.index is not None else 0)
    torch.set_float32_matmul_precision("high")

    records, data_metadata = discover_records(
        dataset_name,
        data_root,
        table_path,
        generated_pet_root=generated_pet_root,
        include_latent_features=use_latent_features,
        latent_sources=latent_sources,
        max_samples=args.max_samples,
        min_scan_date=min_scan_date,
    )
    if use_latent_features:
        image_shape = tuple(nib.load(records[0]["image_path"]).shape)
        latent_shapes = {}
        for source_name in latent_sources:
            latent_path = records[0][source_name]
            raw_shape = tuple(nib.load(latent_path).shape)
            latent_shapes[source_name] = raw_shape
        latent_spatial_shapes = {name: shape[:-1] if shape[-1] == 64 else shape[1:] for name, shape in latent_shapes.items()}
        if len(set(latent_spatial_shapes.values())) != 1:
            raise ValueError(f"Latent spatial shapes are inconsistent: {latent_spatial_shapes}")
        inferred_target = next(iter(latent_spatial_shapes.values()))
        if args.alignment_target_size is None:
            args.alignment_target_size = inferred_target
        elif tuple(args.alignment_target_size) != tuple(inferred_target):
            raise ValueError(f"Requested alignment target {args.alignment_target_size} != latent shape {inferred_target}")
        data_metadata["mri_input_shape"] = list(image_shape)
        data_metadata["latent_input_shapes"] = {name: list(shape) for name, shape in latent_shapes.items()}
        data_metadata["latent_spatial_shape"] = list(inferred_target)
        data_metadata["aligned_spatial_shape"] = list(args.alignment_target_size)
    if feature_mode == "mri_only":
        data_metadata["pet_source"] = "not_used"
    split = split_records(records, args.seed)
    for split_name, indices in split.items():
        for index in indices:
            records[index]["split"] = split_name

    manifest_rows = []
    for index, record in enumerate(records):
        manifest_rows.append(
            {
                "index": index,
                "split": record["split"],
                "name": record["name"],
                "image_path": record["image_path"],
                "generated_pet_path": record.get("generated_pet_path"),
                **{name: record.get(name) for name in LATENT_FEATURE_FILES},
                "subject_id": record["subject_id"],
                "scan_date": record["scan_date"],
                "label": record["label"],
                "table_index": record["table_index"],
                "table_date_diff_days": record["table_date_diff_days"],
                "table_label": record["table_label"],
            }
        )
    pd.DataFrame(manifest_rows).to_csv(output_dir / "manifest.csv", index=False)

    reference_checkpoint: Path | None = None
    cache_dir: Path | None = None
    if generated_pet_root is None and use_pet:
        reference_checkpoint = Path(args.mri_to_pet_checkpoint)
        if not reference_checkpoint.is_file():
            raise FileNotFoundError(f"MRI-to-PET checkpoint not found: {reference_checkpoint}")
        cache_dir = Path(
            args.pet_cache_dir
            or f"/tmp/gmamba_mri_to_pet_cache/{dataset_name}_{output_dir.name}_{len(records)}"
        )
        for index, record in enumerate(records):
            record["pet_cache_path"] = str(cache_dir / f"{index:05d}.pt")

        reference_model = load_mri_to_pet(
            reference_checkpoint,
            device,
            category_sizes=tuple(int(value) for value in data_metadata["category_sizes"]),
            num_continuous=int(data_metadata["num_continuous"]),
        )
        precompute_pet_cache(records, cache_dir, reference_model, device, args.workers, args.amp)
        write_json(
            output_dir / "mri_to_pet_config.json",
            {
                "checkpoint": reference_checkpoint,
                "cache_dir": cache_dir,
                "native_input": True,
                "native_shape_policy": "raw_cropped_norm_mri_and_latent_canvas_resize_only",
                "vit_canvas": list(NativeMriToPet.vit_canvas),
                "tabular_conditioning": isinstance(reference_model, ConditionedNativeMriToPet),
                "table_category_sizes": list(data_metadata["category_sizes"]),
                "table_num_continuous": int(data_metadata["num_continuous"]),
            },
        )
        del reference_model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    elif generated_pet_root is not None:
        write_json(
            output_dir / "mri_to_pet_config.json",
            {
                "source": "existing_generated_pet_files",
                "generated_pet_root": generated_pet_root,
                "inference_performed": False,
                "runtime_preprocessing": "none",
                "file_pattern": "PET_generated.nii.gz",
            },
        )

    loaders = {
        split_name: make_loader(
            records,
            indices,
            args.batch_size,
            split_name == "train",
            args.workers,
            device,
            load_pet=use_pet,
            load_latents=use_latent_features,
            latent_sources=latent_sources,
        )
        for split_name, indices in split.items()
    }
    if args.batch_size != 1:
        raise ValueError("Graph GCN reproduction currently requires --batch-size 1")

    if feature_mode != "mri_pet_latent_concat":
        raise ValueError("Channel graph formal experiments require --feature-mode mri_pet_latent_concat")
    if args.ablation != "full":
        raise ValueError("Channel graph formal experiments require --ablation full")
    model_config: dict[str, Any] = {
        "classifier": "ChannelGraphClassifier",
        "graph_sources": ["mri", "pet_gen", "z_rec", "z_gen_mri", "z_gen_pet"],
        "encoder_mode": args.encoder_mode,
        "spatial_alignment": args.spatial_alignment,
        "alignment_target_size": list(args.alignment_target_size),
        "node_channels": args.node_channels,
        "pool_size": list(args.pool_size),
        "top_k": args.top_k,
        "base_width": args.base_width,
        "categories": list(data_metadata["category_sizes"]),
        "num_continuous": data_metadata["num_continuous"],
        "dim": args.dim,
        "depth": args.depth,
    }
    model = ChannelGraphClassifier(
        source_names=("mri", "pet_gen", "z_rec", "z_gen_mri", "z_gen_pet"),
        categories=tuple(data_metadata["category_sizes"]),
        num_continuous=int(data_metadata["num_continuous"]),
        dim=args.dim, depth=args.depth, node_channels=args.node_channels,
        pool_size=tuple(args.pool_size), top_k=args.top_k,
        encoder_mode=args.encoder_mode, base_width=args.base_width,
        spatial_alignment=args.spatial_alignment,
        alignment_target_size=tuple(args.alignment_target_size),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = (
        BinaryFocalLoss(alpha=args.alpha, gamma=args.gamma)
        if args.focal_loss
        else nn.BCEWithLogitsLoss()
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    autocast_enabled = args.amp and device.type == "cuda"
    run_config = {
        "dataset": dataset_name,
        "data_root": str(data_root),
        "table_path": str(table_path),
        "output_dir": str(output_dir),
        "seed": args.seed,
        "image_size": "native_cropped_norm_shape",
        "spatial_alignment": args.spatial_alignment,
        "mri_input_shape": data_metadata.get("mri_input_shape"),
        "latent_spatial_shape": data_metadata.get("latent_spatial_shape"),
        "aligned_spatial_shape": data_metadata.get("aligned_spatial_shape"),
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "early_stopping_patience": args.early_stopping_patience,
        "model_selection_metric": args.model_selection_metric,
        "num_threads": args.num_threads,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "focal_loss": {"enabled": args.focal_loss, "alpha": args.alpha, "gamma": args.gamma},
        "amp": args.amp,
        "device": str(device),
        "model": model_config,
        "ablation": args.ablation,
        "ablation_description": (
            "MRI-only image source + clinical"
            if feature_mode == "mri_only"
            else {
                "full": "MRI + generated PET + clinical",
                "no_mri": "remove MRI Graph-Mamba representation",
                "no_pet": "remove generated PET Graph-Mamba representation",
                "no_clinical": "remove clinical embedding representation",
            }[args.ablation]
        ),
        "feature_mode": feature_mode,
        "channel_node_pool_size": list(args.pool_size),
        "channel_node_top_k": args.top_k,
        "image_condition_mode": (
            "mri_only"
            if feature_mode == "mri_only"
            else (
                "mri_and_mri_encoder_latent"
            if feature_mode == "mri_mri_encoder"
                else (
                "mri_generated_pet_and_mri_encoder_latent"
            if feature_mode == "mri_pet_mri_encoder"
                else (
                "mri_generated_pet_and_three_latent_features"
                if feature_mode == "mri_pet_latent_concat"
                else ("mri_and_existing_generated_pet" if generated_pet_root else "mri_and_generated_pet")
                )
                )
            )
        ),
        "generated_pet_root": str(generated_pet_root) if generated_pet_root else None,
        "reference_mri_to_pet_checkpoint": str(reference_checkpoint) if reference_checkpoint else None,
        "mri_to_pet_cache_dir": str(cache_dir) if cache_dir else None,
        "mri_to_pet_native_adapter": None if generated_pet_root or not use_pet else "latent_canvas_resize_320x120",
        "runtime_pet_inference": generated_pet_root is None and use_pet,
        "runtime_preprocessing": "none",
        "data_metadata": data_metadata,
    }
    write_json(output_dir / "run_config.json", run_config)
    write_json(output_dir / "data_metadata.json", data_metadata)

    print(
        f"{dataset_name}: usable={len(records)} train={len(split['train'])} "
        f"val={len(split['val'])} test={len(split['test'])} "
        f"subjects={len({r['subject_id'] for r in records})}"
    )
    print(f"clinical numeric columns: {data_metadata['numeric_columns']}")
    print(f"output: {output_dir.resolve()}")

    best_score = -float("inf")
    best_epoch = -1
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []
    for epoch in range(args.epochs):
        model.train()
        train_losses: list[float] = []
        start = time.time()
        for step, batch in enumerate(loaders["train"]):
            moved = move_batch(batch, device, feature_mode)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=autocast_enabled):
                logits = model(
                    moved["cate_x"],
                    moved["conti_x"],
                    build_channel_graph_sources(moved) if feature_mode == "mri_pet_latent_concat" else build_image_condition(moved, feature_mode),
                )
                loss = criterion(logits.squeeze(1), moved["label"])
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
            if args.max_steps is not None and step + 1 >= args.max_steps:
                break

        val_metrics = evaluate(
            model,
            loaders["val"],
            device,
            criterion,
            args.amp,
            feature_mode=feature_mode,
        )
        epoch_row = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(train_losses)) if train_losses else float("nan"),
            "val": val_metrics,
            "elapsed_seconds": time.time() - start,
        }
        history.append(epoch_row)
        (output_dir / "history.jsonl").open("a").write(json.dumps(jsonable(epoch_row)) + "\n")
        if args.model_selection_metric == "accuracy":
            score = val_metrics.get("accuracy", -float("inf"))
        else:
            score = val_metrics.get("auc", float("nan"))
            if not np.isfinite(score):
                score = val_metrics.get("balanced_accuracy", -float("inf"))
        if score > best_score:
            best_score = score
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            save_checkpoint(output_dir / "best.pt", model, optimizer, epoch + 1, val_metrics, run_config)
        else:
            epochs_without_improvement += 1
        save_checkpoint(output_dir / "last.pt", model, optimizer, epoch + 1, val_metrics, run_config)
        print(
            f"epoch={epoch + 1:02d}/{args.epochs} train_loss={epoch_row['train_loss']:.5f} "
            f"val_acc={val_metrics['accuracy']:.4f} val_auc={val_metrics['auc']:.4f} "
            f"val_f1={val_metrics['f1_macro']:.4f} "
            f"no_improve={epochs_without_improvement}/{args.early_stopping_patience} "
            f"({epoch_row['elapsed_seconds']:.1f}s)"
        )
        if (
            args.early_stopping_patience > 0
            and epochs_without_improvement >= args.early_stopping_patience
        ):
            print(
                f"early stopping at epoch {epoch + 1}: "
                f"validation score did not improve for {args.early_stopping_patience} epochs"
            )
            break

    best_checkpoint = torch.load(output_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model_state"])
    test_metrics = evaluate(
        model,
        loaders["test"],
        device,
        criterion,
        args.amp,
        output_dir / "test_predictions.csv",
        feature_mode=feature_mode,
    )
    result = {
        "best_epoch": best_epoch,
        "best_validation": best_checkpoint["metrics"],
        "test": test_metrics,
        "history": history,
        "gpu_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    write_json(output_dir / "metrics.json", result)
    print("test:", json.dumps(jsonable(test_metrics), ensure_ascii=True))
    print(f"checkpoint: {output_dir / 'best.pt'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DEFAULTS), required=True)
    parser.add_argument("--ablation", choices=ABLATION_CHOICES, default="full")
    parser.add_argument("--data-root")
    parser.add_argument("--table-path")
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--feature-mode",
        choices=FEATURE_MODE_CHOICES,
        default="mri_pet_latent_concat",
        help="Formal channel-node model uses MRI, generated PET, and three saved latent maps.",
    )
    parser.add_argument(
        "--generated-pet-root",
        help="Read existing PET_generated.nii.gz files from this mirrored dataset tree; no inference is run.",
    )
    parser.add_argument(
        "--mri-to-pet-checkpoint",
        default="/zjs/cjw/TMI/Exp151_ganmamba_t1mri_fdgpet/outputs/gmamba_vit_baseline/checkpoints/gmamba_vit_best_psnr.pth",
    )
    parser.add_argument("--pet-cache-dir")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--early-stopping-patience", type=int, default=15)
    parser.add_argument(
        "--model-selection-metric",
        choices=("auc", "accuracy"),
        default="auc",
        help="Validation metric maximized when selecting best.pt.",
    )
    parser.add_argument("--lr", type=float, default=6e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=0.0)
    parser.add_argument("--focal-loss", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=3,
        metavar=("D", "H", "W"),
        default=(4, 4, 4),
        help="Non-overlapping 3D graph patch size.",
    )
    parser.add_argument("--encoder-mode", choices=("shared_encoder", "independent_encoder"), default="shared_encoder")
    parser.add_argument("--node-channels", type=int, default=64)
    parser.add_argument("--base-width", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--pool-size", type=int, nargs=3, default=(2, 2, 2))
    parser.add_argument("--spatial-alignment", choices=("adaptive_avg_pool", "learned_cnn"), default="adaptive_avg_pool")
    parser.add_argument("--alignment-target-size", type=int, nargs=3, default=None, metavar=("D", "W", "H"))
    parser.add_argument(
        "--extra-encoder-cnn",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Add one 3D convolutional block to each source feature encoder.",
    )
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--min-scan-date", help="Keep scans on or after this ISO date, e.g. 2011-01-01")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    return parser


if __name__ == "__main__":
    train(build_parser().parse_args())
