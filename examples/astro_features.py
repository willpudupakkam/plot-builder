"""Example Plot Builder extension for merger-catalogue astronomy data.

To enable it locally, copy this file to ``plot_builder_features.py`` in the
repository root. That destination is intentionally ignored by Git so local,
dataset-specific extensions are not accidentally published.
"""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd


ID_COLUMN = "#ID(1)"
COORD_COLUMNS = ("Xc(6)", "Yc(7)", "Zc(8)")
THETA_COLUMN_RE = re.compile(r"^theta_(\d+)$")

MERGER_AXIS_DISTANCE_COLUMN = "merger_axis_distance"
MERGER_AXIS_DISTANCE_NORMALIZED_COLUMN = "merger_axis_distance_normalized"
MERGER_AXIS_PERPENDICULAR_DISTANCE_COLUMN = "merger_axis_perpendicular_distance"
MERGER_AXIS_PERPENDICULAR_RADIUS_COLUMN = "merger_axis_perpendicular_radius"
MERGER_AXIS_SECONDARY_ID_COLUMN = "merger_axis_secondary_id"
MERGER_AXIS_SEPARATION_COLUMN = "merger_axis_separation"
CLUSTER_ROLE_COLUMN = "cluster_role"
CLUSTER_LABEL_COLUMN = "cluster_label"
PLOT_HIGHLIGHT_LABEL_COLUMN = "plot_highlight_label"
PLOT_HIGHLIGHT_COLOR_COLUMN = "plot_highlight_color"


def augment_catalogue(data: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Add merger-axis columns to a compatible astronomy catalogue."""
    del path  # Available to extensions that need filename-specific behavior.

    theta_column = merger_theta_column(data)
    if theta_column is None or not required_columns_available(data):
        return data

    match = THETA_COLUMN_RE.fullmatch(theta_column)
    if match is None:
        return data
    secondary_id = int(match.group(1))

    secondary_matches = data.index[data[ID_COLUMN].astype(str) == str(secondary_id)].tolist()
    if not secondary_matches:
        return data

    coordinate_frame = data.loc[:, list(COORD_COLUMNS)].apply(pd.to_numeric, errors="coerce")
    main_position = coordinate_frame.iloc[0].to_numpy(dtype=float)
    secondary_position = coordinate_frame.loc[secondary_matches[0]].to_numpy(dtype=float)
    axis_vector = secondary_position - main_position
    separation = float(np.linalg.norm(axis_vector))
    if not np.isfinite(separation) or separation == 0:
        return data

    coordinates = coordinate_frame.to_numpy(dtype=float)
    relative_positions = coordinates - main_position
    axis_unit_vector = axis_vector / separation
    perpendicular_unit_vector = perpendicular_axis_unit_vector(axis_unit_vector)
    finite_rows = np.isfinite(relative_positions).all(axis=1)
    signed_axis_distance = np.full(len(data), np.nan)
    signed_perpendicular_distance = np.full(len(data), np.nan)
    perpendicular_radius = np.full(len(data), np.nan)
    signed_axis_distance[finite_rows] = np.sum(
        relative_positions[finite_rows] * axis_unit_vector,
        axis=1,
    )
    signed_perpendicular_distance[finite_rows] = np.sum(
        relative_positions[finite_rows] * perpendicular_unit_vector,
        axis=1,
    )
    radial_distance = np.linalg.norm(relative_positions[finite_rows], axis=1)
    perpendicular_squared = np.maximum(
        radial_distance**2 - signed_axis_distance[finite_rows] ** 2,
        0,
    )
    perpendicular_radius[finite_rows] = np.sqrt(perpendicular_squared)

    augmented = data.copy()
    augmented[MERGER_AXIS_DISTANCE_COLUMN] = signed_axis_distance
    augmented[MERGER_AXIS_DISTANCE_NORMALIZED_COLUMN] = signed_axis_distance / separation
    augmented[MERGER_AXIS_PERPENDICULAR_DISTANCE_COLUMN] = signed_perpendicular_distance
    augmented[MERGER_AXIS_PERPENDICULAR_RADIUS_COLUMN] = perpendicular_radius
    augmented[MERGER_AXIS_SECONDARY_ID_COLUMN] = secondary_id
    augmented[MERGER_AXIS_SEPARATION_COLUMN] = separation
    augmented[CLUSTER_ROLE_COLUMN] = ""
    augmented[CLUSTER_LABEL_COLUMN] = ""
    augmented[PLOT_HIGHLIGHT_LABEL_COLUMN] = ""
    augmented[PLOT_HIGHLIGHT_COLOR_COLUMN] = ""
    augmented.loc[augmented.index[0], CLUSTER_ROLE_COLUMN] = "main"
    augmented.loc[augmented.index[0], CLUSTER_LABEL_COLUMN] = "Main cluster"
    augmented.loc[augmented.index[0], PLOT_HIGHLIGHT_LABEL_COLUMN] = "Main cluster"
    augmented.loc[augmented.index[0], PLOT_HIGHLIGHT_COLOR_COLUMN] = "#000000"
    augmented.loc[secondary_matches[0], CLUSTER_ROLE_COLUMN] = "bullet"
    augmented.loc[secondary_matches[0], CLUSTER_LABEL_COLUMN] = "Bullet cluster"
    augmented.loc[secondary_matches[0], PLOT_HIGHLIGHT_LABEL_COLUMN] = "Bullet cluster"
    augmented.loc[secondary_matches[0], PLOT_HIGHLIGHT_COLOR_COLUMN] = "#d92d20"
    return augmented


def merger_theta_column(data: pd.DataFrame) -> str | None:
    """Return the first theta column that encodes a secondary catalogue ID."""
    theta_columns = [column for column in data.columns if THETA_COLUMN_RE.fullmatch(str(column))]
    return str(theta_columns[0]) if theta_columns else None


def perpendicular_axis_unit_vector(axis_unit_vector: np.ndarray) -> np.ndarray:
    """Choose a stable unit vector perpendicular to the merger axis."""
    basis_vectors = np.eye(3)
    reference = basis_vectors[np.argmin(np.abs(basis_vectors @ axis_unit_vector))]
    perpendicular = reference - np.dot(reference, axis_unit_vector) * axis_unit_vector
    norm = float(np.linalg.norm(perpendicular))
    if not np.isfinite(norm) or norm == 0:
        return np.array([0.0, 1.0, 0.0])
    return perpendicular / norm


def required_columns_available(data: pd.DataFrame) -> bool:
    """Return whether the catalogue contains the columns this example needs."""
    return ID_COLUMN in data.columns and all(column in data.columns for column in COORD_COLUMNS)
