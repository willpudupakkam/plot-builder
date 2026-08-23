from __future__ import annotations

from dataclasses import dataclass
import hashlib
import html
import importlib.util
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components


DATA_DIR = Path("data")
SAVED_PLOTS_DIR = Path("saved_plots")
CUSTOM_FEATURES_PATH = Path("plot_builder_features.py")
STATE_DIR = Path(".plot_builder_state")
LEGACY_STATE_PATH = Path(".plot_builder_state.json")
DEFAULT_WORKSPACE = "default"
SOURCE_COLUMN = "source_file"
MAX_DERIVED_COLUMNS = 50
MAX_FILTERS = 12
COLOR_SCALES = [
    "Viridis",
    "Plasma",
    "Cividis",
    "Inferno",
    "Magma",
    "Turbo",
    "Bluered",
    "Rainbow",
    "Jet",
    "Hot",
    "IceFire",
    "Portland",
    "Electric",
]
DEFAULT_TRACE_COLORS = [
    "#2f80b7",
    "#d94b4b",
    "#2f9e63",
    "#8a5fbf",
    "#d28b26",
    "#1f9aa6",
    "#c94f9b",
    "#637939",
]
ALLOWED_FUNCTIONS: dict[str, Any] = {
    "abs": np.abs,
    "arccos": np.arccos,
    "arcsin": np.arcsin,
    "arctan": np.arctan,
    "clip": np.clip,
    "cos": np.cos,
    "exp": np.exp,
    "log": np.log,
    "log10": np.log10,
    "maximum": np.maximum,
    "minimum": np.minimum,
    "sin": np.sin,
    "sqrt": np.sqrt,
    "tan": np.tan,
    "where": np.where,
}
DERIVED_SCOPES = ["Global", "Per catalogue"]
DERIVED_SORT_OPTIONS = ["Manual", "Name A-Z"]
LINE_DASHES = ["solid", "dash", "dot", "dashdot"]
POINT_LABEL_MODES = ["None", "Hover only", "Show on plot"]


st.set_page_config(
    page_title="Plot Builder",
    page_icon=":bar_chart:",
    layout="wide",
)

st.markdown(
    """
    <style>
    .st-key-add-derived-column button,
    .st-key-add-filter button,
    .st-key-add-reference-line button {
        width: 1.8rem !important;
        min-width: 1.8rem !important;
        min-height: 1.7rem;
        height: 1.7rem;
        padding: 0 !important;
        line-height: 1;
    }

    .st-key-add-derived-column button [data-testid="stMarkdownContainer"],
    .st-key-add-filter button [data-testid="stMarkdownContainer"],
    .st-key-add-reference-line button [data-testid="stMarkdownContainer"],
    [class*="st-key-delete-filter-"] button [data-testid="stMarkdownContainer"],
    [class*="st-key-delete-reference-line-"] button [data-testid="stMarkdownContainer"] {
        display: none;
    }

    .st-key-add-derived-column [data-testid="stButton"],
    .st-key-add-filter [data-testid="stButton"],
    .st-key-add-reference-line [data-testid="stButton"] {
        display: flex;
        align-items: center;
        transform: translateY(0.35rem);
    }

    [class*="st-key-delete-filter-"] button,
    [class*="st-key-delete-reference-line-"] button {
        min-height: 2.45rem;
        padding: 0 0.55rem;
        border-color: #f04438 !important;
        background: #b42318 !important;
        color: white !important;
    }

    [class*="st-key-delete-filter-"] button:hover,
    [class*="st-key-delete-reference-line-"] button:hover {
        border-color: #ff6b5f !important;
        background: #d92d20 !important;
    }

    [class*="st-key-delete-filter-"] button [data-testid="stMarkdownContainer"],
    [class*="st-key-delete-reference-line-"] button [data-testid="stMarkdownContainer"] {
        display: none;
    }

    [class*="st-key-apply-derived-"] button {
        min-height: 2.45rem;
        padding: 0 0.55rem;
        border-color: #32d583 !important;
        background: #027a48 !important;
        color: white !important;
    }

    [class*="st-key-apply-derived-"] button:hover {
        border-color: #6ce9a6 !important;
        background: #039855 !important;
    }

    [class*="st-key-apply-derived-"] button [data-testid="stMarkdownContainer"] {
        display: none;
    }

    [class*="st-key-derived-name-"] [data-testid="stWidgetLabel"],
    [class*="st-key-derived-formula-"] [data-testid="stWidgetLabel"] {
        display: none;
    }

    [class*="st-key-edit-derived-"] button,
    [class*="st-key-toggle-derived-visible-"] button,
    [class*="st-key-delete-derived-"] button {
        min-height: 2.25rem;
        padding: 0 0.55rem;
    }

    [class*="st-key-edit-derived-"] button {
        border-color: #fdb022 !important;
        background: #b54708 !important;
        color: white !important;
    }

    [class*="st-key-edit-derived-"] button:hover {
        border-color: #fec84b !important;
        background: #dc6803 !important;
    }

    [class*="st-key-toggle-derived-visible-"] button {
        border-color: #84caff !important;
        background: #175cd3 !important;
        color: white !important;
    }

    [class*="st-key-toggle-derived-visible-"] button:hover {
        border-color: #b2ddff !important;
        background: #1570ef !important;
    }

    [class*="st-key-delete-derived-"] button {
        border-color: #f04438 !important;
        background: #b42318 !important;
        color: white !important;
    }

    [class*="st-key-delete-derived-"] button:hover {
        border-color: #ff6b5f !important;
        background: #d92d20 !important;
    }

    [class*="st-key-edit-derived-"] button [data-testid="stMarkdownContainer"],
    [class*="st-key-toggle-derived-visible-"] button [data-testid="stMarkdownContainer"],
    [class*="st-key-delete-derived-"] button [data-testid="stMarkdownContainer"] {
        display: none;
    }

    .derived-card-summary {
        display: flex;
        flex-direction: column;
        justify-content: center;
        gap: 0.18rem;
        min-height: 2.25rem;
        margin: 0;
        padding: 0;
        /* Font ascenders make the two-line block look slightly low even when
           it is mathematically centered. This small optical correction makes
           the visible space above and below the text match. */
        transform: translateY(-0.16rem);
    }

    .derived-card-summary p,
    .derived-card-summary > div {
        margin: 0 !important;
        padding: 0 !important;
    }

    .derived-card-name {
        font-size: 1.08rem;
        font-weight: 750;
        color: var(--text-color);
        line-height: 1.25;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }

    .derived-card-equation {
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        font-size: 0.82rem;
        color: var(--text-color);
        opacity: 0.78;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
        line-height: 1.25;
        margin-top: 0;
    }

    [class*="st-key-derived-card-hidden-"] {
        opacity: 0.45;
    }

    /* Keep section titles and their add buttons together, even in a narrow sidebar. */
    .st-key-add-derived-column-header [data-testid="stHorizontalBlock"],
    .st-key-add-filter-header [data-testid="stHorizontalBlock"],
    .st-key-add-reference-line-header [data-testid="stHorizontalBlock"] {
        display: flex;
        flex-wrap: nowrap;
        align-items: center;
        gap: 0.35rem;
    }

    .st-key-add-derived-column-header [data-testid="stHorizontalBlock"] > [data-testid="stColumn"],
    .st-key-add-filter-header [data-testid="stHorizontalBlock"] > [data-testid="stColumn"],
    .st-key-add-reference-line-header [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
        min-width: 0 !important;
    }

    .st-key-add-derived-column-header [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:first-child,
    .st-key-add-filter-header [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:first-child,
    .st-key-add-reference-line-header [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:first-child {
        flex: 0 1 auto !important;
        width: auto !important;
    }

    .st-key-add-derived-column-header [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:last-child,
    .st-key-add-filter-header [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:last-child,
    .st-key-add-reference-line-header [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:last-child {
        flex: 0 0 2rem !important;
        width: 2rem !important;
    }

    .compact-add-header-title {
        font-size: 1.25rem;
        font-weight: 700;
        line-height: 1.7rem;
        white-space: nowrap;
    }

    /* Force true centering on all icon-only buttons */
    [class*="st-key-edit-derived-"] button,
    [class*="st-key-toggle-derived-visible-"] button,
    [class*="st-key-delete-derived-"] button,
    [class*="st-key-delete-filter-"] button,
    [class*="st-key-delete-reference-line-"] button,
    .st-key-add-derived-column button,
    .st-key-add-filter button,
    .st-key-add-reference-line button {
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
    }

    /* Remove the invisible margin Streamlit adds to the right of icons */
    [class*="st-key-edit-derived-"] button span,
    [class*="st-key-toggle-derived-visible-"] button span,
    [class*="st-key-delete-derived-"] button span,
    [class*="st-key-delete-filter-"] button span,
    [class*="st-key-delete-reference-line-"] button span,
    .st-key-add-derived-column button span,
    .st-key-add-filter button span,
    .st-key-add-reference-line button span {
        margin: 0 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@dataclass(frozen=True)
class FilterSpec:
    column: str
    mode: str
    lower: float | None = None
    upper: float | None = None
    value: str | None = None


@dataclass(frozen=True)
class DerivedSpec:
    name: str
    expression: str
    scope: str = "Global"


@dataclass(frozen=True)
class ReferenceLine:
    axis: str
    value: float
    label: str = ""
    color: str = "#d92d20"
    dash: str = "dash"


@dataclass(frozen=True)
class PlotStyle:
    single_color: str = "#2f80b7"
    color_scale: str = "Viridis"
    background_color: str = "#ffffff"
    trace_colors: dict[str, str] | None = None
    trace_order: list[str] | None = None
    reference_lines: list[ReferenceLine] | None = None
    point_label_column: str | None = None
    show_point_labels: bool = False
    show_highlighted_points: bool = True


@st.cache_data(show_spinner=False)
def csv_paths(data_dir: str) -> list[Path]:
    return sorted(Path(data_dir).glob("*.csv"))


@st.cache_data(show_spinner=False)
def data_workspace_paths(data_dir: str) -> list[Path]:
    root = Path(data_dir)
    workspaces = [
        path
        for path in sorted(root.iterdir())
        if path.is_dir() and any(path.glob("*.csv"))
    ]
    if any(root.glob("*.csv")):
        workspaces.insert(0, root)
    return workspaces


def data_workspace_label(path: Path) -> str:
    return DEFAULT_WORKSPACE if path == DATA_DIR else path.name.strip()


def custom_features_mtime() -> float | None:
    return CUSTOM_FEATURES_PATH.stat().st_mtime if CUSTOM_FEATURES_PATH.exists() else None


def apply_custom_catalogue_features(data: pd.DataFrame, path: str) -> pd.DataFrame:
    if not CUSTOM_FEATURES_PATH.exists():
        return data

    spec = importlib.util.spec_from_file_location("plot_builder_custom_features", CUSTOM_FEATURES_PATH)
    if spec is None or spec.loader is None:
        return data

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    augment = getattr(module, "augment_catalogue", None)
    if not callable(augment):
        return data

    augmented = augment(data.copy(), Path(path))
    if not isinstance(augmented, pd.DataFrame):
        raise TypeError("Custom augment_catalogue() must return a pandas DataFrame.")
    return augmented


@st.cache_data(show_spinner=False)
def read_csv(path: str, data_mtime: float, feature_mtime: float | None) -> pd.DataFrame:
    data = pd.read_csv(path)
    data = apply_custom_catalogue_features(data, path)
    data[SOURCE_COLUMN] = Path(path).name
    return data


def load_catalogues(paths: list[Path]) -> pd.DataFrame:
    feature_mtime = custom_features_mtime()
    frames = [read_csv(str(path), path.stat().st_mtime, feature_mtime) for path in paths]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def workspace_slug(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip()).strip("._-").lower()
    return slug or DEFAULT_WORKSPACE


def current_workspace_name() -> str:
    workspace_path = st.session_state.get("workspace-folder")
    if workspace_path:
        workspace = data_workspace_label(Path(workspace_path))
    else:
        workspace = str(st.session_state.get("workspace-name", DEFAULT_WORKSPACE)).strip()
    return workspace or DEFAULT_WORKSPACE


def current_state_path() -> Path:
    return STATE_DIR / f"{workspace_slug(current_workspace_name())}.json"


def clear_persistent_session_keys() -> None:
    exact_keys = {
        "derived-ids",
        "derived-next-id",
        "derived-sort-by",
        "filter-ids",
        "filter-next-id",
    }
    prefixes = (
        "derived-name-",
        "derived-formula-",
        "derived-scope-",
        "derived-visible-",
        "derived-applied-",
        "derived-edit-",
        "derived-error-",
        "derived-name-display-",
        "derived-formula-display-",
        "filter-column-",
        "filter-mode-",
        "filter-lower-",
        "filter-upper-",
        "filter-value-",
        "filter-enabled-",
    )
    for key in list(st.session_state):
        key_text = str(key)
        if key_text in exact_keys or any(key_text.startswith(prefix) for prefix in prefixes):
            del st.session_state[key]


def hydrate_persistent_state() -> None:
    workspace = workspace_slug(current_workspace_name())
    if st.session_state.get("_plot_builder_state_workspace") == workspace:
        return
    clear_persistent_session_keys()
    st.session_state["_plot_builder_state_workspace"] = workspace

    state_path = current_state_path()
    if not state_path.exists() and workspace == DEFAULT_WORKSPACE and LEGACY_STATE_PATH.exists():
        state_path = LEGACY_STATE_PATH

    if not state_path.exists():
        return

    try:
        state = json.loads(state_path.read_text())
    except (OSError, json.JSONDecodeError):
        return

    derived_items = state.get("derived", [])
    if isinstance(derived_items, list) and "derived-ids" not in st.session_state:
        derived_ids: list[int] = []
        for item in derived_items:
            if not isinstance(item, dict) or "id" not in item:
                continue
            derived_id = int(item["id"])
            derived_ids.append(derived_id)
            st.session_state[f"derived-name-{derived_id}"] = str(item.get("name", "derived"))
            st.session_state[f"derived-formula-{derived_id}"] = str(item.get("formula", ""))
            scope = str(item.get("scope", "Global"))
            st.session_state[f"derived-scope-{derived_id}"] = (
                scope if scope in DERIVED_SCOPES else "Global"
            )
            st.session_state[f"derived-visible-{derived_id}"] = bool(item.get("visible", True))
            st.session_state[f"derived-edit-{derived_id}"] = bool(item.get("edit", False))
            st.session_state[f"derived-error-{derived_id}"] = ""
            applied = item.get("applied")
            if isinstance(applied, dict):
                applied_scope = str(applied.get("scope", item.get("scope", "Global")))
                st.session_state[f"derived-applied-{derived_id}"] = {
                    "name": str(applied.get("name", "")),
                    "expression": str(applied.get("expression", "")),
                    "scope": applied_scope if applied_scope in DERIVED_SCOPES else "Global",
                }
        st.session_state["derived-ids"] = derived_ids
        st.session_state["derived-next-id"] = max(derived_ids, default=-1) + 1

    derived_sort = str(state.get("derived_sort", "Manual"))
    st.session_state["derived-sort-by"] = (
        derived_sort if derived_sort in DERIVED_SORT_OPTIONS else "Manual"
    )

    filter_items = state.get("filters", [])
    if isinstance(filter_items, list) and "filter-ids" not in st.session_state:
        filter_ids: list[int] = []
        for item in filter_items:
            if not isinstance(item, dict) or "id" not in item:
                continue
            filter_id = int(item["id"])
            filter_ids.append(filter_id)
            st.session_state[f"filter-column-{filter_id}"] = str(item.get("column", ""))
            st.session_state[f"filter-mode-{filter_id}"] = str(item.get("mode", "Interval"))
            st.session_state[f"filter-lower-{filter_id}"] = str(item.get("lower", "-inf"))
            st.session_state[f"filter-upper-{filter_id}"] = str(item.get("upper", "inf"))
            st.session_state[f"filter-value-{filter_id}"] = str(item.get("value", ""))
            st.session_state[f"filter-enabled-{filter_id}"] = bool(item.get("enabled", True))
        st.session_state["filter-ids"] = filter_ids
        st.session_state["filter-next-id"] = max(filter_ids, default=-1) + 1


def save_persistent_state() -> None:
    derived_items = []
    for derived_id in st.session_state.get("derived-ids", []):
        applied = st.session_state.get(f"derived-applied-{derived_id}")
        derived_items.append(
            {
                "id": derived_id,
                "name": st.session_state.get(f"derived-name-{derived_id}", "derived"),
                "formula": st.session_state.get(f"derived-formula-{derived_id}", ""),
                "scope": st.session_state.get(f"derived-scope-{derived_id}", "Global"),
                "visible": bool(st.session_state.get(f"derived-visible-{derived_id}", True)),
                "edit": bool(st.session_state.get(f"derived-edit-{derived_id}", False)),
                "applied": applied if isinstance(applied, dict) else None,
            }
        )

    filter_items = []
    for filter_id in st.session_state.get("filter-ids", []):
        filter_items.append(
            {
                "id": filter_id,
                "column": st.session_state.get(f"filter-column-{filter_id}", ""),
                "mode": st.session_state.get(f"filter-mode-{filter_id}", "Interval"),
                "lower": st.session_state.get(f"filter-lower-{filter_id}", "-inf"),
                "upper": st.session_state.get(f"filter-upper-{filter_id}", "inf"),
                "value": st.session_state.get(f"filter-value-{filter_id}", ""),
                "enabled": bool(st.session_state.get(f"filter-enabled-{filter_id}", True)),
            }
        )

    state = {
        "derived": derived_items,
        "derived_sort": st.session_state.get("derived-sort-by", "Manual"),
        "filters": filter_items,
        "workspace": current_workspace_name(),
    }
    try:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        current_state_path().write_text(json.dumps(state, indent=2, sort_keys=True))
    except OSError:
        pass


def numeric_columns(data: pd.DataFrame) -> list[str]:
    return [
        column
        for column in data.columns
        if pd.api.types.is_numeric_dtype(data[column]) and data[column].notna().any()
    ]


def plottable_columns(data: pd.DataFrame) -> list[str]:
    return [column for column in data.columns if data[column].notna().any()]


def column_token(column: str) -> str:
    return f"`{column}`"


def clean_derived_name(name: str) -> str:
    cleaned = str(name).strip()
    if not cleaned:
        raise ValueError("Derived column names cannot be blank.")
    return cleaned


def prepare_eval_expression(
    data: pd.DataFrame,
    expression: str,
) -> tuple[str, dict[str, Any]]:
    local_dict: dict[str, Any] = dict(ALLOWED_FUNCTIONS)

    def unique(series: pd.Series) -> float:
        values = pd.to_numeric(series, errors="coerce").dropna().unique()
        if len(values) != 1:
            raise ValueError(
                f"`unique()` needs exactly one non-missing numeric value, but found {len(values)}."
            )
        return float(values[0])

    def lookup_unique(
        value_series: pd.Series,
        key_series: pd.Series,
        key_value: Any,
    ) -> float:
        if pd.api.types.is_numeric_dtype(key_series):
            try:
                mask = pd.to_numeric(key_series, errors="coerce") == float(key_value)
            except (TypeError, ValueError):
                mask = key_series.astype(str) == str(key_value)
        else:
            mask = key_series.astype(str) == str(key_value)

        values = pd.to_numeric(value_series.loc[mask], errors="coerce").dropna().unique()
        if len(values) != 1:
            raise ValueError(
                f"`lookup_unique()` needs exactly one matching non-missing numeric value, "
                f"but found {len(values)}."
            )
        return float(values[0])

    def row_value(series: pd.Series, row_number: Any) -> float:
        try:
            row_index = int(row_number) - 1
        except (TypeError, ValueError) as exc:
            raise ValueError("`row_value()` row number must be a positive integer.") from exc

        if row_index < 0:
            raise ValueError("`row_value()` row number must be 1 or greater.")
        if row_index >= len(series):
            raise ValueError(
                f"`row_value()` row {row_index + 1} is outside the current data with {len(series)} rows."
            )

        value = pd.to_numeric(pd.Series([series.iloc[row_index]]), errors="coerce").iloc[0]
        if pd.isna(value):
            raise ValueError(f"`row_value()` row {row_index + 1} is missing or non-numeric.")
        return float(value)

    def first_value(series: pd.Series) -> float:
        return row_value(series, 1)

    local_dict["unique"] = unique
    local_dict["lookup_unique"] = lookup_unique
    local_dict["row_value"] = row_value
    local_dict["first_value"] = first_value
    column_names: dict[str, str] = {}

    def replace_column_token(match: re.Match[str]) -> str:
        column = match.group(1)
        if column not in data.columns:
            raise ValueError(f"Column `{column}` is not in the selected data.")
        variable = column_names.get(column)
        if variable is None:
            variable = f"__col_{len(column_names)}"
            column_names[column] = variable
            local_dict[variable] = pd.to_numeric(data[column], errors="coerce")
        return variable

    prepared_expression = re.sub(r"`([^`]+)`", replace_column_token, expression)
    return prepared_expression, local_dict


def expression_column_tokens(expression: str) -> set[str]:
    return set(re.findall(r"`([^`]+)`", expression))


def evaluate_derived_expression(data: pd.DataFrame, name: str, expression: str) -> pd.Series:
    try:
        prepared_expression, local_dict = prepare_eval_expression(data, expression)
        result = pd.eval(prepared_expression, engine="python", local_dict=local_dict)
    except Exception as exc:
        msg = (
            f"Could not evaluate `{name}`. Use the column buttons in the formula so "
            "column names are inserted with pandas backticks. "
            f"Details: {exc}"
        )
        raise ValueError(msg) from exc

    if np.isscalar(result):
        series = pd.Series(result, index=data.index)
    else:
        series = pd.Series(result, index=data.index)
    return pd.to_numeric(series, errors="coerce")


def evaluate_derived_columns(data: pd.DataFrame, specs: list[DerivedSpec]) -> pd.DataFrame:
    if not specs:
        return data

    derived = data.copy()
    used_output_names = set(data.columns)
    derived_names = set()
    cleaned_specs: list[DerivedSpec] = []

    for spec in specs:
        name = clean_derived_name(spec.name)
        expression = spec.expression.strip()
        if not expression:
            continue
        if name in used_output_names:
            raise ValueError(
                f"Derived column `{name}` already exists. Choose a new name so the original "
                "CSV columns stay unambiguous."
            )
        if name in derived_names:
            raise ValueError(f"Derived column `{name}` is defined more than once.")
        derived_names.add(name)
        cleaned_specs.append(DerivedSpec(name=name, expression=expression, scope=spec.scope))
        used_output_names.add(name)

    pending = cleaned_specs.copy()
    while pending:
        progressed = False
        next_pending: list[DerivedSpec] = []

        for spec in pending:
            missing_columns = expression_column_tokens(spec.expression) - set(derived.columns)
            unresolved_derived_columns = missing_columns & derived_names
            unknown_columns = missing_columns - derived_names
            if unknown_columns:
                missing = ", ".join(f"`{column}`" for column in sorted(unknown_columns))
                raise ValueError(
                    f"Could not evaluate `{spec.name}`. These columns are not in the selected "
                    f"data or applied derived columns: {missing}."
                )
            if unresolved_derived_columns:
                next_pending.append(spec)
                continue

            if spec.scope == "Per catalogue" and SOURCE_COLUMN in derived.columns:
                series_parts = []
                for source, group in derived.groupby(SOURCE_COLUMN, sort=False):
                    try:
                        series_parts.append(
                            evaluate_derived_expression(group, spec.name, spec.expression)
                        )
                    except ValueError as exc:
                        raise ValueError(f"{exc} Catalogue: {source}.") from exc
                derived[spec.name] = pd.concat(series_parts).reindex(derived.index)
            else:
                derived[spec.name] = evaluate_derived_expression(
                    derived,
                    spec.name,
                    spec.expression,
                )
            progressed = True

        if not progressed:
            blocked = ", ".join(f"`{spec.name}`" for spec in next_pending)
            raise ValueError(
                "Could not resolve derived-column dependencies. Check for a circular reference "
                f"or a missing intermediate derived column involving: {blocked}."
            )
        pending = next_pending

    return derived


def parse_bound(value: str, default: float) -> float:
    text = value.strip().lower()
    if not text:
        return default
    if text in {"inf", "+inf", "infinity", "+infinity"}:
        return np.inf
    if text in {"-inf", "-infinity"}:
        return -np.inf
    return float(text)


def coerce_filter_value(series: pd.Series, value: str) -> Any:
    if pd.api.types.is_numeric_dtype(series):
        if not value.strip():
            return np.nan
        return float(value)
    return value


def apply_filters(data: pd.DataFrame, filters: list[FilterSpec]) -> pd.DataFrame:
    filtered = data.copy()
    for filter_spec in filters:
        if filter_spec.column not in filtered.columns:
            continue

        series = filtered[filter_spec.column]
        if filter_spec.mode == "Interval":
            numeric = pd.to_numeric(series, errors="coerce")
            lower = -np.inf if filter_spec.lower is None else filter_spec.lower
            upper = np.inf if filter_spec.upper is None else filter_spec.upper
            filtered = filtered.loc[numeric.between(lower, upper, inclusive="both")]
        elif filter_spec.mode == "Equals":
            try:
                value = coerce_filter_value(series, filter_spec.value or "")
            except ValueError as exc:
                msg = f"Filter value for `{filter_spec.column}` must be numeric."
                raise ValueError(msg) from exc
            filtered = filtered.loc[series == value]
        elif filter_spec.mode == "Not equals":
            try:
                value = coerce_filter_value(series, filter_spec.value or "")
            except ValueError as exc:
                msg = f"Filter value for `{filter_spec.column}` must be numeric."
                raise ValueError(msg) from exc
            filtered = filtered.loc[series != value]
    return filtered


def format_filter_value(value: float | str | None) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if np.isneginf(value):
            return "-inf"
        if np.isposinf(value):
            return "inf"
        return f"{value:.4g}"
    return str(value)


def filter_legend_lines(filters: list[FilterSpec]) -> list[str]:
    lines = []
    for filter_spec in filters:
        if filter_spec.mode == "Interval":
            lines.append(
                f"{filter_spec.column}: "
                f"{format_filter_value(filter_spec.lower)} <= x <= "
                f"{format_filter_value(filter_spec.upper)}"
            )
        elif filter_spec.mode == "Equals":
            lines.append(f"{filter_spec.column}: = {format_filter_value(filter_spec.value)}")
        elif filter_spec.mode == "Not equals":
            lines.append(f"{filter_spec.column}: != {format_filter_value(filter_spec.value)}")
    return lines


def add_filter_legend(fig: go.Figure, filters: list[FilterSpec], background_color: str) -> None:
    lines = filter_legend_lines(filters)
    if not lines:
        return

    background_color = normalized_hex_color(background_color, "#ffffff")
    axis_colors = readable_axis_colors(background_color)
    red, green, blue = hex_to_rgb(background_color)
    fill = f"rgba({red},{green},{blue},0.84)"
    border = "rgba(0,0,0,0.26)" if axis_colors["text"] == "black" else "rgba(255,255,255,0.34)"
    text = "<br>".join(["Filters", *[html.escape(line) for line in lines]])
    fig.add_annotation(
        text=text,
        xref="paper",
        yref="paper",
        x=0.012,
        y=0.988,
        xanchor="left",
        yanchor="top",
        align="left",
        showarrow=False,
        bgcolor=fill,
        bordercolor=border,
        borderwidth=1,
        borderpad=6,
        font={"color": axis_colors["text"], "size": 12},
    )
    fig.update_layout(margin={"l": 0, "r": 0, "b": 0, "t": 40 + (16 * len(lines))})


def add_reference_lines(fig: go.Figure, style: PlotStyle, has_z_axis: bool = False) -> None:
    if has_z_axis:
        return

    axis_colors = readable_axis_colors(style.background_color)
    for line in style.reference_lines or []:
        line_style = {
            "color": normalized_hex_color(line.color, "#d92d20"),
            "width": 2,
            "dash": line.dash,
        }
        annotation = line.label.strip() or None
        if line.axis == "X":
            fig.add_vline(
                x=line.value,
                line=line_style,
                annotation_text=annotation,
                annotation_position="top",
                annotation_font_color=axis_colors["text"],
            )
        elif line.axis == "Y":
            fig.add_hline(
                y=line.value,
                line=line_style,
                annotation_text=annotation,
                annotation_position="right",
                annotation_font_color=axis_colors["text"],
            )


def default_trace_color(index: int) -> str:
    return DEFAULT_TRACE_COLORS[index % len(DEFAULT_TRACE_COLORS)]


def marker_style(data: pd.DataFrame, color_column: str | None, style: PlotStyle) -> dict[str, Any]:
    marker: dict[str, Any] = {"size": 7, "opacity": 0.72}
    if color_column is None:
        marker["color"] = normalized_hex_color(style.single_color, default_trace_color(0))
        return marker

    if pd.api.types.is_numeric_dtype(data[color_column]):
        colorbar_text_color = readable_axis_colors(style.background_color)["text"]
        marker.update(
            {
                "color": data[color_column],
                "colorscale": style.color_scale,
                "colorbar": {
                    "title": {"text": color_column, "font": {"color": colorbar_text_color}},
                    "tickfont": {"color": colorbar_text_color},
                },
            }
        )
    else:
        codes, labels = pd.factorize(data[color_column].astype(str), sort=True)
        marker.update(
            {
                "color": codes,
                "colorscale": "Viridis",
                "colorbar": {
                    "title": {"text": color_column, "font": {"color": "black"}},
                    "tickvals": list(range(len(labels))),
                    "ticktext": [str(label) for label in labels],
                    "tickfont": {"color": "black"},
                },
            }
        )
    return marker


def layer_points_by_color_rarity(
    data: pd.DataFrame,
    color_column: str | None,
    bin_count: int = 24,
) -> pd.DataFrame:
    """Draw common color bins first and rarer bins last so minority colors stay visible."""
    if color_column is None or color_column not in data.columns or len(data) < 2:
        return data

    color_values = data[color_column]
    if color_values.nunique(dropna=True) <= 1:
        return data

    layered = data.copy()
    if pd.api.types.is_numeric_dtype(color_values):
        numeric_values = pd.to_numeric(layered[color_column], errors="coerce")
        unique_count = numeric_values.nunique(dropna=True)
        if unique_count <= bin_count:
            color_groups = numeric_values
        else:
            color_groups = pd.cut(
                numeric_values,
                bins=min(bin_count, unique_count),
                duplicates="drop",
            )
    else:
        color_groups = layered[color_column].astype(str)

    color_counts = color_groups.map(color_groups.value_counts(dropna=False))
    layered["_plot_layer_color_count"] = color_counts.to_numpy()
    layered["_plot_layer_original_order"] = np.arange(len(layered))
    layered = layered.sort_values(
        ["_plot_layer_color_count", "_plot_layer_original_order"],
        ascending=[False, True],
        kind="mergesort",
    )
    return layered.drop(columns=["_plot_layer_color_count", "_plot_layer_original_order"])


def hex_to_rgb(color: str) -> tuple[int, int, int]:
    text = color.strip().lstrip("#")
    if len(text) == 3:
        text = "".join(character * 2 for character in text)
    if len(text) != 6:
        return (255, 255, 255)
    try:
        return tuple(int(text[index : index + 2], 16) for index in (0, 2, 4))
    except ValueError:
        return (255, 255, 255)


def is_hex_color(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"#[0-9A-Fa-f]{6}", value.strip()) is not None


def normalized_hex_color(value: Any, fallback: str) -> str:
    if is_hex_color(value):
        return str(value).strip()
    return fallback if is_hex_color(fallback) else "#ffffff"


def readable_axis_colors(background_color: str) -> dict[str, str]:
    background_color = normalized_hex_color(background_color, "#ffffff")
    red, green, blue = hex_to_rgb(background_color)
    luminance = (0.2126 * red + 0.7152 * green + 0.0722 * blue) / 255
    if luminance > 0.55:
        return {
            "text": "black",
            "grid": "rgba(0,0,0,0.18)",
            "line": "rgba(0,0,0,0.45)",
            "zero": "rgba(0,0,0,0.55)",
        }
    return {
        "text": "white",
        "grid": "rgba(255,255,255,0.24)",
        "line": "rgba(255,255,255,0.58)",
        "zero": "rgba(255,255,255,0.70)",
    }


def style_plot(fig: go.Figure, has_z_axis: bool = False, background_color: str = "#ffffff") -> None:
    background_color = normalized_hex_color(background_color, "#ffffff")
    axis_colors = readable_axis_colors(background_color)
    fig.update_layout(
        paper_bgcolor=background_color,
        plot_bgcolor=background_color,
        font={"color": axis_colors["text"]},
        margin={"l": 0, "r": 0, "b": 0, "t": 24},
    )
    fig.update_xaxes(
        color=axis_colors["text"],
        gridcolor=axis_colors["grid"],
        linecolor=axis_colors["line"],
        zerolinecolor=axis_colors["zero"],
        title_font={"color": axis_colors["text"]},
        tickfont={"color": axis_colors["text"]},
    )
    fig.update_yaxes(
        color=axis_colors["text"],
        gridcolor=axis_colors["grid"],
        linecolor=axis_colors["line"],
        zerolinecolor=axis_colors["zero"],
        title_font={"color": axis_colors["text"]},
        tickfont={"color": axis_colors["text"]},
    )
    if has_z_axis:
        fig.update_layout(
            scene={
                "bgcolor": background_color,
                "xaxis": {
                    "backgroundcolor": background_color,
                    "color": axis_colors["text"],
                    "gridcolor": axis_colors["grid"],
                    "linecolor": axis_colors["line"],
                    "zerolinecolor": axis_colors["zero"],
                    "title": {"font": {"color": axis_colors["text"]}},
                    "tickfont": {"color": axis_colors["text"]},
                },
                "yaxis": {
                    "backgroundcolor": background_color,
                    "color": axis_colors["text"],
                    "gridcolor": axis_colors["grid"],
                    "linecolor": axis_colors["line"],
                    "zerolinecolor": axis_colors["zero"],
                    "title": {"font": {"color": axis_colors["text"]}},
                    "tickfont": {"color": axis_colors["text"]},
                },
                "zaxis": {
                    "backgroundcolor": background_color,
                    "color": axis_colors["text"],
                    "gridcolor": axis_colors["grid"],
                    "linecolor": axis_colors["line"],
                    "zerolinecolor": axis_colors["zero"],
                    "title": {"font": {"color": axis_colors["text"]}},
                    "tickfont": {"color": axis_colors["text"]},
                },
            }
        )


def missing_plot_values_message(data: pd.DataFrame, required_columns: list[str]) -> str:
    present_counts = {
        column: int(data[column].notna().sum()) for column in required_columns if column in data.columns
    }
    overlap_count = int(data.dropna(subset=required_columns).shape[0])
    message = [
        "No rows have values for all selected plotting columns at the same time.",
        f"Selected columns: {', '.join(f'`{column}`' for column in required_columns)}.",
        f"Rows with all selected columns present: {overlap_count:,}.",
        "Rows with each column present: "
        + ", ".join(f"`{column}`: {count:,}" for column, count in present_counts.items())
        + ".",
    ]

    if SOURCE_COLUMN in data.columns:
        files_with_overlap = (
            data.dropna(subset=required_columns)[SOURCE_COLUMN].drop_duplicates().astype(str).tolist()
        )
        if files_with_overlap:
            message.append("Files with all selected columns: " + ", ".join(files_with_overlap) + ".")
        else:
            message.append(
                "No selected CSV file contains non-missing values for all of those columns together. "
                "Try `One` catalogue mode, or choose columns that exist in the same CSV file."
            )

    return " ".join(message)


def add_highlight_point_traces(
    fig: go.Figure,
    plot_data: pd.DataFrame,
    x_column: str,
    y_column: str,
    z_column: str | None,
    enabled: bool,
) -> None:
    label_column = "plot_highlight_label"
    color_column = "plot_highlight_color"
    if not enabled or label_column not in plot_data.columns:
        return

    labels = plot_data[label_column].fillna("").astype(str)
    highlighted = plot_data[labels.str.len() > 0].copy()
    if highlighted.empty:
        return

    colors = (
        highlighted[color_column].fillna("").astype(str)
        if color_column in highlighted.columns
        else pd.Series("#000000", index=highlighted.index)
    )
    highlighted["_plot_highlight_color"] = [
        normalized_hex_color(color, "#000000") for color in colors
    ]

    groups = highlighted.groupby([label_column, "_plot_highlight_color"], sort=False)
    for (label, color), points in groups:
        if points.empty:
            continue

        hover = (
            f"{label}<br>"
            f"{x_column}: %{{x:.4g}}<br>"
            f"{y_column}: %{{y:.4g}}"
        )
        if SOURCE_COLUMN in points.columns:
            customdata = points[[SOURCE_COLUMN]].astype(str).to_numpy()
            hover = f"{SOURCE_COLUMN}: %{{customdata[0]}}<br>" + hover
        else:
            customdata = None

        marker = {
            "size": 10 if z_column else 13,
            "color": color,
            "opacity": 1.0,
            "line": {"color": "white", "width": 1.5},
        }
        if z_column:
            fig.add_trace(
                go.Scatter3d(
                    x=points[x_column],
                    y=points[y_column],
                    z=points[z_column],
                    mode="markers+text",
                    marker=marker,
                    text=[label] * len(points),
                    customdata=customdata,
                    hovertemplate=hover + f"<br>{z_column}: %{{z:.4g}}<extra></extra>",
                    name=label,
                    showlegend=False,
                )
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=points[x_column],
                    y=points[y_column],
                    mode="markers+text",
                    marker=marker,
                    text=[label] * len(points),
                    textposition="top center",
                    customdata=customdata,
                    hovertemplate=hover + "<extra></extra>",
                    name=label,
                    showlegend=False,
                )
            )


def scatter_figure(
    data: pd.DataFrame,
    x_column: str,
    y_column: str,
    z_column: str | None,
    color_column: str | None,
    style: PlotStyle,
) -> go.Figure:
    required_columns = [x_column, y_column]
    if z_column:
        required_columns.append(z_column)
    if color_column:
        required_columns.append(color_column)

    plot_data = data.dropna(subset=required_columns)
    if plot_data.empty:
        raise ValueError(missing_plot_values_message(data, required_columns))

    custom_columns = [column for column in [SOURCE_COLUMN] if column in plot_data]
    label_column = style.point_label_column
    if label_column and label_column in plot_data.columns and label_column not in custom_columns:
        custom_columns.append(label_column)
    hover = f"{x_column}: %{{x:.4g}}<br>{y_column}: %{{y:.4g}}"
    custom_indices = {column: index for index, column in enumerate(custom_columns)}
    if custom_columns:
        hover_prefixes = []
        if SOURCE_COLUMN in custom_indices:
            hover_prefixes.append(f"{SOURCE_COLUMN}: %{{customdata[{custom_indices[SOURCE_COLUMN]}]}}")
        if label_column and label_column in custom_indices:
            hover_prefixes.append(f"{label_column}: %{{customdata[{custom_indices[label_column]}]}}")
        if hover_prefixes:
            hover = "<br>".join(hover_prefixes) + "<br>" + hover
    if color_column:
        hover += f"<br>{color_column}: %{{marker.color}}"
    hover += "<extra></extra>"

    fig = go.Figure()
    categorical_color = (
        color_column is not None and not pd.api.types.is_numeric_dtype(plot_data[color_column])
    )
    if categorical_color:
        trace_colors = style.trace_colors or {}
        color_values = sorted(plot_data[color_column].dropna().astype(str).unique())
        default_color_indices = {value: index for index, value in enumerate(color_values)}
        groups = sorted(
            plot_data.groupby(color_column, sort=True),
            key=lambda item: (-len(item[1]), str(item[0])),
        )
        for value, group in groups:
            trace_index = default_color_indices.get(str(value), 0)
            color = normalized_hex_color(
                trace_colors.get(str(value), default_trace_color(trace_index)),
                default_trace_color(trace_index),
            )
            marker = {"size": 5 if z_column else 7, "opacity": 0.72, "color": color}
            customdata = group[custom_columns].astype(str).to_numpy() if custom_columns else None
            text = (
                group[label_column].astype(str)
                if label_column and style.show_point_labels and label_column in group.columns
                else None
            )
            group_hover = hover.replace(f"{color_column}: %{{marker.color}}", f"{color_column}: {value}")
            if z_column:
                fig.add_trace(
                    go.Scatter3d(
                        x=group[x_column],
                        y=group[y_column],
                        z=group[z_column],
                        mode="markers+text" if text is not None else "markers",
                        marker=marker,
                        text=text,
                        customdata=customdata,
                        hovertemplate=group_hover.replace(
                            "<extra></extra>", f"<br>{z_column}: %{{z:.4g}}<extra></extra>"
                        ),
                        name=str(value),
                    )
                )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=group[x_column],
                        y=group[y_column],
                        mode="markers+text" if text is not None else "markers",
                        marker=marker,
                        text=text,
                        textposition="top center",
                        customdata=customdata,
                        hovertemplate=group_hover,
                        name=str(value),
                    )
                )
    else:
        plot_data = layer_points_by_color_rarity(plot_data, color_column)
        marker = marker_style(plot_data, color_column, style)
        if z_column:
            marker["size"] = 5

        customdata = plot_data[custom_columns].astype(str).to_numpy() if custom_columns else None
        text = (
            plot_data[label_column].astype(str)
            if label_column and style.show_point_labels and label_column in plot_data.columns
            else None
        )
        if z_column:
            fig.add_trace(
                go.Scatter3d(
                    x=plot_data[x_column],
                    y=plot_data[y_column],
                    z=plot_data[z_column],
                    mode="markers+text" if text is not None else "markers",
                    marker=marker,
                    text=text,
                    customdata=customdata,
                    hovertemplate=hover.replace(
                        "<extra></extra>", f"<br>{z_column}: %{{z:.4g}}<extra></extra>"
                    ),
                    name="Rows",
                )
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=plot_data[x_column],
                    y=plot_data[y_column],
                    mode="markers+text" if text is not None else "markers",
                    marker=marker,
                    text=text,
                    textposition="top center",
                    customdata=customdata,
                    hovertemplate=hover,
                    name="Rows",
                )
            )

    if z_column:
        fig.update_layout(
            scene={
                "xaxis_title": x_column,
                "yaxis_title": y_column,
                "zaxis_title": z_column,
                "aspectmode": "data",
            }
        )
    else:
        fig.update_layout(xaxis_title=x_column, yaxis_title=y_column)

    add_highlight_point_traces(
        fig,
        plot_data,
        x_column,
        y_column,
        z_column,
        style.show_highlighted_points,
    )
    fig.update_layout(showlegend=False)
    style_plot(fig, has_z_axis=z_column is not None, background_color=style.background_color)
    add_reference_lines(fig, style, has_z_axis=z_column is not None)
    return fig


def histogram_figure(
    data: pd.DataFrame,
    column: str,
    bins: int,
    split_column: str | None,
    style: PlotStyle,
) -> go.Figure:
    plot_data = data.dropna(subset=[column])
    if plot_data.empty:
        raise ValueError("No rows remain after dropping missing histogram values.")

    fig = go.Figure()
    if split_column and split_column in plot_data.columns:
        trace_colors = style.trace_colors or {}
        groups = [
            (str(value), group)
            for value, group in plot_data.groupby(split_column, dropna=False)
        ]
        if style.trace_order:
            order = {value: index for index, value in enumerate(style.trace_order)}
            groups.sort(key=lambda item: order.get(item[0], len(order)))

        for trace_index, (value, group) in enumerate(groups):
            fig.add_trace(
                go.Histogram(
                    x=group[column],
                    nbinsx=bins,
                    opacity=0.62,
                    marker={
                        "color": normalized_hex_color(
                            trace_colors.get(str(value), default_trace_color(trace_index)),
                            default_trace_color(trace_index),
                        )
                    },
                    name=str(value),
                )
            )
        fig.update_layout(barmode="overlay")
    else:
        fig.add_trace(
            go.Histogram(
                x=plot_data[column],
                nbinsx=bins,
                marker={"color": normalized_hex_color(style.single_color, default_trace_color(0))},
            )
        )

    fig.update_layout(xaxis_title=column, yaxis_title="Count")
    style_plot(fig, background_color=style.background_color)
    add_reference_lines(fig, style)
    return fig


def build_plot_figure(
    data: pd.DataFrame,
    plot_type: str,
    style: PlotStyle,
    *,
    x_column: str | None = None,
    y_column: str | None = None,
    z_column: str | None = None,
    color_column: str | None = None,
    hist_column: str | None = None,
    bins: int | None = None,
    split_column: str | None = None,
) -> go.Figure:
    if plot_type == "Scatter":
        if x_column is None or y_column is None:
            raise ValueError("Scatter plots need X and Y columns.")
        return scatter_figure(data, x_column, y_column, z_column, color_column, style)

    if hist_column is None or bins is None:
        raise ValueError("Histograms need a column and bin count.")
    return histogram_figure(data, hist_column, bins, split_column, style)


def plot_html_document(figures: list[tuple[str, go.Figure]]) -> str:
    plot_divs = []
    for index, (title, fig) in enumerate(figures):
        plot_divs.append(f"<h2>{html.escape(title)}</h2>")
        plot_divs.append(
            fig.to_html(
                include_plotlyjs="cdn" if index == 0 else False,
                full_html=False,
            )
        )

    return "\n".join(
        [
            "<!doctype html>",
            "<html>",
            "<head>",
            "<meta charset='utf-8'>",
            "<title>Plot Builder</title>",
            "<style>body{font-family:system-ui,sans-serif;margin:24px;}h2{margin-top:32px;}</style>",
            "</head>",
            "<body>",
            *plot_divs,
            "</body>",
            "</html>",
        ]
    )


def safe_filename_stem(name: str, fallback: str) -> str:
    stem = Path(name.strip()).stem if name.strip() else fallback
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._-")
    return stem or fallback


def figure_for_export(fig: go.Figure, title: str = "") -> go.Figure:
    export_fig = go.Figure(fig)
    title = title.strip()
    top_margin = 125 if title else 95
    export_fig.update_layout(
        width=1200,
        height=850,
        margin={"l": 110, "r": 150, "b": 95, "t": top_margin},
    )
    if title:
        export_fig.update_layout(
            title={
                "text": title,
                "x": 0.5,
                "xanchor": "center",
                "font": {"size": 24},
            }
        )
    export_fig.update_xaxes(
        automargin=True,
        showticklabels=True,
        ticks="outside",
        title_standoff=18,
    )
    export_fig.update_yaxes(
        automargin=True,
        showticklabels=True,
        ticks="outside",
        title_standoff=18,
    )
    has_3d_trace = any(str(getattr(trace, "type", "")).endswith("3d") for trace in export_fig.data)
    if has_3d_trace:
        export_fig.update_layout(margin={"l": 70, "r": 70, "b": 70, "t": 120 if title else 80})
        export_fig.update_layout(
            scene={
                "xaxis": {"showticklabels": True, "ticks": "outside"},
                "yaxis": {"showticklabels": True, "ticks": "outside"},
                "zaxis": {"showticklabels": True, "ticks": "outside"},
            }
        )
    return export_fig


def save_plot_file(fig: go.Figure, file_stem: str, file_format: str, title: str = "") -> Path:
    SAVED_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    extension = file_format.lower()
    path = SAVED_PLOTS_DIR / f"{file_stem}.{extension}"
    export_fig = figure_for_export(fig, title)
    if file_format == "PNG":
        export_fig.write_image(path, width=1200, height=850, scale=2)
    else:
        export_fig.write_html(path, include_plotlyjs="cdn", full_html=True)
    return path


def save_plot_controls(fig: go.Figure, default_name: str, key_prefix: str) -> None:
    filename_key = f"{key_prefix}-save-filename"
    title_key = f"{key_prefix}-save-title"
    format_key = f"{key_prefix}-save-format"
    if filename_key not in st.session_state:
        st.session_state[filename_key] = safe_filename_stem(default_name, "plot")
    default_format_index = 1

    save_cols = st.columns([0.34, 0.30, 0.14, 0.22], gap="small", vertical_alignment="bottom")
    with save_cols[0]:
        filename = st.text_input("Save filename", key=filename_key)
    with save_cols[1]:
        plot_title = st.text_input("Plot title", key=title_key)
    with save_cols[2]:
        file_format = stable_selectbox(
            "Format",
            ["PNG", "HTML"],
            key=format_key,
            default_index=default_format_index,
        )
    with save_cols[3]:
        save_clicked = st.button(
            "Save Plot",
            key=f"{key_prefix}-save-button",
            icon=":material/save:",
            width="stretch",
        )

    if not save_clicked:
        return

    file_stem = safe_filename_stem(filename, "plot")
    try:
        path = save_plot_file(fig, file_stem, file_format, plot_title)
    except ValueError as exc:
        if file_format == "PNG" and "kaleido" in str(exc).lower():
            st.error(
                "PNG export needs the `kaleido` package in the venv. "
                "Choose HTML for now, or install Kaleido to enable PNG saving."
            )
            return
        st.error(f"Could not save plot: {exc}")
        return
    except Exception as exc:
        st.error(f"Could not save plot: {exc}")
        return

    st.success(f"Saved to `{path.as_posix()}`")


def saved_plot_paths() -> list[Path]:
    if not SAVED_PLOTS_DIR.exists():
        return []
    return sorted(
        (
            path
            for path in SAVED_PLOTS_DIR.iterdir()
            if path.is_file() and path.suffix.lower() in {".html", ".png"}
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


@st.dialog("Saved Plots", width="large")
def saved_plots_dialog() -> None:
    paths = saved_plot_paths()
    if not paths:
        st.info("No saved plots yet.")
        return

    st.caption(f"{len(paths)} saved plot{'s' if len(paths) != 1 else ''} in `{SAVED_PLOTS_DIR.as_posix()}`")
    with st.container(height=620):
        for path in paths:
            modified = pd.Timestamp.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
            with st.container(border=True):
                name_col, action_col = st.columns([0.72, 0.28], gap="small", vertical_alignment="center")
                with name_col:
                    st.markdown(f"**{path.name}**")
                    st.caption(f"{path.suffix.upper().lstrip('.')} • {path.stat().st_size / 1024:.1f} KB • {modified}")
                with action_col:
                    st.download_button(
                        "Download",
                        data=path.read_bytes(),
                        file_name=path.name,
                        mime="image/png" if path.suffix.lower() == ".png" else "text/html",
                        key=f"download-saved-plot-{hashlib.sha1(path.name.encode('utf-8')).hexdigest()[:12]}",
                        width="stretch",
                    )

                if path.suffix.lower() == ".png":
                    st.image(path.read_bytes(), use_container_width=True)
                else:
                    components.html(path.read_text(), height=420, scrolling=True)


def compact_add_header(
    title: str,
    button_key: str,
    help_text: str,
    on_click: Any,
    disabled: bool = False,
    args: tuple[Any, ...] | None = None,
) -> None:
    # The keyed container is styled as a content-sized flex row above. This keeps the
    # button immediately after the title and prevents longer titles from wrapping.
    with st.container(key=f"{button_key}-header"):
        title_col, add_col = st.columns(
            [1, 0.12],
            gap="small",
            vertical_alignment="center",
        )

        title_col.markdown(
            f"<div class='compact-add-header-title'>{html.escape(title)}</div>",
            unsafe_allow_html=True,
        )

        add_col.button(
            "Add", # The text label (will be hidden by CSS)
            icon=":material/add:", # Use the native Material icon
            key=button_key,
            help=help_text,
            on_click=on_click,
            args=args,
            disabled=disabled,
            width="content",
        )

def initialize_filter_list() -> None:
    hydrate_persistent_state()
    if "filter-ids" not in st.session_state:
        st.session_state["filter-ids"] = []
    if "filter-next-id" not in st.session_state:
        st.session_state["filter-next-id"] = 0

    migrated_ids = sorted(
        {
            int(match.group(1))
            for key in st.session_state
            if (
                match := re.fullmatch(
                    r"filter-(?:column|mode|lower|upper|value|enabled)-(\d+)",
                    str(key),
                )
            )
        }
    )
    if migrated_ids and not st.session_state["filter-ids"]:
        st.session_state["filter-ids"] = migrated_ids
        st.session_state["filter-next-id"] = max(migrated_ids) + 1


def add_filter(columns: list[str]) -> None:
    initialize_filter_list()
    filter_ids = st.session_state["filter-ids"]
    if len(filter_ids) >= MAX_FILTERS or not columns:
        return

    filter_id = st.session_state["filter-next-id"]
    st.session_state["filter-next-id"] = filter_id + 1
    filter_ids.append(filter_id)
    st.session_state[f"filter-column-{filter_id}"] = columns[0]
    st.session_state[f"filter-mode-{filter_id}"] = "Interval"
    st.session_state[f"filter-lower-{filter_id}"] = "-inf"
    st.session_state[f"filter-upper-{filter_id}"] = "inf"
    st.session_state[f"filter-value-{filter_id}"] = ""
    st.session_state[f"filter-enabled-{filter_id}"] = True


def delete_filter(filter_id: int) -> None:
    initialize_filter_list()
    st.session_state["filter-ids"] = [
        existing_id for existing_id in st.session_state["filter-ids"] if existing_id != filter_id
    ]
    for prefix in [
        "filter-column",
        "filter-mode",
        "filter-lower",
        "filter-upper",
        "filter-value",
        "filter-enabled",
    ]:
        st.session_state.pop(f"{prefix}-{filter_id}", None)


def filter_controls(columns: list[str]) -> list[FilterSpec]:
    filters: list[FilterSpec] = []
    initialize_filter_list()
    filter_ids = list(st.session_state["filter-ids"])

    compact_add_header(
        "Filters",
        button_key="add-filter",
        help_text="Add filter",
        on_click=add_filter,
        args=(columns,),
        disabled=len(filter_ids) >= MAX_FILTERS or not columns,
    )
    st.checkbox(
        "Show filters on plot",
        value=False,
        key="show-filter-legend",
    )

    for filter_id in filter_ids:
        with st.container(border=True):
            enabled_key = f"filter-enabled-{filter_id}"
            if enabled_key not in st.session_state:
                st.session_state[enabled_key] = True
            enabled = st.checkbox("Enable", key=enabled_key)
            column_col, mode_col, first_value_col, second_value_col, delete_col = st.columns(
                [0.30, 0.24, 0.17, 0.17, 0.12],
                gap="small",
                vertical_alignment="bottom",
            )
            with column_col:
                column = stable_selectbox(
                    "Column",
                    columns,
                    key=f"filter-column-{filter_id}",
                )
            with mode_col:
                mode = stable_selectbox(
                    "Mode",
                    ["Interval", "Equals", "Not equals"],
                    key=f"filter-mode-{filter_id}",
                )
            if mode == "Interval":
                with first_value_col:
                    lower_text = st.text_input(
                        "a",
                        value="-inf",
                        key=f"filter-lower-{filter_id}",
                    )
                with second_value_col:
                    upper_text = st.text_input(
                        "b",
                        value="inf",
                        key=f"filter-upper-{filter_id}",
                    )
                if enabled:
                    try:
                        filters.append(
                            FilterSpec(
                                column=column,
                                mode=mode,
                                lower=parse_bound(lower_text, -np.inf),
                                upper=parse_bound(upper_text, np.inf),
                            )
                        )
                    except ValueError:
                        st.warning("Interval endpoints must be numbers, blank, inf, or -inf.")
            else:
                with first_value_col:
                    value = st.text_input("Value", key=f"filter-value-{filter_id}")
                with second_value_col:
                    st.empty()
                if enabled:
                    filters.append(FilterSpec(column=column, mode=mode, value=value))
            with delete_col:
                st.button(
                    "Delete",
                    key=f"delete-filter-{filter_id}",
                    on_click=delete_filter,
                    args=(filter_id,),
                    icon=":material/delete:",
                    help="Delete filter",
                    width="content",
                )

    save_persistent_state()
    return filters


def append_to_formula(formula_key: str, insertion: str) -> None:
    st.session_state[formula_key] = st.session_state.get(formula_key, "") + insertion


def begin_edit_formula(
    edit_key: str,
    name_key: str,
    formula_key: str,
    scope_key: str,
    applied_key: str,
) -> None:
    applied = st.session_state.get(applied_key)
    if applied:
        st.session_state[name_key] = applied["name"]
        st.session_state[formula_key] = applied["expression"]
        st.session_state[scope_key] = applied.get("scope", "Global")
    st.session_state[edit_key] = True


def initialize_derived_list() -> None:
    hydrate_persistent_state()
    if "derived-ids" not in st.session_state:
        st.session_state["derived-ids"] = []
    if "derived-next-id" not in st.session_state:
        st.session_state["derived-next-id"] = 0

    migrated_ids = sorted(
        {
            int(match.group(1))
            for key in st.session_state
            if (
                match := re.fullmatch(
                    r"derived-(?:name|formula|scope|visible|applied|edit)-(\d+)",
                    str(key),
                )
            )
        }
    )
    if migrated_ids and not st.session_state["derived-ids"]:
        st.session_state["derived-ids"] = migrated_ids
        st.session_state["derived-next-id"] = max(migrated_ids) + 1


def add_derived_quantity() -> None:
    initialize_derived_list()
    derived_ids = st.session_state["derived-ids"]
    if len(derived_ids) >= MAX_DERIVED_COLUMNS:
        return

    derived_id = st.session_state["derived-next-id"]
    st.session_state["derived-next-id"] = derived_id + 1
    derived_ids.insert(0, derived_id)
    st.session_state[f"derived-name-{derived_id}"] = f"derived_{len(derived_ids)}"
    st.session_state[f"derived-formula-{derived_id}"] = ""
    st.session_state[f"derived-scope-{derived_id}"] = "Global"
    st.session_state[f"derived-visible-{derived_id}"] = True
    st.session_state[f"derived-edit-{derived_id}"] = True
    st.session_state[f"derived-error-{derived_id}"] = ""


def delete_derived_quantity(derived_id: int) -> None:
    initialize_derived_list()
    st.session_state["derived-ids"] = [
        existing_id for existing_id in st.session_state["derived-ids"] if existing_id != derived_id
    ]
    key_patterns = [
        "derived-name",
        "derived-formula",
        "derived-scope",
        "derived-visible",
        "derived-applied",
        "derived-edit",
        "derived-error",
        "derived-name-display",
        "derived-formula-display",
    ]
    for prefix in key_patterns:
        st.session_state.pop(f"{prefix}-{derived_id}", None)


def toggle_derived_visibility(derived_id: int) -> None:
    key = f"derived-visible-{derived_id}"
    st.session_state[key] = not bool(st.session_state.get(key, True))


def apply_derived_formula(
    applied_key: str,
    edit_key: str,
    name_key: str,
    formula_key: str,
    scope_key: str,
    error_key: str,
) -> None:
    name = st.session_state.get(name_key, "").strip()
    expression = st.session_state.get(formula_key, "").strip()
    scope = st.session_state.get(scope_key, "Global")
    if scope not in DERIVED_SCOPES:
        scope = "Global"
    if not name or not expression:
        st.session_state[error_key] = "Add both a derived-column name and a formula before applying."
        return

    st.session_state[applied_key] = {"name": name, "expression": expression, "scope": scope}
    st.session_state[edit_key] = False
    st.session_state[error_key] = ""


def insert_buttons(items: list[tuple[str, str]], formula_key: str, key_prefix: str) -> None:
    if not items:
        return
    button_columns = st.columns(3)
    for index, (label, insertion) in enumerate(items):
        with button_columns[index % 3]:
            st.button(
                label,
                key=f"{key_prefix}-{index}",
                on_click=append_to_formula,
                args=(formula_key, insertion),
                width="stretch",
            )


def derived_column_controls(data: pd.DataFrame) -> list[DerivedSpec]:
    specs: list[DerivedSpec] = []
    number_columns = numeric_columns(data)
    initialize_derived_list()
    derived_ids = list(st.session_state["derived-ids"])

    compact_add_header(
        "Derived Columns",
        button_key="add-derived-column",
        help_text="Add derived quantity",
        on_click=add_derived_quantity,
        disabled=len(derived_ids) >= MAX_DERIVED_COLUMNS,
    )
    derived_sort = stable_selectbox("Sort by", DERIVED_SORT_OPTIONS, key="derived-sort-by")

    if derived_sort == "Name A-Z":
        derived_ids.sort(
            key=lambda derived_id: (
                str(
                    (
                        st.session_state.get(f"derived-applied-{derived_id}") or {}
                    ).get("name", st.session_state.get(f"derived-name-{derived_id}", ""))
                ).casefold(),
                derived_id,
            )
        )

    if derived_ids and not number_columns:
        st.warning("Derived formulas need at least one numeric column.")
        return specs

    for derived_id in derived_ids:
        name_key = f"derived-name-{derived_id}"
        formula_key = f"derived-formula-{derived_id}"
        scope_key = f"derived-scope-{derived_id}"
        applied_key = f"derived-applied-{derived_id}"
        edit_key = f"derived-edit-{derived_id}"
        error_key = f"derived-error-{derived_id}"
        if name_key not in st.session_state:
            st.session_state[name_key] = "derived"
        if formula_key not in st.session_state:
            st.session_state[formula_key] = ""
        if st.session_state.get(scope_key) not in DERIVED_SCOPES:
            st.session_state[scope_key] = "Global"
        visible_key = f"derived-visible-{derived_id}"
        if visible_key not in st.session_state:
            st.session_state[visible_key] = True
        if edit_key not in st.session_state:
            st.session_state[edit_key] = False

        applied = st.session_state.get(applied_key)
        is_editing = st.session_state[edit_key] or not applied

        visible = bool(st.session_state.get(visible_key, True))
        card_key = f"derived-card-{derived_id}" if visible else f"derived-card-hidden-{derived_id}"
        with st.container(border=True, key=card_key):
            if not is_editing:
                equation_col, visible_col, edit_col, delete_col = st.columns(
                    [0.68, 0.10, 0.11, 0.11],
                    gap="small",
                    vertical_alignment="center",
                )
                equation_col.markdown(
                    "<div class='derived-card-summary'>"
                    f"<div class='derived-card-name'>{html.escape(applied['name'])}</div>"
                    f"<div class='derived-card-equation'>{html.escape(applied.get('scope', 'Global'))}: "
                    f"{html.escape(applied['expression'])}</div>"
                    "</div>",
                    unsafe_allow_html=True,
                )
                visible_col.button(
                    "Visibility",
                    key=f"toggle-derived-visible-{derived_id}",
                    on_click=toggle_derived_visibility,
                    args=(derived_id,),
                    icon=":material/visibility:" if visible else ":material/visibility_off:",
                    help="Hide from axes lists" if visible else "Show in axes lists",
                    width="content",
                )
                edit_col.button(
                    "Edit",
                    key=f"edit-derived-{derived_id}",
                    on_click=begin_edit_formula,
                    args=(edit_key, name_key, formula_key, scope_key, applied_key),
                    icon=":material/edit:",
                    help="Edit formula",
                    width="content",
                )
                delete_col.button(
                    "Delete",
                    key=f"delete-derived-{derived_id}",
                    on_click=delete_derived_quantity,
                    args=(derived_id,),
                    icon=":material/delete:",
                    help="Delete derived quantity",
                    width="content",
                )
                specs.append(
                    DerivedSpec(
                        name=applied["name"],
                        expression=applied["expression"],
                        scope=applied.get("scope", "Global"),
                    )
                )
                continue

            name_col, scope_col, formula_col, apply_col = st.columns(
                [0.24, 0.22, 0.42, 0.12],
                gap="small",
                vertical_alignment="center",
            )
            with name_col:
                st.text_input("Name", key=name_key)
            with scope_col:
                stable_selectbox("Scope", DERIVED_SCOPES, key=scope_key)
            with formula_col:
                st.text_area(
                    "Formula",
                    key=formula_key,
                    placeholder="Example: sqrt(`x`**2 + `y`**2)",
                    height=68,
                )
            with apply_col:
                st.button(
                    "Apply",
                    key=f"apply-derived-{derived_id}",
                    on_click=apply_derived_formula,
                    args=(applied_key, edit_key, name_key, formula_key, scope_key, error_key),
                    icon=":material/check:",
                    help="Apply formula",
                    width="content",
                )

            unsupported_columns = [column for column in number_columns if "`" in str(column)]
            column_items = [
                (str(column), column_token(str(column)))
                for column in number_columns
                if "`" not in str(column)
            ]
            arithmetic_items = [
                ("+", "+"),
                ("-", "-"),
                ("/", "/"),
                ("x", "*"),
                ("**", "**"),
                ("(", "("),
                (")", ")"),
            ]

            with st.expander("Insert Into Formula"):
                st.caption("Arithmetic")
                insert_buttons(arithmetic_items, formula_key, f"insert-arithmetic-{derived_id}")

                st.caption("Columns")
                insert_buttons(column_items, formula_key, f"insert-column-{derived_id}")
                if unsupported_columns:
                    st.warning(
                        "Columns containing backticks cannot be used in formulas: "
                        + ", ".join(str(column) for column in unsupported_columns)
                    )

                st.caption("Functions")
                function_items = [
                    ("sqrt()", "sqrt()"),
                    ("log10()", "log10()"),
                    ("exp()", "exp()"),
                    ("unique()", "unique()"),
                    ("lookup_unique()", "lookup_unique(, , )"),
                    ("row_value()", "row_value(, 1)"),
                    ("first_value()", "first_value()"),
                    ("abs()", "abs()"),
                    ("sin()", "sin()"),
                    ("cos()", "cos()"),
                    ("where()", "where(, , )"),
                    ("minimum()", "minimum(, )"),
                    ("maximum()", "maximum(, )"),
                ]
                insert_buttons(function_items, formula_key, f"insert-function-{derived_id}")

            if st.session_state.get(error_key):
                st.warning(st.session_state[error_key])

            applied = st.session_state.get(applied_key)
            if applied:
                st.caption(
                    f"Currently applied: `{applied['name']}` = {applied['expression']} "
                    f"({applied.get('scope', 'Global')})"
                )
                specs.append(
                    DerivedSpec(
                        name=applied["name"],
                        expression=applied["expression"],
                        scope=applied.get("scope", "Global"),
                    )
                )
            else:
                st.caption("Draft formula has not been applied yet.")

    save_persistent_state()
    return specs


def hidden_derived_column_names() -> set[str]:
    initialize_derived_list()
    hidden = set()
    for derived_id in st.session_state["derived-ids"]:
        if bool(st.session_state.get(f"derived-visible-{derived_id}", True)):
            continue
        applied = st.session_state.get(f"derived-applied-{derived_id}")
        if applied:
            hidden.add(str(applied["name"]))
    return hidden


def collect_applied_derived_specs() -> list[DerivedSpec]:
    initialize_derived_list()
    specs: list[DerivedSpec] = []
    for derived_id in st.session_state["derived-ids"]:
        applied = st.session_state.get(f"derived-applied-{derived_id}")
        if applied:
            specs.append(
                DerivedSpec(
                    name=applied["name"],
                    expression=applied["expression"],
                    scope=applied.get("scope", "Global"),
                )
            )
    return specs

def stable_selectbox(
    label: str,
    options: list[Any],
    key: str,
    default_index: int = 0,
    preserve_current: bool = False,
    persist: bool = False,
    **kwargs: Any,
) -> Any:
    if not options:
        raise ValueError(f"No options available for `{label}`.")

    value_key = f"_stable-selectbox-value-{key}"

    # Safely fetch current value Streamlit knows about, or the persisted one
    current = st.session_state.get(key)
    if current is None and persist:
        current = st.session_state.get(value_key)

    render_options = list(options)
    if preserve_current and current is not None and current not in render_options:
        render_options.insert(0, current)

    # Determine the correct index to select
    if current in render_options:
        target_index = render_options.index(current)
    else:
        target_index = min(default_index, len(render_options) - 1)

    # Instead of deleting the key, update the session state to a valid option
    # so st.selectbox doesn't crash or drop its state.
    if key in st.session_state and st.session_state[key] not in render_options:
        st.session_state[key] = render_options[target_index]

    selected = st.selectbox(label, render_options, index=target_index, key=key, **kwargs)

    if persist:
        st.session_state[value_key] = selected

    return selected

def stable_color_picker(label: str, key: str, default: str) -> str:
    value_key = f"_stable-color-value-{key}"
    default = normalized_hex_color(default, "#ffffff")

    # Retrieve the successfully persisted value, or fallback to default
    committed = normalized_hex_color(
        st.session_state.get(value_key, default),
        default,
    )

    # Native Streamlit handles defaults natively using the `value` argument
    selected = normalized_hex_color(
        st.color_picker(label, value=committed, key=key),
        committed
    )

    # Persist the change if the user selects a new color
    if selected != committed:
        st.session_state[value_key] = selected

    return selected

def color_widget_key(prefix: str, column: str, value: Any) -> str:
    digest = hashlib.sha1(f"{column}:{value}".encode("utf-8")).hexdigest()[:12]
    return f"{prefix}-{digest}"


def unique_plot_values(data: pd.DataFrame, column: str, max_values: int = 20) -> tuple[list[Any], int]:
    values = sorted(data[column].dropna().drop_duplicates().tolist(), key=lambda value: str(value))
    return values[:max_values], len(values)


def unique_plot_value_labels(data: pd.DataFrame, column: str, max_values: int = 20) -> tuple[list[str], int]:
    values, total_values = unique_plot_values(data, column, max_values=max_values)
    return [str(value) for value in values], total_values


def trace_color_controls(prefix: str, data: pd.DataFrame, column: str) -> dict[str, str]:
    trace_colors: dict[str, str] = {}
    values, total_values = unique_plot_value_labels(data, column)
    if total_values > len(values):
        st.caption(f"Showing color controls for {len(values)} of {total_values} values.")

    for row_start in range(0, len(values), 4):
        row_values = values[row_start : row_start + 4]
        color_columns = st.columns(len(row_values), gap="small")
        for offset, value in enumerate(row_values):
            with color_columns[offset]:
                key = color_widget_key(prefix, column, value)
                trace_colors[str(value)] = stable_color_picker(
                    str(value),
                    key=key,
                    default=default_trace_color(row_start + offset),
                )
    return trace_colors


def trace_color_controls_in_columns(
    prefix: str,
    column: str,
    values: list[str],
    columns: list[Any],
) -> dict[str, str]:
    trace_colors: dict[str, str] = {}
    for index, value in enumerate(values):
        key = color_widget_key(prefix, column, value)
        with columns[index]:
            trace_colors[value] = stable_color_picker(
                value,
                key=key,
                default=default_trace_color(index),
            )
    return trace_colors


def initialize_reference_lines() -> None:
    if "reference-line-ids" not in st.session_state:
        st.session_state["reference-line-ids"] = []
    if "reference-line-next-id" not in st.session_state:
        st.session_state["reference-line-next-id"] = 0


def add_reference_line_control() -> None:
    initialize_reference_lines()
    line_id = st.session_state["reference-line-next-id"]
    st.session_state["reference-line-next-id"] = line_id + 1
    st.session_state["reference-line-ids"].append(line_id)
    st.session_state[f"reference-line-axis-{line_id}"] = "X"
    st.session_state[f"reference-line-value-{line_id}"] = "1"
    st.session_state[f"reference-line-label-{line_id}"] = ""
    st.session_state[f"reference-line-color-{line_id}"] = "#d92d20"
    st.session_state[f"reference-line-dash-{line_id}"] = "dash"


def delete_reference_line_control(line_id: int) -> None:
    initialize_reference_lines()
    st.session_state["reference-line-ids"] = [
        existing_id for existing_id in st.session_state["reference-line-ids"] if existing_id != line_id
    ]
    for prefix in (
        "reference-line-axis",
        "reference-line-value",
        "reference-line-label",
        "reference-line-color",
        "reference-line-dash",
    ):
        st.session_state.pop(f"{prefix}-{line_id}", None)


def reference_line_controls(plot_type: str) -> list[ReferenceLine]:
    initialize_reference_lines()
    reference_lines: list[ReferenceLine] = []

    compact_add_header(
        "Reference Lines",
        button_key="add-reference-line",
        help_text="Add reference line",
        on_click=add_reference_line_control,
    )

    if plot_type == "Scatter":
        axis_options = ["X", "Y"]
    else:
        axis_options = ["X", "Y"]

    for line_id in list(st.session_state["reference-line-ids"]):
        axis_key = f"reference-line-axis-{line_id}"
        value_key = f"reference-line-value-{line_id}"
        label_key = f"reference-line-label-{line_id}"
        color_key = f"reference-line-color-{line_id}"
        dash_key = f"reference-line-dash-{line_id}"

        with st.container(border=True):
            axis_col, value_col, label_col, color_col, dash_col, delete_col = st.columns(
                [0.13, 0.20, 0.27, 0.16, 0.16, 0.08],
                gap="small",
                vertical_alignment="bottom",
            )
            with axis_col:
                axis = stable_selectbox("Axis", axis_options, key=axis_key)
            with value_col:
                value_text = st.text_input("Value", key=value_key)
            with label_col:
                label = st.text_input("Label", key=label_key)
            with color_col:
                color = stable_color_picker("Color", key=color_key, default="#d92d20")
            with dash_col:
                dash = stable_selectbox("Line", LINE_DASHES, key=dash_key, default_index=1)
            with delete_col:
                st.button(
                    "Delete",
                    key=f"delete-reference-line-{line_id}",
                    on_click=delete_reference_line_control,
                    args=(line_id,),
                    icon=":material/delete:",
                    help="Delete reference line",
                    width="content",
                )

            try:
                value = float(str(value_text).strip())
            except ValueError:
                st.warning("Line value must be a number.")
                continue

            reference_lines.append(
                ReferenceLine(
                    axis=axis,
                    value=value,
                    label=label,
                    color=color,
                    dash=dash,
                )
            )

    return reference_lines


def point_label_controls(plot_type: str, data: pd.DataFrame) -> tuple[str | None, bool]:
    if plot_type != "Scatter":
        return None, False

    label_options = ["None", *plottable_columns(data)]
    mode_col, column_col = st.columns([0.42, 0.58], gap="small")
    with mode_col:
        label_mode = stable_selectbox("Point labels", POINT_LABEL_MODES, key="point-label-mode")
    if label_mode == "None":
        return None, False

    with column_col:
        label_column = stable_selectbox("Label column", label_options[1:], key="point-label-column")
    return label_column, label_mode == "Show on plot"


def highlight_point_controls(plot_type: str, data: pd.DataFrame) -> bool:
    if plot_type != "Scatter" or "plot_highlight_label" not in data.columns:
        return True

    st.session_state.setdefault("show-highlighted-points", True)
    return st.checkbox("Show highlighted points", key="show-highlighted-points")


def plot_style_controls(
    plot_type: str,
    data: pd.DataFrame,
    color_column: str | None,
    split_column: str | None,
) -> PlotStyle:
    with st.sidebar:
        st.header("Style")
        reference_lines: list[ReferenceLine] = []
        point_label_column, show_point_labels = point_label_controls(plot_type, data)
        show_highlighted_points = highlight_point_controls(plot_type, data)

        if plot_type == "Scatter":
            if color_column is None:
                background_col, point_col = st.columns(2, gap="small", vertical_alignment="bottom")
                with background_col:
                    background_color = stable_color_picker(
                        "Plot background",
                        key="plot-background-color",
                        default="#ffffff",
                    )
                with point_col:
                    single_color = stable_color_picker(
                        "Point color",
                        key="scatter-single-color",
                        default=default_trace_color(0),
                    )
                reference_lines = reference_line_controls(plot_type)
                return PlotStyle(
                    single_color=single_color,
                    background_color=background_color,
                    reference_lines=reference_lines,
                    point_label_column=point_label_column,
                    show_point_labels=show_point_labels,
                    show_highlighted_points=show_highlighted_points,
                )

            if pd.api.types.is_numeric_dtype(data[color_column]):
                background_col, palette_col = st.columns(
                    [0.34, 0.66],
                    gap="small",
                    vertical_alignment="bottom",
                )
                with background_col:
                    background_color = stable_color_picker(
                        "Plot background",
                        key="plot-background-color",
                        default="#ffffff",
                    )
                with palette_col:
                    color_scale = stable_selectbox(
                        "Color palette",
                        COLOR_SCALES,
                        key="scatter-color-scale",
                        persist=True,
                    )
                reference_lines = reference_line_controls(plot_type)
                return PlotStyle(
                    color_scale=color_scale,
                    background_color=background_color,
                    reference_lines=reference_lines,
                    point_label_column=point_label_column,
                    show_point_labels=show_point_labels,
                    show_highlighted_points=show_highlighted_points,
                )

            color_values, total_values = unique_plot_value_labels(data, color_column)
            if total_values > len(color_values):
                st.caption(f"Showing color controls for {len(color_values)} of {total_values} values.")
            if len(color_values) <= 3:
                style_cols = st.columns(
                    [0.28, *([0.18] * len(color_values)), 1.0],
                    gap="small",
                    vertical_alignment="bottom",
                )
                with style_cols[0]:
                    background_color = stable_color_picker(
                        "Plot background",
                        key="plot-background-color",
                        default="#ffffff",
                    )
                trace_colors = trace_color_controls_in_columns(
                    "scatter-category-color",
                    color_column,
                    color_values,
                    style_cols[1 : 1 + len(color_values)],
                )
            else:
                background_color = stable_color_picker(
                    "Plot background",
                    key="plot-background-color",
                    default="#ffffff",
                )
                trace_colors = trace_color_controls("scatter-category-color", data, color_column)
            reference_lines = reference_line_controls(plot_type)
            return PlotStyle(
                trace_colors=trace_colors,
                background_color=background_color,
                reference_lines=reference_lines,
                point_label_column=point_label_column,
                show_point_labels=show_point_labels,
                show_highlighted_points=show_highlighted_points,
            )

        if split_column and split_column in data.columns:
            split_values, _ = unique_plot_value_labels(data, split_column)
            trace_order = None
            if split_values and len(split_values) <= 3:
                style_cols = st.columns(
                    [0.24, *([0.16] * len(split_values)), 0.28],
                    gap="small",
                    vertical_alignment="bottom",
                )
                with style_cols[0]:
                    background_color = stable_color_picker(
                        "Plot background",
                        key="plot-background-color",
                        default="#ffffff",
                    )
                trace_colors = trace_color_controls_in_columns(
                    "hist-split-color",
                    split_column,
                    split_values,
                    style_cols[1 : 1 + len(split_values)],
                )
                with style_cols[-1]:
                    top_value = stable_selectbox(
                        "Draw on top",
                        split_values,
                        key=f"hist-top-layer-{color_widget_key('split', split_column, tuple(split_values))}",
                        default_index=max(len(split_values) - 1, 0),
                        persist=True,
                    )
                trace_order = [value for value in split_values if value != top_value]
                trace_order.append(top_value)
            else:
                background_top_cols = st.columns(2, gap="small", vertical_alignment="bottom")
                with background_top_cols[0]:
                    background_color = stable_color_picker(
                        "Plot background",
                        key="plot-background-color",
                        default="#ffffff",
                    )
                if split_values:
                    with background_top_cols[1]:
                        top_value = stable_selectbox(
                            "Draw on top",
                            split_values,
                            key=f"hist-top-layer-{color_widget_key('split', split_column, tuple(split_values))}",
                            default_index=max(len(split_values) - 1, 0),
                            persist=True,
                        )
                    trace_order = [value for value in split_values if value != top_value]
                    trace_order.append(top_value)
                trace_colors = trace_color_controls("hist-split-color", data, split_column)
            reference_lines = reference_line_controls(plot_type)
            return PlotStyle(
                background_color=background_color,
                trace_colors=trace_colors,
                trace_order=trace_order,
                reference_lines=reference_lines,
            )

        background_col, histogram_col = st.columns(2, gap="small", vertical_alignment="bottom")
        with background_col:
            background_color = stable_color_picker(
                "Plot background",
                key="plot-background-color",
                default="#ffffff",
            )
        with histogram_col:
            single_color = stable_color_picker(
                "Histogram color",
                key="hist-single-color",
                default=default_trace_color(0),
            )
        reference_lines = reference_line_controls(plot_type)
        return PlotStyle(
            single_color=single_color,
            background_color=background_color,
            reference_lines=reference_lines,
        )


def main() -> None:
    st.title("Plot Builder")

    workspace_paths = data_workspace_paths(str(DATA_DIR))
    if not workspace_paths:
        st.info("Add CSV files to a folder inside `data/`, then refresh this page.")
        return

    with st.sidebar:
        st.header("Data")
        selected_workspace_path = stable_selectbox(
            "Workspace",
            workspace_paths,
            key="workspace-folder",
            format_func=data_workspace_label,
            persist=True,
        )
        st.caption(f"Saved settings: `{current_state_path().as_posix()}`")
        catalogue_mode = st.segmented_control(
            "Catalogues",
            ["One", "All", "Separate"],
            default="One",
            key="catalogue-mode",
            help="`All` combines every catalogue into one plot. `Separate` shows one plot per catalogue.",
        )

    paths = csv_paths(str(selected_workspace_path))
    if not paths:
        st.info(f"Add CSV files to `{selected_workspace_path.as_posix()}`, then refresh this page.")
        return

    with st.sidebar:
        selected_path = stable_selectbox(
            "CSV file",
            paths,
            key="selected-csv-file",
            format_func=lambda path: path.name,
            disabled=catalogue_mode != "One",
        )

    selected_paths = paths if catalogue_mode in {"All", "Separate"} else [selected_path]
    data = load_catalogues(selected_paths)

    derived_error: ValueError | None = None
    derived_specs = collect_applied_derived_specs()
    try:
        data = evaluate_derived_columns(data, derived_specs)
    except ValueError as exc:
        derived_error = exc

    columns = plottable_columns(data)
    numbers = numeric_columns(data)
    hidden_columns = hidden_derived_column_names()
    plot_columns = [column for column in columns if column not in hidden_columns]
    plot_numbers = [column for column in numbers if column not in hidden_columns]

    if not columns:
        st.warning("The selected CSV file(s) do not contain any plottable columns.")
        return
    if len(plot_numbers) < 1:
        st.warning("At least one numeric column is required for plotting.")
        return

    with st.sidebar:
        st.header("Plot")
        plot_type = st.segmented_control(
            "Type",
            ["Scatter", "Histogram"],
            default="Scatter",
            key="plot-type",
        )

        x_column = None
        y_column = None
        z_column = None
        color_column = None
        hist_column = None
        bins = None
        split_column = None

        if plot_type == "Scatter":
            x_column = stable_selectbox(
                "X",
                plot_numbers,
                key="scatter-x-column",
                default_index=0,
                preserve_current=True,
                persist=True,
            )
            y_column = stable_selectbox(
                "Y",
                plot_numbers,
                key="scatter-y-column",
                default_index=min(1, len(plot_numbers) - 1),
                preserve_current=True,
                persist=True,
            )
            z_choice = stable_selectbox(
                "Z",
                ["None", *plot_numbers],
                key="scatter-z-column",
                preserve_current=True,
                persist=True,
            )
            z_column = None if z_choice == "None" else z_choice
            color_choice = stable_selectbox(
                "Color",
                ["None", *plot_columns],
                key="scatter-color-column",
                preserve_current=True,
                persist=True,
            )
            color_column = None if color_choice == "None" else color_choice
        else:
            hist_column = stable_selectbox(
                "Column",
                plot_numbers,
                key="hist-column",
                preserve_current=True,
                persist=True,
            )
            bins = st.slider("Bins", min_value=5, max_value=150, value=40, step=5, key="hist-bins")
            split_choice = stable_selectbox(
                "Split by",
                ["None", SOURCE_COLUMN, *plot_columns],
                key="hist-split-column",
                preserve_current=True,
                persist=True,
            )
            split_column = None if split_choice == "None" else split_choice

        filters = filter_controls(columns)
        show_filter_legend = bool(st.session_state.get("show-filter-legend", False))
        derived_column_controls(data)

    if derived_error:
        st.warning(str(derived_error))

    try:
        filtered = apply_filters(data, filters)
    except ValueError as exc:
        st.warning(str(exc))
        return

    metric_columns = st.columns(4)
    metric_columns[0].metric("Rows", f"{len(filtered):,} / {len(data):,}")
    metric_columns[1].metric("Files", len(selected_paths))
    metric_columns[2].metric("Columns", len(data.columns))
    metric_columns[3].metric("Plot", plot_type)

    if filtered.empty:
        st.info("No rows match the current filters.")
        return

    style = plot_style_controls(
        plot_type,
        filtered,
        color_column if plot_type == "Scatter" else None,
        split_column if plot_type == "Histogram" else None,
    )

    st.divider()

    figures: list[tuple[str, go.Figure]] = []
    if catalogue_mode == "Separate":
        source_names = [path.name for path in selected_paths]
        filtered_by_source = {
            str(source): group
            for source, group in filtered.groupby(SOURCE_COLUMN, sort=False)
        }

        with st.container(height=780):
            for source_name in source_names:
                source_data = filtered_by_source.get(source_name, filtered.iloc[0:0])
                st.subheader(source_name)
                if source_data.empty:
                    st.info("No rows match the current filters.")
                    continue

                try:
                    fig = build_plot_figure(
                        source_data,
                        plot_type,
                        style,
                        x_column=x_column,
                        y_column=y_column,
                        z_column=z_column,
                        color_column=color_column,
                        hist_column=hist_column,
                        bins=bins,
                        split_column=split_column,
                    )
                    if show_filter_legend:
                        add_filter_legend(fig, filters, style.background_color)
                except ValueError as exc:
                    st.warning(str(exc))
                    continue

                figures.append((source_name, fig))
                st.plotly_chart(fig, use_container_width=True, key=f"plot-{source_name}")
                save_plot_controls(
                    fig,
                    Path(source_name).stem,
                    f"plot-{hashlib.sha1(source_name.encode('utf-8')).hexdigest()[:12]}",
                )
    else:
        try:
            fig = build_plot_figure(
                filtered,
                plot_type,
                style,
                x_column=x_column,
                y_column=y_column,
                z_column=z_column,
                color_column=color_column,
                hist_column=hist_column,
                bins=bins,
                split_column=split_column,
            )
            if show_filter_legend:
                add_filter_legend(fig, filters, style.background_color)
        except ValueError as exc:
            st.warning(str(exc))
            return

        figures.append(("Plot", fig))
        st.plotly_chart(fig, use_container_width=True)
        save_plot_controls(fig, "plot", "plot-main")

    st.download_button(
        "Download Filtered CSV",
        data=filtered.to_csv(index=False),
        file_name="filtered_data.csv",
        mime="text/csv",
    )
    if figures:
        st.download_button(
            "Download Plot HTML" if len(figures) == 1 else "Download Plots HTML",
            data=plot_html_document(figures),
            file_name="plot.html",
            mime="text/html",
        )
    if st.button("View Saved Plots", icon=":material/folder_open:"):
        saved_plots_dialog()

    with st.expander("Data Preview"):
        st.dataframe(filtered, width="stretch", hide_index=True)


if __name__ == "__main__":
    main()
