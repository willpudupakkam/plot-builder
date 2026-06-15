from __future__ import annotations

from dataclasses import dataclass
import hashlib
import html
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


DATA_DIR = Path("data")
STATE_DIR = Path(".plot_builder_state")
LEGACY_STATE_PATH = Path(".plot_builder_state.json")
DEFAULT_WORKSPACE = "default"
SOURCE_COLUMN = "source_file"
MAX_DERIVED_COLUMNS = 8
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


st.set_page_config(
    page_title="Plot Builder",
    page_icon=":bar_chart:",
    layout="wide",
)

st.markdown(
    """
    <style>
    .st-key-add-derived-column button,
    .st-key-add-filter button {
        min-height: 1.7rem;
        height: 1.7rem;
        padding: 0.1rem 0.45rem;
        line-height: 1;
    }

    .st-key-add-derived-column [data-testid="stButton"],
    .st-key-add-filter [data-testid="stButton"] {
        display: flex;
        align-items: center;
        transform: translateY(0.35rem);
    }

    [class*="st-key-delete-filter-"] button {
        min-height: 2.45rem;
        padding: 0 0.55rem;
        border-color: #f04438 !important;
        background: #b42318 !important;
        color: white !important;
    }

    [class*="st-key-delete-filter-"] button:hover {
        border-color: #ff6b5f !important;
        background: #d92d20 !important;
    }

    [class*="st-key-delete-filter-"] button [data-testid="stMarkdownContainer"] {
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

    /* Keep section titles and their add buttons together, even in a narrow sidebar. */
    .st-key-add-derived-column-header [data-testid="stHorizontalBlock"],
    .st-key-add-filter-header [data-testid="stHorizontalBlock"] {
        display: flex;
        flex-wrap: nowrap;
        align-items: center;
        gap: 0.35rem;
    }

    .st-key-add-derived-column-header [data-testid="stHorizontalBlock"] > [data-testid="stColumn"],
    .st-key-add-filter-header [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
        flex: 0 0 auto !important;
        width: auto !important;
        min-width: 0 !important;
    }

    .compact-add-header-title {
        font-size: 1.25rem;
        font-weight: 700;
        line-height: 1.7rem;
        white-space: nowrap;
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


@dataclass(frozen=True)
class PlotStyle:
    single_color: str = "#2f80b7"
    color_scale: str = "Viridis"
    background_color: str = "#ffffff"
    trace_colors: dict[str, str] | None = None
    trace_order: list[str] | None = None


@st.cache_data(show_spinner=False)
def csv_paths(data_dir: str) -> list[Path]:
    return sorted(Path(data_dir).glob("*.csv"))


@st.cache_data(show_spinner=False)
def read_csv(path: str) -> pd.DataFrame:
    data = pd.read_csv(path)
    data[SOURCE_COLUMN] = Path(path).name
    return data


def load_catalogues(paths: list[Path]) -> pd.DataFrame:
    frames = [read_csv(str(path)) for path in paths]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def workspace_slug(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip()).strip("._-").lower()
    return slug or DEFAULT_WORKSPACE


def current_workspace_name() -> str:
    workspace = str(st.session_state.get("workspace-name", DEFAULT_WORKSPACE)).strip()
    return workspace or DEFAULT_WORKSPACE


def current_state_path() -> Path:
    return STATE_DIR / f"{workspace_slug(current_workspace_name())}.json"


def clear_persistent_session_keys() -> None:
    exact_keys = {
        "derived-ids",
        "derived-next-id",
        "filter-ids",
        "filter-next-id",
    }
    prefixes = (
        "derived-name-",
        "derived-formula-",
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
            st.session_state[f"derived-edit-{derived_id}"] = bool(item.get("edit", False))
            st.session_state[f"derived-error-{derived_id}"] = ""
            applied = item.get("applied")
            if isinstance(applied, dict):
                st.session_state[f"derived-applied-{derived_id}"] = {
                    "name": str(applied.get("name", "")),
                    "expression": str(applied.get("expression", "")),
                }
        st.session_state["derived-ids"] = derived_ids
        st.session_state["derived-next-id"] = max(derived_ids, default=-1) + 1

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
            }
        )

    state = {
        "derived": derived_items,
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


def evaluate_derived_columns(data: pd.DataFrame, specs: list[DerivedSpec]) -> pd.DataFrame:
    if not specs:
        return data

    derived = data.copy()
    used_output_names = set(data.columns)

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

        try:
            prepared_expression, local_dict = prepare_eval_expression(derived, expression)
            result = pd.eval(prepared_expression, engine="python", local_dict=local_dict)
        except Exception as exc:
            msg = (
                f"Could not evaluate `{name}`. Use the column buttons in the formula so "
                "column names are inserted with pandas backticks. "
                f"Details: {exc}"
            )
            raise ValueError(msg) from exc

        if np.isscalar(result):
            series = pd.Series(result, index=derived.index)
        else:
            series = pd.Series(result, index=derived.index)
        derived[name] = pd.to_numeric(series, errors="coerce")
        used_output_names.add(name)

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


def default_trace_color(index: int) -> str:
    return DEFAULT_TRACE_COLORS[index % len(DEFAULT_TRACE_COLORS)]


def marker_style(data: pd.DataFrame, color_column: str | None, style: PlotStyle) -> dict[str, Any]:
    marker: dict[str, Any] = {"size": 7, "opacity": 0.72}
    if color_column is None:
        marker["color"] = style.single_color
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


def readable_axis_colors(background_color: str) -> dict[str, str]:
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
    hover = f"{x_column}: %{{x:.4g}}<br>{y_column}: %{{y:.4g}}"
    if custom_columns:
        hover = f"{SOURCE_COLUMN}: %{{customdata[0]}}<br>" + hover
    if color_column:
        hover += f"<br>{color_column}: %{{marker.color}}"
    hover += "<extra></extra>"

    fig = go.Figure()
    categorical_color = (
        color_column is not None and not pd.api.types.is_numeric_dtype(plot_data[color_column])
    )
    if categorical_color:
        trace_colors = style.trace_colors or {}
        for trace_index, (value, group) in enumerate(plot_data.groupby(color_column, sort=True)):
            color = trace_colors.get(str(value), default_trace_color(trace_index))
            marker = {"size": 5 if z_column else 7, "opacity": 0.72, "color": color}
            customdata = group[custom_columns].astype(str).to_numpy() if custom_columns else None
            group_hover = hover.replace(f"{color_column}: %{{marker.color}}", f"{color_column}: {value}")
            if z_column:
                fig.add_trace(
                    go.Scatter3d(
                        x=group[x_column],
                        y=group[y_column],
                        z=group[z_column],
                        mode="markers",
                        marker=marker,
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
                        mode="markers",
                        marker=marker,
                        customdata=customdata,
                        hovertemplate=group_hover,
                        name=str(value),
                    )
                )
    else:
        marker = marker_style(plot_data, color_column, style)
        if z_column:
            marker["size"] = 5

        customdata = plot_data[custom_columns].astype(str).to_numpy() if custom_columns else None
        if z_column:
            fig.add_trace(
                go.Scatter3d(
                    x=plot_data[x_column],
                    y=plot_data[y_column],
                    z=plot_data[z_column],
                    mode="markers",
                    marker=marker,
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
                    mode="markers",
                    marker=marker,
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

    style_plot(fig, has_z_axis=z_column is not None, background_color=style.background_color)
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
                    marker={"color": trace_colors.get(str(value), default_trace_color(trace_index))},
                    name=str(value),
                )
            )
        fig.update_layout(barmode="overlay")
    else:
        fig.add_trace(go.Histogram(x=plot_data[column], nbinsx=bins, marker={"color": style.single_color}))

    fig.update_layout(xaxis_title=column, yaxis_title="Count")
    style_plot(fig, background_color=style.background_color)
    return fig


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
            [1, 1],
            gap="small",
            vertical_alignment="center",
        )

        title_col.markdown(
            f"<div class='compact-add-header-title'>{html.escape(title)}</div>",
            unsafe_allow_html=True,
        )

        add_col.button(
            "+",
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
            if (match := re.fullmatch(r"filter-(?:column|mode|lower|upper|value)-(\d+)", str(key)))
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


def delete_filter(filter_id: int) -> None:
    initialize_filter_list()
    st.session_state["filter-ids"] = [
        existing_id for existing_id in st.session_state["filter-ids"] if existing_id != filter_id
    ]
    for prefix in ["filter-column", "filter-mode", "filter-lower", "filter-upper", "filter-value"]:
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

    for filter_id in filter_ids:
        with st.container(border=True):
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


def begin_edit_formula(edit_key: str, name_key: str, formula_key: str, applied_key: str) -> None:
    applied = st.session_state.get(applied_key)
    if applied:
        st.session_state[name_key] = applied["name"]
        st.session_state[formula_key] = applied["expression"]
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
            if (match := re.fullmatch(r"derived-(?:name|formula|applied|edit)-(\d+)", str(key)))
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
    derived_ids.append(derived_id)
    st.session_state[f"derived-name-{derived_id}"] = f"derived_{len(derived_ids)}"
    st.session_state[f"derived-formula-{derived_id}"] = ""
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
        "derived-applied",
        "derived-edit",
        "derived-error",
        "derived-name-display",
        "derived-formula-display",
    ]
    for prefix in key_patterns:
        st.session_state.pop(f"{prefix}-{derived_id}", None)


def apply_derived_formula(
    applied_key: str,
    edit_key: str,
    name_key: str,
    formula_key: str,
    error_key: str,
) -> None:
    name = st.session_state.get(name_key, "").strip()
    expression = st.session_state.get(formula_key, "").strip()
    if not name or not expression:
        st.session_state[error_key] = "Add both a derived-column name and a formula before applying."
        return

    st.session_state[applied_key] = {"name": name, "expression": expression}
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

    if derived_ids and not number_columns:
        st.warning("Derived formulas need at least one numeric column.")
        return specs

    for derived_id in derived_ids:
        name_key = f"derived-name-{derived_id}"
        formula_key = f"derived-formula-{derived_id}"
        applied_key = f"derived-applied-{derived_id}"
        edit_key = f"derived-edit-{derived_id}"
        error_key = f"derived-error-{derived_id}"
        if name_key not in st.session_state:
            st.session_state[name_key] = "derived"
        if formula_key not in st.session_state:
            st.session_state[formula_key] = ""
        if edit_key not in st.session_state:
            st.session_state[edit_key] = False

        applied = st.session_state.get(applied_key)
        is_editing = st.session_state[edit_key] or not applied

        with st.container(border=True):
            if not is_editing:
                equation_col, edit_col, delete_col = st.columns(
                    [0.76, 0.12, 0.12],
                    gap="small",
                    vertical_alignment="center",
                )
                equation_col.markdown(
                    "<div class='derived-card-summary'>"
                    f"<div class='derived-card-name'>{html.escape(applied['name'])}</div>"
                    f"<div class='derived-card-equation'>{html.escape(applied['expression'])}</div>"
                    "</div>",
                    unsafe_allow_html=True,
                )
                edit_col.button(
                    "Edit",
                    key=f"edit-derived-{derived_id}",
                    on_click=begin_edit_formula,
                    args=(edit_key, name_key, formula_key, applied_key),
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
                specs.append(DerivedSpec(name=applied["name"], expression=applied["expression"]))
                continue

            name_col, formula_col, apply_col = st.columns(
                [0.30, 0.58, 0.12],
                gap="small",
                vertical_alignment="center",
            )
            with name_col:
                st.text_input("Name", key=name_key)
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
                    args=(applied_key, edit_key, name_key, formula_key, error_key),
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

            with st.expander("Insert Into Formula"):
                st.caption("Columns")
                insert_buttons(column_items, formula_key, f"insert-column-{derived_id}")
                if unsupported_columns:
                    st.warning(
                        "Columns containing backticks cannot be used in formulas: "
                        + ", ".join(str(column) for column in unsupported_columns)
                    )

                st.caption("Arithmetic")
                arithmetic_items = [
                    ("+", "+"),
                    ("-", "-"),
                    ("/", "/"),
                    ("x", "*"),
                    ("**", "**"),
                    ("(", "("),
                    (")", ")"),
                ]
                insert_buttons(arithmetic_items, formula_key, f"insert-arithmetic-{derived_id}")

                st.caption("Functions")
                function_items = [
                    ("sqrt()", "sqrt()"),
                    ("log10()", "log10()"),
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
                st.caption(f"Currently applied: `{applied['name']}` = {applied['expression']}")
                specs.append(DerivedSpec(name=applied["name"], expression=applied["expression"]))
            else:
                st.caption("Draft formula has not been applied yet.")

    save_persistent_state()
    return specs


def collect_applied_derived_specs() -> list[DerivedSpec]:
    initialize_derived_list()
    specs: list[DerivedSpec] = []
    for derived_id in st.session_state["derived-ids"]:
        applied = st.session_state.get(f"derived-applied-{derived_id}")
        if applied:
            specs.append(DerivedSpec(name=applied["name"], expression=applied["expression"]))
    return specs


def stable_selectbox(
    label: str,
    options: list[Any],
    key: str,
    default_index: int = 0,
    **kwargs: Any,
) -> Any:
    if not options:
        raise ValueError(f"No options available for `{label}`.")

    default = options[min(default_index, len(options) - 1)]
    if st.session_state.get(key) not in options:
        st.session_state[key] = default
    selected_index = options.index(st.session_state[key])
    return st.selectbox(label, options, index=selected_index, key=key, **kwargs)


def stable_color_picker(label: str, key: str, default: str) -> str:
    if key not in st.session_state:
        st.session_state[key] = default
    return st.color_picker(label, key=key)


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


def plot_style_controls(
    plot_type: str,
    data: pd.DataFrame,
    color_column: str | None,
    split_column: str | None,
) -> PlotStyle:
    with st.sidebar:
        st.header("Style")

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
                return PlotStyle(single_color=single_color, background_color=background_color)

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
                    )
                return PlotStyle(color_scale=color_scale, background_color=background_color)

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
            return PlotStyle(trace_colors=trace_colors, background_color=background_color)

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
                        )
                    trace_order = [value for value in split_values if value != top_value]
                    trace_order.append(top_value)
                trace_colors = trace_color_controls("hist-split-color", data, split_column)
            return PlotStyle(
                background_color=background_color,
                trace_colors=trace_colors,
                trace_order=trace_order,
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
        return PlotStyle(single_color=single_color, background_color=background_color)


def main() -> None:
    st.title("Plot Builder")

    paths = csv_paths(str(DATA_DIR))
    if not paths:
        st.info("Add CSV files to the `data/` folder, then refresh this page.")
        return

    with st.sidebar:
        st.header("Data")
        st.text_input(
            "Workspace",
            value=DEFAULT_WORKSPACE,
            key="workspace-name",
            help="Filters and derived quantities are saved separately for each workspace name.",
        )
        st.caption(f"Saved settings: `{current_state_path().as_posix()}`")
        catalogue_mode = st.segmented_control(
            "Catalogues",
            ["One", "All"],
            default="One",
            key="catalogue-mode",
        )
        selected_path = stable_selectbox(
            "CSV file",
            paths,
            key="selected-csv-file",
            format_func=lambda path: path.name,
            disabled=catalogue_mode == "All",
        )

    selected_paths = paths if catalogue_mode == "All" else [selected_path]
    data = load_catalogues(selected_paths)

    derived_error: ValueError | None = None
    derived_specs = collect_applied_derived_specs()
    try:
        data = evaluate_derived_columns(data, derived_specs)
    except ValueError as exc:
        derived_error = exc

    columns = plottable_columns(data)
    numbers = numeric_columns(data)

    if not columns:
        st.warning("The selected CSV file(s) do not contain any plottable columns.")
        return
    if len(numbers) < 1:
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

        if plot_type == "Scatter":
            axis_cols = st.columns(4, gap="small")
            with axis_cols[0]:
                x_column = stable_selectbox("X", numbers, key="scatter-x-column", default_index=0)
            with axis_cols[1]:
                y_column = stable_selectbox(
                    "Y",
                    numbers,
                    key="scatter-y-column",
                    default_index=min(1, len(numbers) - 1),
                )
            with axis_cols[2]:
                z_choice = stable_selectbox("Z", ["None", *numbers], key="scatter-z-column")
            z_column = None if z_choice == "None" else z_choice
            with axis_cols[3]:
                color_choice = stable_selectbox(
                    "Color",
                    ["None", *columns],
                    key="scatter-color-column",
                )
            color_column = None if color_choice == "None" else color_choice
        else:
            hist_column = stable_selectbox("Column", numbers, key="hist-column")
            bins = st.slider("Bins", min_value=5, max_value=150, value=40, step=5, key="hist-bins")
            split_choice = stable_selectbox(
                "Split by",
                ["None", SOURCE_COLUMN, *columns],
                key="hist-split-column",
            )
            split_column = None if split_choice == "None" else split_choice

        filters = filter_controls(columns)
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

    try:
        if plot_type == "Scatter":
            fig = scatter_figure(filtered, x_column, y_column, z_column, color_column, style)
        else:
            fig = histogram_figure(filtered, hist_column, bins, split_column, style)
    except ValueError as exc:
        st.warning(str(exc))
        return

    st.plotly_chart(fig, width="stretch")

    st.download_button(
        "Download Filtered CSV",
        data=filtered.to_csv(index=False),
        file_name="filtered_data.csv",
        mime="text/csv",
    )
    st.download_button(
        "Download Plot HTML",
        data=fig.to_html(include_plotlyjs="cdn"),
        file_name="plot.html",
        mime="text/html",
    )

    with st.expander("Data Preview"):
        st.dataframe(filtered, width="stretch", hide_index=True)


if __name__ == "__main__":
    main()
