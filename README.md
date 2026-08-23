# Plot Builder

Plot Builder is a local Streamlit app for exploring and plotting CSV data without writing plotting code.

## Features

- Load one CSV, combine all CSVs in a workspace, or plot them separately
- Build 2D and 3D scatter plots and histograms
- Create global or per-catalogue derived numeric columns
- Filter with intervals, equality, and inequality conditions
- Color by numeric or categorical columns and control trace order
- Add reference lines, point labels, and highlighted points
- Save plots locally as HTML or PNG
- Download filtered data and standalone Plotly HTML
- Keep derived columns, filters, and styles in per-workspace local state
- Optionally augment loaded catalogues with a local Python extension

## Requirements

- Python 3.10 or newer
- A modern web browser
- Chrome or Chromium only if you want PNG export; HTML export needs no external browser

## Quick start

```bash
git clone https://github.com/willpudupakkam/plot-builder.git
cd plot-builder
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

The repository includes a small synthetic dataset, so the app works immediately after cloning.

## Add data

Place CSV files directly in `data/` or organize them into one-level workspace folders:

```text
data/
├── sample.csv
├── experiment-a/
│   ├── run-01.csv
│   └── run-02.csv
└── experiment-b/
    └── run-01.csv
```

Each folder containing CSV files appears as a separate workspace. Real CSV files are ignored by Git by default; only the synthetic `data/sample.csv` is tracked.

When multiple files are loaded together, Plot Builder adds a `source_file` column that can be used for colors, histogram splits, and filters.

## Derived columns

Open **Derived Columns** in the sidebar, add a column, and enter an expression. Column buttons insert exact names using pandas backtick syntax, allowing names such as `M_star(65)` or `x position [kpc/h]`.

Examples:

```text
sqrt(`x`**2 + `y`**2)
log10(`size`)
where(`temperature` > 20, `size` / 2, 0)
```

Supported operators include `+`, `-`, `*`, `/`, `**`, and comparisons. Supported functions include `sqrt`, `log`, `log10`, `abs`, `sin`, `cos`, `tan`, `exp`, `where`, `minimum`, `maximum`, and `clip`.

## Optional catalogue extensions

At load time, the app looks for an untracked file named `plot_builder_features.py` in the repository root. If it contains a callable named `augment_catalogue`, the app calls it for every loaded CSV:

```python
def augment_catalogue(data: pandas.DataFrame, path: pathlib.Path) -> pandas.DataFrame:
    ...
```

An astronomy-specific example is provided in [`examples/astro_features.py`](examples/astro_features.py). Enable it locally with:

```bash
cp examples/astro_features.py plot_builder_features.py
```

The destination is ignored by Git so that local or dataset-specific logic is not accidentally published. See [`examples/README.md`](examples/README.md) for its expected catalogue columns and generated fields.

## PNG export

Kaleido is installed with the project dependencies. It needs Chrome or Chromium to render PNG files. If no compatible browser is installed, run:

```bash
plotly_get_chrome
```

Downloaded browsers and saved plots remain local and are ignored by Git. HTML saving and downloading continue to work without Chrome.

## Local files and privacy

Plot Builder writes settings to `.plot_builder_state/` and saved files to `saved_plots/`. These paths, local extensions, browser binaries, secrets, and user CSV data are ignored by Git.

Before sharing an exported HTML plot, remember that Plotly HTML generally contains the plotted values.

## Development checks

```bash
python -m py_compile app.py examples/astro_features.py
python -m unittest discover -s tests
```
