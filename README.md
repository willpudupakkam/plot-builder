# Plot Builder

A general-purpose Streamlit app for plotting CSV files.

## Features

- Reads CSV files from `data/`
- Plot one CSV file or all CSV files together
- Uses CSV column names as available axes and colors
- Create derived numeric columns with a small equation editor
- 2D and 3D scatter plots
- Histograms
- Plot styling controls for plot background, point colors, histogram colors, split colors, numeric color palettes, and split-histogram draw order
- Custom filters:
  - interval filter `[a, b]`
  - equals one value
  - not equals one value
- Interval endpoints can be blank, `inf`, `infinity`, `-inf`, or `-infinity`
- Download filtered data as CSV
- Download plots as standalone HTML
- Saves derived quantities and filters locally between app restarts, separated by workspace name

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL shown by Streamlit.

## Add Your Data

Put CSV files in:

```text
data/
```

The app automatically finds every `*.csv` file in that folder.

## Derived Columns

Open **Derived Columns** in the sidebar, choose how many new columns to make, and write formulas using the clickable column buttons. Each button is labeled with the exact CSV column name and inserts that same column name using pandas backtick syntax, so columns like `M_star(65)` or `x position [kpc/h]` can still be evaluated correctly.

Column and function buttons only insert text into the formula box. The derived column is not evaluated until you press **Apply Formula**.

Examples:

```text
sqrt(`x`**2 + `y`**2)
log10(`M_star(65)`)
where(`M_gas(45)` > 0, `M_gas(45)` / `M_star(65)`, 0)
```

Supported operators include `+`, `-`, `*`, `/`, `**`, and comparisons. Supported functions include `sqrt`, `log`, `log10`, `abs`, `sin`, `cos`, `tan`, `exp`, `where`, `minimum`, `maximum`, and `clip`.

## Notes

- Scatter plot axes require numeric columns.
- Histogram columns require numeric columns.
- Derived columns are evaluated as numeric columns and can be used as axes, colors, histogram inputs, or filters.
- Derived quantities and filters are saved in `.plot_builder_state/<workspace>.json`, which is ignored by Git. The default workspace is `default`; use a different workspace name when multiple people share the same deployed app.
- Color can use numeric or categorical columns.
- Numeric color columns use a selectable color palette; categorical colors and histogram splits can be colored independently.
- Split histograms can choose which split value is drawn on top.
- When plotting all CSV files together, the app adds a `source_file` column so rows can be colored, split, or filtered by file.
