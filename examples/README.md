# Optional feature extensions

Plot Builder can augment every loaded catalogue with a local module named `plot_builder_features.py` in the repository root. The module may define:

```python
def augment_catalogue(data: pandas.DataFrame, path: pathlib.Path) -> pandas.DataFrame:
    ...
```

The function receives a copy of the loaded DataFrame and its CSV path. It must return a pandas DataFrame. If the root module does not exist or does not provide the function, Plot Builder loads the original data unchanged.

## Astronomy example

`astro_features.py` demonstrates an extension for merger-catalogue data. It expects:

- An object ID column named `#ID(1)`
- Position columns named `Xc(6)`, `Yc(7)`, and `Zc(8)`
- A column named `theta_<secondary ID>`
- The first row to represent the main cluster
- A row whose `#ID(1)` matches the secondary ID encoded in the theta column

For compatible catalogues it calculates signed merger-axis distance, normalized distance, perpendicular distance and radius, secondary ID, separation, cluster roles, and plot-highlight labels and colors. Incompatible catalogues are returned unchanged.

Enable the example locally:

```bash
cp examples/astro_features.py plot_builder_features.py
```

Restart Streamlit after enabling or changing the extension. The root destination is ignored by Git intentionally.

Use this file as a starting point for other domains, but keep reusable examples free of private data, credentials, and machine-specific paths.
