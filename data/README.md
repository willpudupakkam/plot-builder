# Data folder

`sample.csv` is a small synthetic dataset included so Plot Builder works immediately after cloning.

Add personal CSV files directly to this folder or group them into one-level workspace directories, for example:

```text
data/
├── sample.csv
├── observation-a/
│   └── catalogue.csv
└── observation-b/
    ├── catalogue-1.csv
    └── catalogue-2.csv
```

Every directory containing CSV files appears as a workspace in the app. CSV column names become available as axes, colors, histogram fields, derived-column inputs, and filters.

All CSV files except `sample.csv` are ignored by Git. Before deliberately publishing another dataset, confirm that you have permission to redistribute it and document its source, license, processing, and any privacy considerations.
