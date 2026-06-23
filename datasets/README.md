# CSV Dataset Release (RGRGAD)

This folder documents the human- and machine-readable CSV versions of the benchmark datasets used in RGRGAD.

## Download

The full dataset archive is provided via the GitHub Release:

https://github.com/LifeisworthlivingWXiao/RGRGAD/releases/tag/dataset-v1.0

Direct download:

https://github.com/LifeisworthlivingWXiao/RGRGAD/releases/download/dataset-v1.0/csvdata.zip

## Contents

The release contains the following six benchmark datasets:

- ACM
- BlogCatalog
- CiteSeer
- Cora
- PubMed
- Reddit

Each dataset directory contains:

- `nodes.csv`
- `edges.csv`
- `attributes.csv`
- `metadata.json`
- `README.txt`

The `nodes.csv` file records node identifiers and available node-level labels. The `edges.csv` file contains the graph edge list, and the `attributes.csv` file contains the sparse node attribute matrix. The `metadata.json` and `README.txt` files describe the dataset structure, source, matrix properties, indexing conventions, and format-conversion procedure.

## Note

These CSV files are converted from the original MATLAB-format benchmark datasets obtained from the public CoLA repository. The conversion changes only the storage format and preserves the graph contents and node ordering. Node labels are used only for evaluation and are not used during unsupervised model training.
