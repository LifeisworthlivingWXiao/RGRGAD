````markdown
# RGRGAD

Implementation of routing-guided graph anomaly detection on attributed networks.

This repository only keeps the core implementation files needed to run the model. No datasets, checkpoints, logs, JSON files, NPZ files, figures, or experimental result files are included.

## File Structure

```text
RGRGAD/
├── run.py      # Training and evaluation entry
├── model.py    # GCN encoder, discriminator, and routing gate
├── aug.py      # Redundancy pruning and neighbor completion
├── utils.py    # .mat loading, preprocessing, and subgraph sampling
└── README.md
````

## Environment

The code was tested with the following environment:

```text
Python 3.8.13
PyTorch 1.12.1+cu113
DGL 0.4.3
NumPy 1.23.5
SciPy 1.9.1
scikit-learn 1.2.2
torch-scatter 2.0.9
tqdm 4.64.1
```

## Dataset

Create a `Data/` folder under the project directory and place the `.mat` dataset file in it.

For example, to run Cora:

```text
RGRGAD/
├── Data/
│   └── cora.mat
├── run.py
├── model.py
├── aug.py
├── utils.py
└── README.md
```

The loader supports the following common `.mat` field names:

```text
Labels:     Label / gnd / label
Features:   Attributes / X / attr
Adjacency:  Network / A / adj
```

## Run

To run RGRGAD on Cora:

```bash
python run.py --dataset cora --data_dir ./Data
```

The program only prints the progress and final result in the terminal.

Example output:

```text
====================================
[RGRGAD] FINAL RESULT
Dataset    : cora
Seed       : 1
Best epoch : xx
Best loss  : x.xxxxxx
ROC-AUC    : x.xxxx
====================================
```

## Notes

This release is intended for simple reproduction and code inspection. It does not save checkpoints, logs, JSON files, NPZ files, or figures by default.

```
```
