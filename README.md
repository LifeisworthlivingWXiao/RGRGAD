# RGRGAD

## Get Started

This is the source code of **RGRGAD**, a routing-guided graph anomaly detection framework for attributed networks.
The datasets are included in the `Data/` folder.

RGRGAD performs unsupervised node-level anomaly detection by using routing-guided structural refinement, including redundancy pruning and neighbor completion, to construct informative graph views for contrastive learning.

## Code Structure

| File / Folder | Description |
|:-------------:|:------------|
| `Data` | Datasets in `.mat` format. |
| `run.py` | Training and evaluation entry. |
| `model.py` | GCN encoder, discriminator, and routing gate. |
| `aug.py` | Redundancy pruning and neighbor completion. |
| `utils.py` | Data loading, preprocessing, and subgraph sampling. |

## Datasets

| **Dataset** | # Nodes | # Edges | # Attributes | # Anomalies |
|:-----------:|:-------:|:-------:|:------------:|:-----------:|
| **Cora** | 2,708 | 5,429 | 1,433 | 5.5% |
| **Citeseer** | 3,327 | 4,732 | 3,703 | 4.5% |
| **Pubmed** | 19,717 | 44,338 | 500 | 3.0% |
| **ACM** | 16,484 | 71,980 | 8,337 | 3.6% |
| **BlogCatalog** | 5,196 | 171,743 | 8,189 | 5.8% |
| **Reddit** | 10,984 | 168,016 | 64 | 3.3% |

The datasets should be organized as follows:

```text
RGRGAD/
└── Data/
    ├── acm.mat
    ├── blogcatalog.mat
    ├── citeseer.mat
    ├── cora.mat
    ├── pubmed.mat
    └── reddit.mat
```

The `.mat` files should contain node labels, node attributes, and the adjacency matrix. The loader supports the following field names:

```text
Labels:     Label / gnd / label
Attributes: Attributes / X / attr
Adjacency:  Network / A / adj
```

## Usage

Run RGRGAD on Cora:

```bash
python run.py --dataset cora --data_dir ./Data --lr 0.001 --train_epoch 100 --threshold 8 --alpha 0.2 --tau 0.07 --gpu_id 0
```

Run RGRGAD on other datasets by changing the dataset name:

```bash
python run.py --dataset acm --data_dir ./Data
python run.py --dataset blogcatalog --data_dir ./Data
python run.py --dataset citeseer --data_dir ./Data
python run.py --dataset pubmed --data_dir ./Data
python run.py --dataset reddit --data_dir ./Data
```

## Requirements

This code requires the following:

- python>=3.8
- pyTorch>=1.12.1
- dgl<=0.4.3
- numpy>=1.23.5
- scipy>=1.9.1
- scikit-learn>=1.2.2
- torch-scatter>=2.0.9
- tqdm>=4.64.1

## Output

The program reports the final node-level anomaly detection performance using ROC-AUC:

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

## Cite

If this repository is useful for your research, please cite the corresponding paper when it becomes available.
