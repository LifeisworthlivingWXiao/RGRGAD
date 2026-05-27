# RGRGAD

PyTorch implementation of **RGRGAD**, a routing-guided graph anomaly detection framework for attributed networks.

RGRGAD is designed for unsupervised node-level anomaly detection on attributed graphs. The method refines graph structure through routing-guided augmentation and learns anomaly-sensitive representations with contrastive subgraph modeling.

## Overview

The main idea of RGRGAD is to use training dynamics and structural information to guide graph refinement during representation learning. The implementation contains the following components:

- **Routing-guided structural refinement** for dynamically adjusting graph views.
- **Redundancy pruning** for suppressing unreliable or redundant structural signals.
- **Neighbor completion** for enriching insufficient local contexts of low-degree nodes.
- **Routing gate** for estimating node-level routing confidence from pseudo anomaly scores and degree information.
- **Contrastive subgraph learning** for unsupervised anomaly scoring.

The code reports the final node-level anomaly detection performance using ROC-AUC.

## Repository Structure

```text
RGRGAD/
├── Data/
│   ├── acm.mat
│   ├── blogcatalog.mat
│   ├── citeseer.mat
│   ├── cora.mat
│   ├── pubmed.mat
│   └── reddit.mat
├── aug.py        # Graph augmentation: redundancy pruning and neighbor completion
├── model.py      # GCN encoder, discriminator, and routing gate
├── run.py        # Training and evaluation entry point
├── utils.py      # Data loading, preprocessing, and subgraph sampling utilities
└── README.md
```

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

A typical environment can be created as follows:

```bash
conda create -n rgrgad python=3.8
conda activate rgrgad
```

Install PyTorch, DGL, and `torch-scatter` according to your CUDA version. Then install the remaining dependencies:

```bash
pip install numpy==1.23.5 scipy==1.9.1 scikit-learn==1.2.2 tqdm==4.64.1
```

## Dataset

All `.mat` dataset files should be placed under the `Data/` directory:

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

The dataset name used in the command line should match the file name without the `.mat` suffix. For example, `--dataset cora` loads `Data/cora.mat`.

The loader supports the following common `.mat` field names:

```text
Labels:     Label / gnd / label
Features:   Attributes / X / attr
Adjacency:  Network / A / adj
```

Each `.mat` file should contain one label vector, one node attribute matrix, and one adjacency matrix. The labels are only used for final evaluation, not for supervised training.

## Usage

Run RGRGAD on Cora:

```bash
python run.py --dataset cora --data_dir ./Data
```

Run RGRGAD on other datasets:

```bash
python run.py --dataset acm --data_dir ./Data
python run.py --dataset blogcatalog --data_dir ./Data
python run.py --dataset citeseer --data_dir ./Data
python run.py --dataset pubmed --data_dir ./Data
python run.py --dataset reddit --data_dir ./Data
```

You can also specify the random seed and GPU id:

```bash
python run.py --dataset cora --data_dir ./Data --seed 1 --gpu_id 0
```

## Main Arguments

| Argument | Description | Default |
| --- | --- | --- |
| `--dataset` | Dataset name without `.mat` suffix | `cora` |
| `--data_dir` | Directory containing `.mat` datasets | `./Data` |
| `--seed` | Random seed | `1` |
| `--gpu_id` | GPU id | `0` |
| `--train_epoch` | Number of training epochs | `100` |
| `--test_rounds` | Number of test rounds for score averaging | `196` |
| `--batch_size` | Batch size | `128` |
| `--embedding_dim` | Hidden representation dimension | `64` |
| `--threshold` | Degree threshold for structural refinement | `8` |
| `--routing_mode` | Routing mode: `feat_only`, `feat_ano`, `feat_routing`, or `full` | `full` |
| `--train_stage` | Augmentation stage: `pruning_only`, `completion_only`, or `staged` | `staged` |
| `--amp` | Enable automatic mixed precision | Disabled |
| `--fast_cuda` | Enable faster CUDA behavior with less deterministic settings | Disabled |

## Output

The program prints the training progress and final evaluation result in the terminal. A typical output is:

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

## Reproducibility Notes

- The default setting uses one random seed specified by `--seed`.
- For multiple-seed evaluation, run the script repeatedly with different seeds and report the mean and standard deviation.
- The code uses the labels only when computing the final ROC-AUC score.
- Dataset files should remain in the `Data/` directory to avoid path-related loading errors.
- If CUDA memory is limited, reduce `--batch_size` or disable `--amp` depending on your environment.

## Citation

If this repository is useful for your research, please cite the corresponding paper when it becomes available.

## Contact

For questions about the implementation, please open an issue in this repository.
