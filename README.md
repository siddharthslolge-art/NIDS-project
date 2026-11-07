# NIDS Project

A network intrusion detection system built with Python and machine learning,
trained on the NSL-KDD dataset.

Repository: https://github.com/siddharthslolge-art/NIDS-project

## What it does

Takes network connection records and classifies each one as normal or attack.
Compares three models:

- Random Forest
- K-Nearest Neighbours
- PyTorch Neural Network

Also includes a socket-based server/client demo for real-time classification.

## Setup

Install the required packages:

    pip install pandas numpy scikit-learn matplotlib seaborn joblib torch

## How to run

Run the scripts in this order:

    python part1_eda.py          # Exploratory data analysis + charts
    python part2_ml_model.py     # Train Random Forest and KNN
    python part3_pytorch_nn.py   # Train PyTorch neural network
    python part4_compare.py      # Compare all three models on Test.csv

For the live demo, open two terminals:

Terminal 1:

    python part5_server.py

Terminal 2:

    python part5_client.py

## Results

Internal validation (Train.csv, 80/20 split):

| Model | Accuracy |
|---|---|
| Random Forest | 99.90% |
| KNN (k=5) | 98.96% |
| PyTorch NN | 99.43% |

Hold-out evaluation (Test.csv):

| Model | Accuracy |
|---|---|
| Random Forest | 76.59% |
| KNN (k=5) | 77.06% |
| PyTorch NN | 77.85% |

The drop from 99% to 77% on the hold-out set is expected for NSL-KDD.
Train.csv is easy to learn; Test.csv contains harder and unseen attack
types.

## Files

| File | Description |
|---|---|
| part1_eda.py | Data exploration and charts |
| part2_ml_model.py | Random Forest + KNN training |
| part3_pytorch_nn.py | Neural network training |
| part4_compare.py | Final comparison on test set |
| part5_server.py | Socket server for live classification |
| part5_client.py | Socket client that sends test records |
| 2_preprocess.py | Shared preprocessing utilities |
| data/ | Train and test CSV files |
| figures/ | Output charts |

## Dataset

Tavallaee, M., Bagheri, E., Lu, W., & Ghorbani, A. A. (2009).
A detailed analysis of the KDD CUP 99 data set. IEEE CISDA.
