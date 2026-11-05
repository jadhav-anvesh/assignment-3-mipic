# Indic MRL Experiment

This folder contains the Assignment 3 experiment testing **Matryoshka
Representation Learning (MRL)** for Indic-language sentence representations.

## Experiment

- Base model: `huawei-noah/TinyBERT_General_6L_768D`
- Languages: Hindi, Bengali, Gujarati, Tamil, and Telugu
- MRL dimensions: `16, 32, 64, 128, 256, 512, 768`
- Training data: unlabeled Indic sentences
- Evaluation: Indic sentiment classification using a Logistic Regression probe
- Metrics: accuracy and macro-F1

The MRL training data and labeled probe data are kept separate. The probe
classifier is trained on the sentiment training split and evaluated on the
corresponding test split.

## Contents

```text
indic_experiment/
├── code/
│   └── a3-indic.ipynb
└── result/
    ├── comparison_metrics.csv
    ├── mrl_metrics.csv
    ├── mrl_vs_frozen.png
    └── table.png
```

Open `code/a3-indic.ipynb` in Kaggle or Jupyter to reproduce the experiment.
Use a GPU runtime for training.
