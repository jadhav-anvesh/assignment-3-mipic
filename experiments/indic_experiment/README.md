# Indic MRL Experiment

This folder contains an additional Assignment 3 experiment testing whether
**Matryoshka Representation Learning (MRL)** improves sentence representations
beyond the English-focused experiments in the main report.

## Experiment

- Base model: `huawei-noah/TinyBERT_General_6L_768D`
- Languages: Hindi, Bengali, Gujarati, Tamil, and Telugu
- MRL dimensions: `16, 32, 64, 128, 256, 512, 768`
- Training data: 20,240 unlabeled Indic sentences
- Evaluation: Indic sentiment classification using a separate Logistic
  Regression probe for each language and dimension
- Metrics: accuracy and macro-F1

The MRL training data and labeled probe data are kept separate. Each
language-specific sentiment split contains 156 training examples and 998 test
examples. The probe classifier is trained on the sentiment training split and
evaluated on the corresponding test split.

The encoder is trained for five epochs with AdamW (learning rate `2e-5`,
weight decay `0.01`), batch size `32`, maximum sequence length `128`, and
InfoNCE temperature `0.05`. Sentence representations use mean pooling over
non-padding tokens. The frozen baseline is the original pretrained TinyBERT
checkpoint; it uses the same pooling and probing pipeline as MRL.

This is a single-seed (`42`) exploratory experiment. The results show whether
the direction of the MRL effect transfers across these languages, but they do
not establish statistical significance or language independence.

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
