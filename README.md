# [NeurIPS 2022] Supervising the Multi-Fidelity Race of Hyperparameter Configurations

## Introduction

This repo contains the source code accompanying the paper:

**Supervising the Multi-Fidelity Race of Hyperparameter Configurations**

Authors: Martin Wistuba*, Arlind Kadra* and Josif Grabocka

Multi-fidelity (gray-box) hyperparameter optimization techniques (HPO) have recently emerged as a promising direction for tuning Deep Learning methods. However, existing methods suffer from a sub-optimal allocation of the HPO budget to the hyperparameter configurations. In this work, we introduce DyHPO, a Bayesian Optimization method that learns to decide which hyperparameter configuration to train further in a dynamic race among all feasible configurations. We propose a new deep kernel for Gaussian Processes that embeds the learning curve dynamics, and an acquisition function that incorporates multi-budget information. We demonstrate the significant superiority of DyHPO against state-of-the-art hyperparameter optimization methods through large-scale experiments comprising 50 datasets (Tabular, Image, NLP) and diverse architectures (MLP, CNN/NAS, RNN).


## Citation
```
@inproceedings{
wistuba2022supervising,
title={Supervising the Multi-Fidelity Race of Hyperparameter Configurations},
author={Martin Wistuba and Arlind Kadra and Josif Grabocka},
booktitle={Thirty-Sixth Conference on Neural Information Processing Systems},
year={2022},
url={https://openreview.net/forum?id=0Fe7bAWmJr}
}
