# Infinitely Wide Neural Network Kernels for Gaussian Process Regression

This project is an investigation of different types of single hidden layer infinitely wide neural networks and how they relate to Gaussian Processes. 
In this study, we replicate the derivation of the covariance functions (kernel) of a single hidden layer neural network with no biases under a cosine activation
function, and we also derive an alternate non-analytic expression for the covariance under a hyperbolic tangent activation function. 
These derived covariance functions are also applied to a GP for regression on five benchmark regression datasets.

All models were trained on a training subset and evaluated on a seperate test set. 
Predictive performance the kernels were assessed using the Root Mean Squared Error (RMSE) between predicted and actual outputs. 
We also examine the convergence behaviour, using the Mean Squared Error (MSE) between the finite and infinite kernels.

A link to the paper will be provided later. In the meantime, check the bottom for [derivations](#derivations).

## Motivation

Neural networks with randomly initialized weights converge to Gaussian processes as the width of their hidden layers approaches infinity (Neal, 1995).  
This connection allows neural networks to be interpreted entirely through kernels used in Gaussian Process models.

While several kernels derived from neural networks are known, closed-form expressions are not available for many activation functions used in modern architectures.

This project investigates:

1. How covariance functions can be derived from neural networks with random weights.
2. Whether these kernels perform well in Gaussian Process regression.

## Installing Dependencies

Clone the repository and install the requirements:

```bash
pip install -r requirements.txt
```

## Reproducing the results

To replicate the GP model training experiment with the implemented kernels:

```bash
python Experiments/regression/gp_regression_experiment.py
```

To replicate the MSE tests for examining convergence behaviour:

```bash
python Experiments/regression/mean_std_mse.py
```

Datasets can be chosen by changing ```datasets``` in ```Experiments/regression/gp_regression_experiment.py``` to any of the datasets listed out in ```Experiments/datasets/dataset_utils.py```
## Derivations

### Background

In this section, we outline concepts from existing literature required for the kernel derivations, including the definition of a GP, the structure of a single hidden layer neural network and the model for GP regression.

A Gaussian process (GP) is a potentially infinite set of random variables such that the joint distribution of every finite subset of these random variables is multivariate Gaussian:
```math
f \sim \mathcal{GP}(m(\mathbf{x}), k(\mathbf{x}_1, \mathbf{x}_2)) \label{eq:gp}
```

where the mean function and covariance function are defined as:
```math
    m(\mathbf{x}) = \mathbb{E}[f(\mathbf{x})]
```    
```math
    k(\mathbf{x}_1, \mathbf{x}_2) = \mathbb{E}\big[(f(\mathbf{x}_1) - \mathbb{E}[f(\mathbf{x}_1)]) (f(\mathbf{x}_2) - \mathbb{E}[f(\mathbf{x}_2)])\big]
```
\end{equation}
where $\mathbf{x} \in \mathbb{R}^d$ 

## Notes

This project was completed as part of the FIT2082 research project at Monash University under the supervision of Russell Tsuchida.
