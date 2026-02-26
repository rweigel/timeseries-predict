# Overview

`timeseries-predict` computes time series models given a Pandas `DataFrame`.

For $n_i$ inputs and $n_o$ outputs, it can be configured to
* compute $n_o$ linear regression models, each with $n_i$ inputs
* create a neural network model with $n_i$ inputs and $n_o$ outputs
* create a $n_o$ neural network model with $n_i$ inputs
* create a neural network model that predicts the residuals of a regression model
* compute $n_i-1$ models by leaving one input out

Options include
* Parallel computation of models
* Parallelization of neural network training
* Options for splitting time series into training and testing intervals by continuous chunks
* Extensive diagnostics, logging, and metrics calculations
* Uncertainty in metrics using the bootstrap
* Visualization of predictions

See `configs/satellite-b` for example usage.

# Installation

Setup:

```
git clone https://github.com/rweigel/timeseries-predict
conda create -n python3.9.12-timeseries-predict python=3.9.12
conda activate python3.9.12-timeseries-predict
pip install -e .
```

# Running

From the directory `timeseries-predict/config/satellite-b`, execute

```
python data.py        # Downloads data to data/raw/satellite-b/files
python data_plot.py   # Creates plots in data/raw/satellite-b/plots
```

From the directory `timeseries-predict`, execute

```
python main.py --config configs/satellite-b/test-parallel.yaml
```
