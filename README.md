# Overview

`main.py` runs neural network (both with 3 outputs and 1 output) and linear regression to train and test on the .pkl files while removing one parameter in each iteration.

The leave-one-out method is used, where a year of data is used as the test set and all remaining data is used as training, generating out-of-sample predictions. Repetitions with random sampling can be applied.

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
