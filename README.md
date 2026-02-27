# Overview

`timeseries-predict` computes time series models given Pandas `DataFrame`s.

Given a `DataFrame` with $n_i$ inputs and $n_o$ outputs, it can be configured to
* compute $n_o$ ordinary linear regression models, each with $n_i$ inputs;
* create $n_o$ neural network models with $n_i$ inputs;
* create a neural network model with $n_i$ inputs and $n_o$ outputs;
* create a neural network model that predicts the output residuals of a regression model; and
* compute $n_i-1$ models by leaving one input out.

Options include
* parallel computation of models;
* parallelization of neural network training;
* options for splitting time series into training, testing, and validation intervals  continuous chunks;
* extensive diagnostics, logging, and metrics calculations;
* uncertainty in metrics using the bootstrap; and
* visualization of predictions.

See `configs/satellite-b` for example usage.

# Installation

Setup:

```
git clone https://github.com/rweigel/timeseries-predict
conda create -n python3.9.12-timeseries-predict python=3.9.12
conda activate python3.9.12-timeseries-predict
pip install -e .
```

# Terms

* `run` - A list of jobs. The output results are stored in a `run_dir` given in this file. The user provides a function named `job_list` that is passed the configuration file specified as the command line value for `--config`.
* `job` - One or more `DataFrame`s and configuration. The configuration for each job is typically based on modifying the `run` configuration.
* `segment` - Each job contains $n_s$ `DataFrame`s. Each `DataFrame` in a job is called a segment. When $n_s>1$, training and testing is done using all unique sets of $n_s-1$ `DataFrame`s and validation is done on the remaining `DataFrame`. (In the code this is referred to as leave-one-out).

`timeseries-predict/config/satellite-b` has a collection of run configurations (YAML files).

Each configuration specifies an output directory of the form

`timeseries-predict/data/results/{run_dir}`

Each job has an associated subdirectory

```
timeseries-predict/data/results/{run_dir}/{job1}
timeseries-predict/data/results/{run_dir}/{job2}
...
```

# Example

From the directory `timeseries-predict/config/satellite-b`, execute

```
python data.py        # Downloads data to data/raw/satellite-b/files
python data_plot.py   # Creates plots in data/raw/satellite-b/plots
```

From the directory `timeseries-predict`, execute a run using

```
python main.py --config configs/satellite-b/test-parallel.yaml
python main.py --config configs/satellite-b/test-serial.yaml
```

# Development

If any post-processing code changes, runs can be re-postprocessed.

Re-postprocess all jobs from a run by passing the run results directory:
```
python main.py --postprocess data/results/satellite-b/results-0-serial
```

Re-postprocess job with job from a run (will process only job results in subdir `cluster1`)
```
python main.py --job cluster1 --postprocess data/results/satellite-b/results-0-serial
```