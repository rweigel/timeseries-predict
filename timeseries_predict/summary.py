"""
Summary Generation Script

This script automates the process of summarizing model evaluation results for 
leave-one-out (LOO) experiments with optional input removals. It reads configuration 
and result files, computes evaluation statistics using bootstrap samples, 
generates plots for visual comparison, and compiles the results into a summary table.

"""
import os
import json

import pandas as pd

from .plot import plot
from .table import table
from .stats import stats

def summary(tag, results_dir=None):
    """
    Generate a summary of model performance for a given experiment tag.

    This function reads experiment configurations and results, computes ARV-based
    statistics using bootstrap samples, generates plots for actual vs. predicted
    comparisons, and compiles a summary table. It is designed for leave-one-out 
    experiments where certain input features may have been removed.

    Parameters:
    ----------
    tag : str
        The identifier for the experiment, used to locate its result subdirectory.

    results_dir : str, optional
        The base directory where experiment results are stored. If None, defaults to './results'.

    Returns:
    -------
    None
        Outputs plots and a summary table to disk within the results directory.
    """
    if results_dir is None:
        results_dir = './results'

    config_path = os.path.join(results_dir, tag, 'config.json')
    print(f"Reading: {config_path}")
    with open(config_path, 'r') as f:
        kwargs = json.load(f)

    directory = os.path.join(results_dir, tag)

    _stats = {}

    # Iterate through subdirectories under each pattern
    for removed_input in kwargs['removed_inputs']:
        if removed_input is None:
            removed_input = 'None'

        removed_input_dir = os.path.join(directory, removed_input)

        if not os.path.isdir(removed_input_dir):
            print(f"    Removed input dir '{removed_input_dir}' not found. Skipping.")
            continue

        print(f"    Processing removed input dir: {removed_input_dir}")

        _stats[removed_input] = []

        loo_dir = os.path.join(removed_input_dir, 'loo')
        for file_name in sorted(os.listdir(loo_dir)):
            if not file_name.endswith('.pkl'):
                continue
            file_path = os.path.join(loo_dir, file_name)

            # pkl file contains a list of bootstrap results, each in the form of a dict
            # [{actual: df, ols: df, nn_miso: df, nn_mimo: df}, ...]
            # where df is a DataFrame with columns for each output
            print(f"      Reading: {file_path}")
            boots = pd.read_pickle(file_path)  # Array of bootstrap results

            # Each element of stats[removed_input] is a loo repetition across
            # all bootstrap repetitions for a given removed input.
            stat = stats(boots, kwargs['outputs'])
            _stats[removed_input].append(stat)

            plot_path = os.path.join(loo_dir, 'figures', file_name)
            plot(boots, plot_path.replace('.pkl', ''), stats)

    table(_stats, directory, **kwargs)
