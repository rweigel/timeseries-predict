import os
import json

import pandas as pd

from .plot import plot
from .table import table
from .stats import stats

def summary(run_dir, job=None):

  if job is None:
    # If job is not provided, loop over subdirectories in run_dir
    if not os.path.isdir(run_dir):
      print(f"Error: Config file '{run_dir}' not found. Skipping postprocessing.")
      return

    for subdir in sorted(os.listdir(run_dir)):
      subdir_path = os.path.join(run_dir, subdir)
      if os.path.isdir(subdir_path):
        print(f"\n    Processing job: {subdir}")
        summary(run_dir, job=subdir)
    return

  config_path = os.path.join(run_dir, job, 'config.json')
  if not os.path.isfile(config_path):
    emsg = f"Config file '{config_path}' not found for job '{job}'. Skipping."
    print(f"    Error: {emsg}")
    return

  print(f"    Reading: {config_path}")
  with open(config_path, 'r') as f:
    kwargs = json.load(f)

  directory = os.path.dirname(config_path)

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

      """pkl file contains a list of bootstrap results, each in
         the form of a dict
      [
        {
          actual: df, # Validation DataFrame
          ols: df,    # OLS predicted DataFrame
          nn_miso: df,
          nn_mimo: df
        }, ...
      ]
      """

      print(f"      Reading: {file_name}")
      boots = pd.read_pickle(file_path) # Array of bootstrap results

      # Each element of stats[removed_input] is stats for a given removed input
      # and for a loo repetition across all bootstrap repetitions.
      stat = stats(boots, kwargs['outputs'])

      _stats[removed_input].append(stat)

      plot_subdir = os.path.join(job, removed_input, 'loo', 'figures')
      file_base = file_name.replace('.pkl', '')
      plot(boots, run_dir, plot_subdir, file_base)

  table(_stats, directory, **kwargs)
