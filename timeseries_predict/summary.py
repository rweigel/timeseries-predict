import os
import json

import pandas as pd

from .plot import plot
from .table import table
from .stats import stats

def summary(run_dir, job=None):

  if job is None:
    # Check if a subdirectory of a run directory was given. Technically an
    # error, but we know what was meant.
    if os.path.exists(os.path.join(run_dir, 'config.json')):
      # Use the subdirectory name as the job name
      job = os.path.basename(run_dir)
      # Use the parent directory of given run_dir as the run directory
      run_dir = os.path.dirname(run_dir)

  if job is None:
    # If job is not provided, loop over subdirectories in run_dir
    if not os.path.isdir(run_dir):
      print(f"Error: Run directory '{run_dir}' not found. Skipping postprocessing.")
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

  job_dir = os.path.dirname(config_path)

  _stats = {}

  # Iterate through subdirectories under each pattern
  for method in ['loo', 'lno']:
    print(f"    Looking for {os.path.join(job_dir, method)}")
    if not os.path.isdir(os.path.join(job_dir, method)):
      continue

    for removed_input in kwargs['removed_inputs']:
      if removed_input is None:
        removed_input = 'None'

      method_dir = os.path.join(job_dir, method)

      removed_input_dir = os.path.join(method_dir, removed_input)

      if not os.path.isdir(removed_input_dir):
        print(f"    Removed input dir '{removed_input_dir}' not found. Skipping.")
        continue

      print(f"    Processing removed input dir: {removed_input_dir}")

      _stats[removed_input] = []

      for file_name in sorted(os.listdir(removed_input_dir)):
        if not file_name.endswith('.pkl'):
          continue

        file_path = os.path.join(removed_input_dir, file_name)

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
        reps = pd.read_pickle(file_path) # Array of bootstrap repetitions

        # Each element of stats[removed_input] is stats for a given removed input
        # and for a loo repetition across all bootstrap repetitions.
        stat = stats(reps, kwargs['outputs'])

        _stats[removed_input].append(stat)

        plot_dir = os.path.join(removed_input_dir, 'figures')
        file_base = file_name.replace('.pkl', '')
        plot(reps, plot_dir, file_base)

  table(_stats, job_dir, **kwargs)
