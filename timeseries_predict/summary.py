import os
import yaml

import pandas as pd

from .plot import plot
from .table import table
from .stats import stats

def summary(run_dir, job=None):

  if job is None:
    # Check if a subdirectory of a run directory was given. Technically an
    # error, but we know what was meant.
    if os.path.exists(os.path.join(run_dir, 'config.yaml')):
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

  print("  Starting table and plot generation.")

  config_path = os.path.join(run_dir, job, 'config.yaml')
  if not os.path.isfile(config_path):
    emsg = f"Config file '{config_path}' not found for job '{job}'. Skipping."
    print(f"    Error: {emsg}")
    return

  print(f"    Reading: {config_path}")
  with open(config_path, 'r') as f:
    kwargs = yaml.safe_load(f)

  job_dir = os.path.dirname(config_path)

  reps_stats = {}

  # Iterate through subdirectories under each pattern
  for method in ['loo', 'lno']:
    print(f"    Looking for {os.path.join(job_dir, method)}")
    if not os.path.isdir(os.path.join(job_dir, method)):
      continue

    for removed_input in kwargs['removed_inputs']:
      if method == 'lno':
        sub_dir = ''
      else:
        if removed_input is None:
          removed_input = 'None'
          sub_dir = removed_input

      method_dir = os.path.join(job_dir, method)

      sub_dir = os.path.join(method_dir, sub_dir)

      if not os.path.isdir(sub_dir):
        print(f"    Dir '{sub_dir}' not found. Skipping.")
        continue

      print(f"    Processing dir: {sub_dir}")

      reps_stats[removed_input] = []

      for file_name in sorted(os.listdir(sub_dir)):
        if not file_name.endswith('.pkl'):
          continue

        file_path = os.path.join(sub_dir, file_name)

        print(f"      Reading: {file_name}")
        reps = pd.read_pickle(file_path) # Array of repetitions

        reps_stats[removed_input].append(stats(reps))

        plot_dir = os.path.join(sub_dir, 'figures')
        file_base = file_name.replace('.pkl', '')
        plot(reps, plot_dir, file_base)

  table(reps_stats, job_dir, **kwargs)
  print("  Finished table and plot generation.")

def aggregate(run_dir):
  print("  Agregating results for all jobs.")
  desc_file = os.path.join(run_dir, 'description.txt')
  if not os.path.isfile(desc_file):
    msg = f"   Description file '{desc_file}' not found in '{run_dir}'. Looking in parent directory."
    desc_file = os.path.join(run_dir, '..', 'description.txt')
    if not os.path.isfile(desc_file):
      msg = f"   Description file '{desc_file}' not found in '{run_dir}' or its parent directory."
      raise FileNotFoundError(msg)

  with open(desc_file, 'r') as f:
    desc_text = f.read().strip()
  if desc_text:
    desc_text = f"Description: {desc_text}"
  else:
    desc_text = "Description: (empty)"

  md = desc_text
  for subdir in sorted(os.listdir(run_dir)):
    subdir_path = os.path.join(run_dir, subdir)
    if os.path.isdir(subdir_path):
      lno_file = os.path.join(subdir_path, 'lno.md')
      if os.path.isfile(lno_file):
        with open(lno_file, 'r') as f:
          lno_content = f.read()
        md += f"\n\n# {subdir}\n\n#{lno_content}"

  print(f"    {desc_text}")

  with open(os.path.join(run_dir, 'tables.md'), 'w') as f:
    f.write(md)
    print(f"    Wrote {os.path.join(run_dir, 'tables.md')}")

