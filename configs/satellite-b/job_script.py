def job_list(conf):
  """
  Return a list of (job_dfs, job_conf) tuples, where job_dfs is a list of
  DataFrames for the job and job_conf is the job configuration dictionary.

  Each dataframe must have a column 'datetime' with datetime values.
  Each dataframe must of columns for the input and output variables
  specified in the job configuration.
  """
  jobs = []

  if conf['data']['satellites'] is None:
    conf['data']['satellites'] = satellite_list(conf['data']['data_directory'])
    print(f"  Found satellites: {conf['data']['satellites']}")

  if isinstance(conf['data']['satellites'], str):
    conf['data']['satellites'] = [conf['data']['satellites']]

  for satellite in conf['data']['satellites']:
    job_conf = conf.copy()
    job_conf['job'] = satellite
    job_conf['data']['file_pattern'] = f"{satellite}*.pkl"
    config = {
      **job_conf['data'],
      "inputs": job_conf['inputs'],
      "outputs": job_conf['outputs']
    }
    job_dfs = job_data(**config)
    jobs.append((job_dfs, job_conf))

  return jobs

def satellite_list(data_directory):
  import os
  import glob
  fglob = os.path.join(data_directory, "*.pkl")
  files = sorted(glob.glob(fglob))
  satellites = set()
  for f in files:
    basename = os.path.basename(f)
    satellite = basename.split('_')[0]
    satellites.add(satellite)

  return list(satellites)


def job_data(**config):
  import os
  import glob
  import numpy as np
  import pandas as pd

  print("  job_data(): Reading data files")

  fglob = os.path.join(config['data_directory'], config['file_pattern'])
  files = sorted(glob.glob(fglob))
  if not files:
    print(f"Error: No files found matching '{fglob}'. Exiting.")
    exit(1)

  dataframes = []
  n_r = 0 # Number of DataFrames read
  for f in files:
    print(f"    Reading: {f}")
    df = pd.read_pickle(f) # Load the DataFrame from pickle
    n_r = n_r + 1

    n_before = len(df)
    cols_to_check = list(set(config.get('inputs', []) + config.get('outputs', [])))
    cols_to_check = [c for c in cols_to_check if c in df.columns]
    if cols_to_check:
      df = df.dropna(subset=cols_to_check)
    n_dropped = n_before - len(df)
    if n_dropped > 0:
      print(f"    Dropped {n_dropped}/{n_before} rows with NaN/NaT values in: {f}")

    dataframes.append(df)

    if config['n_years_max'] is not None and n_r == config['n_years_max']:
      # Break if n_years_max (equal to number of DataFrames to return)
      # is specified b/c one file per year.
      break

  if not config['loo']:
    dataframes = [pd.concat(dataframes, ignore_index=True)]

  return dataframes
