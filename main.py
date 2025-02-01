test_run = False # For debugging code
parallel_jobs = True # Do jobs in parallel

conf = {
    "data": { # Passed to user-defined _data_load function
      "n_df": None, # Num. of DataFrames for data_load to return. None => all DataFrames.
      "data_directory": "./data",
      "satellites": [
        'cluster1',
        'cluster2',
        'cluster3',
        'cluster4',
        'geotail',
        'goes8',
        'goes9',
        'goes10',
        'goes12',
        'imp8',
        'polar',
        'rbspa',
        'rbspa',
        'themisa',
        'themisb',
        'themisc',
        'themisd',
        'themise'
      ]
    },

    "tag": None,
    "results_dir": None,

    "device": None,   # Parallelize using CPU or GPU in PyTorch. (Not implemented.)

    "num_epochs": 1,
    "num_boot_reps": 1,
    "batch_size": 256,
    "hidden_size": 32,
    "activation": "Tanh",
    "optimizer": "Adam",
    "optimizer_kwargs": {"lr": 0.01},
    "lr": 0.001,

    # True => [None, **outputs]; None or no attribute => # Only use all inputs
    "removed_inputs": True,

    "models": ["ols", "nn_mimo", "nn_miso"],

    "inputs": ["r", "theta", "phi", "imfby", "imfbz", "vsw", "nsw", "ey", "ey_avg"],
    "outputs": ["bx", "by", "bz"],
}

if test_run:
  conf['data']['satellites'] = conf['data']['satellites'][0:2]
  conf['data']['n_df'] = 2
  conf['num_epochs'] = 3
  conf['num_boot_reps'] = 1
  conf['removed_inputs'] = [None, "r"]

if False:
  # If only summary code modified
  from satellite_predict.summary import summary
  summary("cluster1", results_dir=conf['results_dir'])
  exit()

def _data_load(**config):
  import os
  import glob
  import numpy as np
  import pandas as pd

  def rename_columns(df):
    # Rename columns so no special characters (avoids issues with directory names)
    for column in df.columns:
      column_orig = column
      if '[' in column:
        column = column.split('[')[0]

      replacements = {'<': '', '>': '_avg', '*': 'star'}
      for char, replacement in replacements.items():
        if char in column:
          column = column.replace(char, replacement)

      df.rename(columns={column_orig: column}, inplace=True)

  def appendSpherical_np(xyz):
    xy = xyz[:, 0]**2 + xyz[:, 1]**2
    r = np.sqrt(xy + xyz[:, 2]**2)
    theta = np.arctan2(xyz[:, 2], np.sqrt(xy)) * 180 / np.pi
    phi = np.arctan2(xyz[:, 1], xyz[:, 0])
    phi = np.where(phi < 0, phi + 2 * np.pi, phi) * 180 / np.pi
    return np.column_stack((r, theta, phi))

  # Labels for Cartesian position columns
  position_cart = ["x[km]", "y[km]", "z[km]"]

  # Labels for derived columns used to convert from Cartesian to spherical
  position_sph = ["r", "theta", "phi"]

  fglob = os.path.join(config['data_directory'], config['file_pattern'])
  files = sorted(glob.glob(fglob))

  dataframes = []
  n_r = 0 # Number of DataFrames read
  for f in files:
    df = pd.read_pickle(f) # Load the DataFrame from pickle
    n_r = n_r + 1
    cartesian = df[position_cart].to_numpy()
    spherical = appendSpherical_np(cartesian)

    # Add spherical coordinates to the DataFrame
    for i, col in enumerate(position_sph):
      df[col] = spherical[:, i]

    ymdhms = df[['year', 'month', 'day', 'hour', 'minute', 'second']]
    df['datetime'] = pd.to_datetime(ymdhms)

    rename_columns(df)
    dataframes.append(df)

    if config['n_df'] is not None and n_r == config['n_df']:
      # Break if n_df (number of DataFrames to return) is specified
      break

  return dataframes

def _job_list(conf, data_load_func):
  job_confs = [] # One config for each satellite
  job_dfs = []
  for satellite in conf['data']['satellites']:
    job_conf = conf.copy()
    job_conf['tag'] = satellite
    job_conf['data']['file_pattern'] = f"{satellite}*.pkl"
    job_confs.append(job_conf)
    job_dfs.append(data_load_func(**job_conf['data']))
  return job_dfs, job_confs

if "_job_list" in globals():
  job_dfs, job_confs = _job_list(conf, _data_load)
else:
  job_dfs = _data_load(**conf['data'])
  job_confs = [conf]

from satellite_predict.train_and_test import train_and_test

def job(combined_dfs, conf):
  try:
    train_and_test(combined_dfs, conf)
  except Exception as e:
    import os
    error_fname = os.path.join(conf['results_dir'], conf['tag'], "main.error.txt")
    with open(error_fname, "a") as f:
      f.write(f"Error in job with config {conf['tag']}:\n{str(e)}\n")

def job_wrapper(args):
  return job(args[0], args[1])

if __name__ == "__main__":

  if not parallel_jobs:
    for idx, conf in enumerate(job_confs):
      job(job_dfs[idx], conf)
  else:
    import multiprocessing
    n_cpu_available = multiprocessing.cpu_count()
    print(f"# CPUs available for parallel processing: {n_cpu_available}")
    n_cpu_needed = len(job_confs)
    if n_cpu_needed >= n_cpu_available:
      n_cpu = n_cpu_available - 1
    else:
      n_cpu = n_cpu_needed
    print(f"Using {n_cpu} CPUs for parallel processing {n_cpu_needed} jobs")
    with multiprocessing.Pool(n_cpu) as p:
      p.map(job_wrapper, list(zip(job_dfs, job_confs)))
      # list(zip(job_dfs, job_confs)) contains
      # [(job_confs[0], job_confs[0]), (job_confs[1], job_confs[1]), ...]
      # which is passed to job_wrapper