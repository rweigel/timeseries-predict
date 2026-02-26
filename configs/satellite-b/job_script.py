def job_list(conf):
  job_confs = [] # One config for each satellite
  job_dfs = []
  for satellite in conf['data']['satellites']:
    job_conf = conf.copy()
    job_conf['tag'] = satellite
    job_conf['data']['file_pattern'] = f"{satellite}*.pkl"
    job_confs.append(job_conf)
    job_dfs.append(job_data(**job_conf['data']))

  return job_dfs, job_confs


def job_data(**config):
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
