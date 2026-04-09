import os
import sys
job_script_dir = os.path.dirname(__file__)
sys.path.append(job_script_dir)

def job_list(conf):
  """
  Return a list of (job_dfs, job_conf) tuples, where job_dfs is a list of
  DataFrames for the job and job_conf is the job configuration dictionary.

  Each dataframe must have a column 'datetime' with datetime values.
  Each dataframe must of columns for the input and output variables
  specified in the job configuration.
  """
  sites = conf['data'].get('sites', None)
  if sites is None:
    from data_prep import get_ids
    sites = get_ids(conf['data']['event'])

  jobs = []
  for job_site in sites:
    job_conf = conf.copy()
    job_conf['job'] = f"gic_vs_b_{job_site}"
    job_df = job_data(job_site, job_conf)
    if job_df is None:
      print(f"  Skipping job for site {job_site} due to NaN values in data after interpolation.")
    else:
      jobs.append(([job_df], job_conf))

  return jobs


def job_data(site, config):

  from data_prep import swerve_data_download

  # Get file paths for GIC and B data for `site``, downloading them if needed
  gic_file, b_file = swerve_data_download(config['data']['event'], site)
  gic, b = read_gic_b(gic_file, b_file)

  # Note that in combine_gic_b, config['inputs'] is modified to use actual columns names.
  site_df = combine_gic_b(gic, b, config)
  site_df = nan_interpolate(site_df, limit=3)

  return site_df

def nan_interpolate(df, limit=None):
  # Interpolate NaN values in specified columns using linear interpolation
  columns = df.columns.difference(['datetime'])
  for col in columns:
    if col in df.columns:
      # Interpolate NaN values with limit on number of consecutive NaNs to fill.
      df[col] = df[col].interpolate(method='linear', limit=limit, limit_direction='both')

  if df[columns].isna().sum().sum() > 0:
    # If there are still NaN values after interpolation, return None.
    return None

  return df

def read_gic_b(gic_file, b_file):
  import pandas as pd

  # Load and extract data

  wmsg = "Warning: More than one source found in"
  gic_data = pd.read_pickle(gic_file)['GIC']['measured']
  sources = list(gic_data.keys())
  if len(sources) > 1:
    print(f"  {wmsg} GIC data: {sources}. Using the first one: {sources[0]}")
  gic = gic_data[sources[0]]['modified']

  b_data = pd.read_pickle(b_file)['B']['measured']
  sources = list(b_data.keys())
  if len(sources) > 1:
    print(f"  {wmsg} B data: {sources}. Using the first one: {sources[0]}")
  b = b_data[sources[0]]['modified']

  return gic, b


def combine_gic_b(gic, b, config):
  import pandas as pd

  inputs = config['inputs']
  average = config['data']['average']

  gic_df = pd.DataFrame(gic['data'][:, 0], columns=[gic['labels'][0]], index=gic['time'])

  # B data can have column labels of Bx, By, Bz or B_N, B_E, B_v. Here we
  # translate B1 => first column, B2 => second column, etc.
  col_idxs = []
  col_names = []
  col_names = []
  for input in inputs:
    # Extract integer from input like 'B1' => column 0
    col_idx = int(input[1]) - 1
    col_idxs.append(col_idx)
    col_names.append(b['labels'][col_idx])

  config['inputs'] = col_names

  b_df = pd.DataFrame(b['data'][:, col_idxs], columns=col_names, index=b['time'])

  if average is not None:
    print(f"  Averaging data to {average} frequency")
    gic_df = gic_df.resample(average).mean()
    b_df = b_df.resample(average).mean()

  # 'inner' join to keep only timestamps present in both datasets
  site_df = gic_df.join(b_df, how='inner').reset_index(names='datetime')

  print(f"  Number of timestamps in GIC: {len(gic_df)}")
  print(f"  Number of timestamps in B:   {len(b_df)}")
  print(f"  Number of common timestamps: {len(site_df)}")

  return site_df
