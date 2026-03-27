import sys

from numpy import average
import os


def job_list(conf):
  """
  Return a list of (job_dfs, job_conf) tuples, where job_dfs is a list of
  DataFrames for the job and job_conf is the job configuration dictionary.

  Each dataframe must have a column 'datetime' with datetime values.
  Each dataframe must of columns for the input and output variables
  specified in the job configuration.
  """
  jobs = []
  for jidx in range(len(conf['data']['sites'])):
    job_conf = conf.copy()
    job_site = conf['data']['sites'][jidx]
    job_conf['job'] = f"gic_vs_b_{job_site}"
    job_df = job_data(job_site,**job_conf['data'])
    jobs.append((job_df, job_conf))

  return jobs

def job_data(site, average, **config):
  """
  Returns a data frame with averaged time series with a column named 'datetime'
  with datetime values and columns for the input and output variables. The input
  variable is named input1 and the output variables are named output1, output2, 
  and output3.

  input1 is measured GIC.
  output1 is Bx.
  output2 is By.
  output3 is Bz.
  """

  import numpy as np
  import pandas as pd

  # read in csv
  csv_name = config['data_directory'] + f"/{site}/{site}_gic_b.csv"
  if not os.path.exists(csv_name): # reading in files if data_prep not run yet
    job_script_dir = os.path.dirname(__file__)
    sys.path.append(job_script_dir)
    print(f"CSV file not found: {csv_name}. Running data prep script.")
    import swerve_data_prep
  data_csv = pd.read_csv(csv_name)

  # Create a time series with a datetime column and input/output columns
  df = pd.DataFrame({
    'datetime': pd.to_datetime(data_csv['datetime']),
    'bx': data_csv['bx'].values,
    'by': data_csv['by'].values,
    'bz': data_csv['bz'].values,
    'gic': data_csv['gic'].values
  })
  df.set_index('datetime', inplace=True)
  df = df.resample(f'{average}min').mean()
  df = df.reset_index().to_dict('list')

  return [pd.DataFrame(df)]

if __name__ == "__main__":
  # Test the job_list function
  import yaml

  conf = yaml.safe_load(open("./configs/swerve/swerve.yaml"))
  job_list = job_list(conf)

  for jidx, (job_dfs, job_conf) in enumerate(job_list):
    print(f"Job {jidx+1}: name: {job_conf['job']}, Number of DataFrames: {len(job_dfs)}")
    for sidx, df in enumerate(job_dfs):
      print(f"  Segment {sidx+1}/{len(job_dfs)} first 5 rows:")
      msg = df.head().to_string(index=False)
      print(f"    {msg.replace(chr(10), chr(10)+'    ')}")
