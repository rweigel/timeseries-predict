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
    job_df = job_data(job_site, **job_conf['data'])
    jobs.append((job_df, job_conf))

  return jobs


def job_data(site, **config):
  import os
  import sys

  job_script_dir = os.path.dirname(__file__)
  sys.path.append(job_script_dir)
  from swerve_data_prep import swerve_data_download

  # Get file paths for GIC and B data for `site``, downloading them if needed
  gic_file, b_file = swerve_data_download(config['event'], site)

  gic, b = extract_gic_b(gic_file, b_file)
  site_df = combine_gic_b(gic, b, config['average'])
  return [site_df]


def extract_gic_b(gic_file, b_file):
  import pandas as pd

  # Load and extract modified data

  gic_data = pd.read_pickle(gic_file)['GIC']['measured']
  sources = list(gic_data.keys())
  if len(sources) > 1:
    print(f"  Warning: More than one source found in GIC data: {sources}. Using the first one: {sources[0]}")
  gic = gic_data[sources[0]]['modified']

  b_data = pd.read_pickle(b_file)['B']['measured']
  sources = list(b_data.keys())
  if len(sources) > 1:
    print(f"  Warning: More than one source found in B data: {sources}. Using the first one: {sources[0]}")
  b = b_data[sources[0]]['modified']

  return gic, b


def combine_gic_b(gic, b, average=None):
  import pandas as pd

  gic_df = pd.DataFrame(gic['data'][:, 0], columns=[gic['labels'][0]], index=gic['time'])
  b_df   = pd.DataFrame(b['data'], columns=b['labels'], index=b['time'])

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

