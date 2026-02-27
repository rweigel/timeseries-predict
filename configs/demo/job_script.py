def job_list(conf):
  """
  Return a list of (job_dfs, job_conf) tuples, where job_dfs is a list of
  DataFrames for the job and job_conf is the job configuration dictionary.

  Each dataframe must have a column 'datetime' with datetime values.
  Each dataframe must of columns for the input and output variables
  specified in the job configuration.
  """
  jobs = []
  for jidx in range(3):
    job_conf = conf.copy()
    job_conf['job'] = f"Noise multiplier: {jidx}"
    job_df = job_data(jidx, **job_conf['data'])
    jobs.append((job_df, job_conf))

  return jobs

def job_data(noise_multiplier, **config):
  """
  Returns a data frame with a 1-second time series with a column named 'datetime'
  with datetime values and columns for the input and output variables. The input
  variable is named input1 and the output variable is named output1.

  input1 is a unit amplitude sine wave.
  output1 is input1 plus Gaussian noise with standard deviation specified by
  config['noise_level'].
  """

  import numpy as np
  import pandas as pd

  noise_amplitude = noise_multiplier*config['noise_amplitude']
  freq = config['cadence']

  # Period is 60 times the time resolution in seconds.
  period = 60*pd.to_timedelta(freq).total_seconds()
  start = config['start']
  end = config['end']

  # Create a time series with a datetime column and input/output columns
  time_index = pd.date_range(start=start, end=end, freq=freq, inclusive='left')

  # Sine wave with period of 1 second
  input1 = np.sin(2 * np.pi * time_index.astype(np.int64) / period) 

  # Add Gaussian noise
  output1 = input1 + np.random.normal(scale=noise_amplitude, size=len(input1))

  df = {
    'datetime': time_index,
    'input1': input1,
    'output1': output1
  }

  return [pd.DataFrame(df)]

if __name__ == "__main__":
  # Test the job_list function
  import yaml

  conf = yaml.safe_load(open("demo-0.yaml"))
  job_list = job_list(conf)

  for jidx, (job_dfs, job_conf) in enumerate(job_list):
    print(f"Job {jidx+1}: name: {job_conf['job']}, Number of DataFrames: {len(job_dfs)}")
    for sidx, df in enumerate(job_dfs):
      print(f"  Segment {sidx+1}/{len(job_dfs)} first 5 rows:")
      msg = df.head().to_string(index=False)
      print(f"    {msg.replace(chr(10), chr(10)+'    ')}")
