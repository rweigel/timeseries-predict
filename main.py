import timeseries_predict as tsp

args = tsp.cli()

if args['postprocess'] is not None:
  # Update summary, plot, stats, and table without running the rest of the code.
  tsp.summary(args['postprocess'], job=args['job'])
  exit(0)

print(f"Reading: {args['config']}")
conf = tsp.read_conf(args['config'])

print(f"Getting job list from {conf['job_script']}")
job_list = tsp.job_list_function(conf['job_script'])(conf)

def check_job_list(job_list):
  for idx, (job_dfs, job_conf) in enumerate(job_list):
    print(f"Job {idx+1}: name: {job_conf['job']}, Number of segments: {len(job_dfs)}")

    msg = f"Job '{job_conf['job']}' configuration is missing"
    if 'inputs' not in job_conf:
      raise ValueError(f"{msg} 'inputs' in the job configuration.")
    if 'outputs' not in job_conf:
      raise ValueError(f"{msg} 'outputs' in the job configuration.")

    # Validate that each job_df has required columns
    required_cols = set(job_conf['inputs'] + job_conf['outputs'] + ['timestamp'])
    for df in job_dfs:
      missing_cols = required_cols - set(df.columns)
      if missing_cols:
        msg = f"Job '{job_conf['job']}' is missing columns {missing_cols}. "
        msg += "Check configuration or {conf['job_script']}."
        raise ValueError(msg)


def job(combined_dfs, conf):
  import traceback
  try:
    tsp.train_and_test(combined_dfs, conf)
  except Exception as e:
    import os
    error_fname = os.path.join(conf['run_dir'], conf['job'], "main.error.txt")
    with open(error_fname, "a") as f:
      emsg = f"Error in job {conf['job']}: {str(e)}\n"
      print(emsg)
      traceback.print_exc()
      print(f"Error written to {error_fname}")
      f.write(emsg)


def job_wrapper(args):
  return job(args[0], args[1])


if __name__ == "__main__":

  check_job_list(job_list)

  if not conf.get('parallel_jobs', False):
    for idx, (job_dfs, job_conf) in enumerate(job_list):
      job(job_dfs, job_conf)
  else:
    import multiprocessing
    n_cpu_available = multiprocessing.cpu_count()
    print(f"# CPUs available for parallel processing: {n_cpu_available}")
    n_cpu_needed = len(job_list)
    if n_cpu_needed >= n_cpu_available:
      n_cpu = n_cpu_available - 1
    else:
      n_cpu = n_cpu_needed
    print(f"Using {n_cpu} CPUs for parallel processing {n_cpu_needed} jobs")
    with multiprocessing.Pool(n_cpu) as p:
      p.map(job_wrapper, job_list)
