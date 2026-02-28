def run(combined_dfs, conf):
  import timeseries_predict as tsp
  import traceback
  try:
    tsp.train_and_test(combined_dfs, conf)
  except Exception as e:
    import os
    tb = traceback.format_exc()
    try:
      error_fname = os.path.join(conf['run_dir'], conf['job'], "main.error.txt")
      emsg = f"Error in job {conf['job']}: {str(e)}\n"
      print(emsg)
      print(tb)
      with open(error_fname, "a") as f:
        f.write(emsg)
        f.write(tb)
      print(f"Error written to {error_fname}")
    except Exception as write_err:
      print(f"Error in job (could not write error file: {write_err}):\n{tb}")


def job_list(conf):
  print(f"Getting job list from {conf['job_script']}")
  return job_function(conf['job_script'])(conf)


def job_function(job_script):
  import os
  import importlib.util
  job_script = os.path.expanduser(job_script)
  spec = importlib.util.spec_from_file_location("job_list", job_script)
  mod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mod)
  return mod.job_list


def check(job_list):
  for idx, (job_dfs, job_conf) in enumerate(job_list):
    print(f"Job {idx+1}: name: {job_conf['job']}, Number of segments: {len(job_dfs)}")

    msg = f"Job '{job_conf['job']}' configuration is missing"
    if 'inputs' not in job_conf:
      raise ValueError(f"{msg} 'inputs' in the job configuration.")
    if 'outputs' not in job_conf:
      raise ValueError(f"{msg} 'outputs' in the job configuration.")

    # Validate that each job_df has required columns
    required_cols = set(job_conf['inputs'] + job_conf['outputs'] + ['datetime'])
    for df in job_dfs:
      missing_cols = required_cols - set(df.columns)
      if missing_cols:
        msg = f"Job '{job_conf['job']}' is missing columns {missing_cols}. "
        msg += "Check configuration or job script."
        raise ValueError(msg)
