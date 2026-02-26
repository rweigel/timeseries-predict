import timeseries_predict as tsp

if False:
  # Update summary, plot, stats, and table without running the rest of the code.
  from timeseries_predict.summary import summary
  summary("cluster1", results_dir="./output/results-1")
  exit()

conf_file = tsp.cli()['config']
conf = tsp.read_conf(conf_file)

job_dfs, job_confs = tsp.job_list_function(conf['job_script'])(conf)

def job(combined_dfs, conf):
  import traceback
  try:
    tsp.train_and_test(combined_dfs, conf)
  except Exception as e:
    import os
    error_fname = os.path.join(conf['results_dir'], conf['tag'], "main.error.txt")
    with open(error_fname, "a") as f:
      emsg = f"Error in job {conf['tag']}: {str(e)}\n"
      print(emsg)
      traceback.print_exc()
      print(f"Error written to {error_fname}")
      f.write(emsg)


def job_wrapper(args):
  return job(args[0], args[1])


if __name__ == "__main__":

  if not conf.get('parallel_jobs', False):
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