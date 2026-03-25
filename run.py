import os
import sys

import utilrsw
import timeseries_predict as tsp

arg = tsp.cli()

if os.path.isdir(arg):
  # Update summary, stats, plot, and table without running the rest of the code.
  tsp.summary(arg)
  tsp.aggregate(arg)
  sys.exit(0)

conf = tsp.config(arg)

utilrsw.hline()
print(f"{arg}:")
utilrsw.print_dict(conf, indent=2)
utilrsw.hline()

if __name__ == "__main__":

  print(f"Getting job list by calling {conf['job_script']}:job_list(conf)")
  job_list = tsp.job.job_list(conf)

  tsp.job.check(job_list)

  for idx, (job_dfs, job_conf) in enumerate(job_list):
    if 'job' not in job_conf or job_conf['job'] is None:
      job_conf['job'] = f"job_{idx+1}"

  if not conf.get('parallel_jobs', False):
    for idx, (job_dfs, job_conf) in enumerate(job_list):
      tsp.job.run(job_dfs, job_conf)
  else:
    import multiprocessing
    n_cpu_available = multiprocessing.cpu_count()
    print(f"# CPUs available for parallel processing: {n_cpu_available}")
    n_cpu_needed = len(job_list)
    # Keep one CPU free for system processes
    if n_cpu_needed >= n_cpu_available:
      n_cpu = n_cpu_available - 1
    else:
      n_cpu = n_cpu_needed
    print(f"Using {n_cpu} CPUs for parallel processing {n_cpu_needed} jobs")
    with multiprocessing.Pool(n_cpu) as p:
      p.starmap(tsp.job.run, job_list)

  #tsp.summary(conf['run_dir'])
  tsp.aggregate(conf['run_dir'])
