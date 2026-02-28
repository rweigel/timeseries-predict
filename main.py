import timeseries_predict as tsp

args = tsp.cli()

if args['postprocess'] is not None:
  import sys
  # Update summary, plot, stats, and table without running the rest of the code.
  tsp.summary(args['postprocess'], job=args['job'])
  sys.exit(0)

print(f"Reading: {args['config']}")
conf = tsp.read_conf(args['config'])

print(f"Getting job list by calling {conf['job_script']}:job_list(conf)")

if __name__ == "__main__":

  job_list = tsp.job.job_list(conf)

  tsp.job.check(job_list)

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
