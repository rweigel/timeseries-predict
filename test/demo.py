import os
import subprocess

import utilrsw

base_cmd = ["python", "run.py"]

args_list = [
  "configs/demo/demo-0.yaml",
  "data/results/demo/demo-0",
  "data/results/demo/demo-0/noise-multiplier-0",

  "configs/demo/demo-1.yaml",
  "data/results/demo/demo-1",
  "data/results/demo/demo-1/noise-multiplier-0",
  "data/results/demo/demo-1/noise-multiplier-1",
  "data/results/demo/demo-1/noise-multiplier-2",

  "configs/demo/demo-2.yaml",
  "data/results/demo/demo-2",
  "data/results/demo/demo-2/noise-multiplier-0",
  "data/results/demo/demo-2/noise-multiplier-1",
  "data/results/demo/demo-2/noise-multiplier-2",

  "configs/satellite-b/test/serial-test.yaml",
  "data/results/satellite-b/test/serial-test",
  "data/results/satellite-b/test/serial-test/cluster1",

  "configs/satellite-b/test/parallel-test.yaml",
  "data/results/satellite-b/test/parallel-test",
  "data/results/satellite-b/test/parallel-test/cluster1"
]

def _print(msg, log):
  print(msg, end='')
  log.write(msg)
  log.flush()

log_file = os.path.join(os.path.dirname(__file__), 'demo.log')

kwargs = {
  "stdout": subprocess.PIPE,
  "stderr": subprocess.STDOUT,
  "text": True,
  "bufsize": 1
}

with open(log_file, 'w') as log:
  for arg in args_list:
    cmd = base_cmd + [arg]
    hline = "\n" + utilrsw.hline(char="*", display=False) + "\n"
    _print(hline, log)
    _print(f"Running command: {' '.join(cmd)}", log)
    _print(hline, log)
    with subprocess.Popen(cmd, **kwargs) as proc:
      for line in proc.stdout:
        _print(line, log)
