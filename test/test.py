import os
import subprocess

base_cmd = ["python", "run.py"]

args_list = [
  ["--config", "configs/demo/demo-0.yaml"],
  ["--postprocess", "data/results/demo/demo-0"],
  ["--postprocess", "data/results/demo/demo-0/noise-multiplier-0"],
  ["--postprocess", "data/results/demo/demo-0", "--job", "noise-multiplier-0"],

  ["--config", "configs/satellite-b/parallel-test.yaml"],
  ["--postprocess", "data/results/satellite-b/parallel-test"],
  ["--postprocess", "data/results/satellite-b/parallel-test/cluster1"],
  ["--postprocess", "data/results/satellite-b/parallel-test", "--job", "cluster1"],

  ["--config", "configs/satellite-b/serial-test.yaml"],
  ["--postprocess", "data/results/satellite-b/serial-test"],
  ["--postprocess", "data/results/satellite-b/serial-test/cluster1"],
  ["--postprocess", "data/results/satellite-b/serial-test", "--job", "cluster1"]
]

def _print(msg, log, end=''):
  print(msg)
  log.write(msg)
  log.flush()

log_path = os.path.join(os.path.dirname(__file__), 'test.log')
kwargs = {
  "stdout": subprocess.PIPE,
  "stderr": subprocess.STDOUT,
  "text": True,
  "bufsize": 1
}
with open('test.log', 'w') as log:
  for args in args_list:
    cmd = base_cmd + args
    _print(f"\n***\nRunning command: {' '.join(cmd)}\n***\n", log, end='\n')
    with subprocess.Popen(cmd, **kwargs) as proc:
      for line in proc.stdout:
        _print(line, log)
