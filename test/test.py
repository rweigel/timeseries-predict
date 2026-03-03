import os
import subprocess

base_cmd = ["python", "run.py"]

args_list = [
  ["configs/demo/demo-0.yaml"],
  ["data/results/demo/demo-0"],
  ["data/results/demo/demo-0/noise-multiplier-0"],

  ["configs/satellite-b/parallel-test.yaml"],
  ["data/results/satellite-b/parallel-test"],
  ["data/results/satellite-b/parallel-test/cluster1"],

  ["configs/satellite-b/serial-test.yaml"],
  ["data/results/satellite-b/serial-test"],
  ["data/results/satellite-b/serial-test/cluster1"],
  ["data/results/satellite-b/serial-test"]
]

def _print(msg, log):
  print(msg, end='')
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
    _print(f"\n***\nRunning command: {' '.join(cmd)}\n***\n\n", log)
    with subprocess.Popen(cmd, **kwargs) as proc:
      for line in proc.stdout:
        _print(line, log)
