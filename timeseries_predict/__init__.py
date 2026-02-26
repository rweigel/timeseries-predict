from .train_and_test import train_and_test as train_and_test

def cli():
  import argparse
  parser = argparse.ArgumentParser(description='Run time series prediction jobs.')
  parser.add_argument('--config', type=str, default="./configs/satellite-test-serial.yaml",
                      help='Path to the YAML configuration file.')
  args = parser.parse_args()
  return vars(args)

def read_conf(conf_file):
  import yaml
  from collections.abc import Mapping

  def deep_merge(base, override):
    for k, v in override.items():
      if isinstance(v, Mapping) and isinstance(base.get(k), Mapping):
        deep_merge(base[k], v)
      else:
        base[k] = v

  with open(conf_file) as f:
    conf = yaml.safe_load(f)

  if 'base_config' in conf:
    with open(conf['base_config']) as f:
      base_conf = yaml.safe_load(f)
    deep_merge(base_conf, conf)
    conf = base_conf

  return conf

def job_list_function(job_script):
  import os
  import importlib.util
  job_script = os.path.expanduser(job_script)
  spec = importlib.util.spec_from_file_location("job_list", job_script)
  mod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mod)
  return mod.job_list


