from .train_and_test import train_and_test as train_and_test
from .summary import summary as summary
from . import job

def cli():
  import argparse

  description = 'Run time series prediction jobs.'
  args = {
    'config': {
      'type': str,
      'default': "./configs/satellite-test-serial.yaml",
      'help': 'Path to the YAML run configuration file.'
    },
    'postprocess': {
      'type': str,
      'default': None,
      'help': 'Postprocess results starting at this results directory.'
    },
    'job': {
      'type': str,
      'default': None,
      'help': 'Name of the job to run or postprocess. Default is to run or postprocess all jobs.'
    }
  }

  parser = argparse.ArgumentParser(description=description)
  group = parser.add_mutually_exclusive_group(required=True)
  group.add_argument('--config', **args['config'])
  group.add_argument('--postprocess', **args['postprocess'])
  parser.add_argument('--job', **args['job'])
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
