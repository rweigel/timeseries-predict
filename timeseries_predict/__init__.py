from .train_and_test import train_and_test as train_and_test
from .summary import summary as summary
from . import job

def cli():
  import argparse

  description = 'Run time series prediction jobs.'
  parser = argparse.ArgumentParser(description=description, usage='run.py DIR | CONFIG_FILE')
  parser.add_argument('target', help='Results [sub]directory to postprocess, or YAML config file to run.')

  target = parser.parse_args().target

  return target


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
