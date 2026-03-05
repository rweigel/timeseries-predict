def cli():
  import argparse

  description = 'Run time series prediction jobs.'
  parser = argparse.ArgumentParser(description=description, usage='run.py DIR | CONFIG_FILE')
  parser.add_argument('target', help='Results [sub]directory to postprocess, or YAML config file to run.')

  target = parser.parse_args().target

  return target
