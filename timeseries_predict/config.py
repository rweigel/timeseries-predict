def config(conf_file):
  import os
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

  known_models = [
    'ols',
    'nn_miso',
    'nn_mimo',
    'nn_miso_resid',
    'nn_mimo_resid'
  ]

  defaults = {
    'lr': 0.001,
    'run_desc': '',
    'removed_inputs': None,
    'models': known_models,
    'train_fraction': 0.8,
    'device': None,
    'num_epochs': 200,
    'batch_size': 256,
    'n_reps': 1,
    'lags': None
  }

  if 'base_dir' not in conf or conf['base_dir'] is None:
    conf['base_dir'] = './results'

  if 'run_dir' not in conf or conf['run_dir'] is None:
    conf['run_dir'] = os.path.splitext(os.path.basename(conf_file))[0]

  conf['run_dir'] = os.path.join(conf['base_dir'], conf['run_dir'])

  for key in defaults.keys():
    if key not in conf or conf[key] is None:
      conf[key] = defaults[key]

  if conf['lags'] is not None and isinstance(conf['lags'], dict):
    for key, value in conf['lags'].items():
      if not isinstance(value, int):
        raise ValueError(f"Lag value for '{key}' must be an integer, got {type(value)}")
      if key not in conf['inputs']:
        raise ValueError(f"Lag key '{key}' must be in inputs, got {conf['inputs']}")

  for key in ['inputs', 'outputs', 'models']:
    if isinstance(conf[key], str):
      conf[key] = [key]

  for model in conf['models']:
    if model not in known_models:
      raise ValueError(f"Model '{model}' not in list of known models: {known_models}")

  if isinstance(conf['inputs'], str):
    conf['inputs'] = [conf['inputs']]
  if isinstance(conf['outputs'], str):
    conf['outputs'] = [conf['outputs']]

  removed_inputs = conf['removed_inputs']
  if removed_inputs is None:
    removed_inputs = [None]
  if removed_inputs is True:
    removed_inputs = [None] + conf['inputs']
  if removed_inputs is False:
    removed_inputs = [None]
  conf['removed_inputs'] = removed_inputs

  return conf

