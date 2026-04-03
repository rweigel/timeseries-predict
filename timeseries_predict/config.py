def config(conf_file):
  import os
  import yaml
  import copy

  import utilrsw

  def deep_merge(base, override):
    for k, v in override.items():
      if isinstance(v, dict) and isinstance(base.get(k), dict):
        deep_merge(base[k], v)
      else:
        base[k] = v

  with open(conf_file) as f:
    conf = yaml.safe_load(f)

  utilrsw.hline()
  print(f"Loaded configuration from {conf_file}:")
  utilrsw.print_dict(conf, indent=2)
  utilrsw.hline()

  if 'base_config' in conf:
    with open(conf['base_config']) as f:
      base_conf = yaml.safe_load(f)

    utilrsw.hline()
    print(f"Loaded base_config from {conf['base_config']}:")
    utilrsw.print_dict(base_conf, indent=2)
    utilrsw.hline()

    deep_merge(base_conf, conf)
    conf = base_conf

    utilrsw.hline()
    print("Merged configuration:")
    utilrsw.print_dict(conf, indent=2)
    utilrsw.hline()

  known_models = [
    'ols',
    'nn_miso',
    'nn_mimo',
    'nn_miso_resid',
    'nn_mimo_resid'
  ]

  defaults = {
    'run_desc': '',
    'removed_inputs': None,
    'models': known_models,
    'train_fraction': 0.8,
    'n_epochs': 200,
    'n_reps': 1,
    'lags': None,
    'nn': {
      'device': None,
      'dtype': 'float32',
      'nn_class': 'NeuralNetworkOneLayer',
      'hidden_size': 32,
      'activation': 'Tanh',
      'optimizer': 'Adam',
      'optimizer_kwargs': {
        'lr': 0.001
      }
    }
  }

  if 'base_dir' not in conf or conf['base_dir'] is None:
    conf['base_dir'] = './results'

  if 'run_dir' not in conf or conf['run_dir'] is None:
    conf['run_dir'] = os.path.splitext(os.path.basename(conf_file))[0]

  conf['run_dir'] = os.path.join(conf['base_dir'], conf['run_dir'])

  conf_merged = copy.deepcopy(defaults)
  deep_merge(conf_merged, conf)
  conf = conf_merged

  utilrsw.hline()
  print("Configuration after merging with defaults:")
  utilrsw.print_dict(conf, indent=2)
  utilrsw.hline()

  if conf['lags'] is not None:
    if isinstance(conf['lags'], dict):
      for key, value in conf['lags'].items():
        if not isinstance(value, int):
          raise ValueError(f"Lag value for '{key}' must be an integer, got {type(value)}")
        if key not in conf['inputs']:
          raise ValueError(f"Lag key '{key}' must be in inputs, got {conf['inputs']}")
    elif isinstance(conf['lags'], int):
      if conf['lags'] < 0:
        raise ValueError(f"Lags must be >= 0, got {conf['lags']}")
      if conf['lags'] == 0:
        conf['lags'] = None
    else:
      example = "{conf['inputs'][0]: 3, ...}"
      msg = f"Lags must be an integer or a dictionary of input names to max lag values, e.g. {example}, got {conf['lags']}"
      raise ValueError(msg)

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

  utilrsw.hline()
  print("Final configuration after normalizing inputs, outputs, models, and removed_inputs:")
  utilrsw.print_dict(conf, indent=2)
  utilrsw.hline()

  return conf

