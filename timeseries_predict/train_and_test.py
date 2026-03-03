def train_and_test(job_dfs, conf):
  import shutil
  import pandas

  from .summary import summary

  conf = _prep_config(conf)

  print(f"{'-'*shutil.get_terminal_size(fallback=(80, 24)).columns}")

  s = "s" if len(job_dfs) != 1 else ""
  print(f"Starting job '{conf['job']}' with {len(job_dfs)} DataFrame{s}")

  if len(job_dfs) == 0:
    raise ValueError(f"len(job_dfs) = 0 for job '{conf['job']}'.")

  num_dfs = len(job_dfs)

  for removed_input in conf['removed_inputs']:
    print(f"  Removed input: {removed_input}")
    for i in range(num_dfs): # Loop over segments
      if len(job_dfs) > 1:
        # For leave-one-out, concatenate training job_dfs excluding the ith
        print(f"    Training with DataFrame {i + 1}/{len(job_dfs)} excluded")
        method_label = "loo"
        train_test_dfs = [df for j, df in enumerate(job_dfs) if j != i]
        train_test_data = pandas.concat(train_test_dfs, ignore_index=True)
        validation_data = job_dfs[i]
      else:
        print("    Training using single DataFrame")
        method_label = "lno"
        train_test_data = job_dfs[0]
        validation_data = job_dfs[0]

      # Loop for bootstrap repetitions
      results = []
      for rep in range(conf['num_boot_reps']):
        print(f"        Bootstrap repetition {rep + 1}/{conf['num_boot_reps']}")
        train_data_rep = train_test_data.sample(frac=conf['train_fraction'], random_state=rep)
        test_data_rep = train_test_data.drop(train_data_rep.index)
        result = _train_and_test_single_rep(train_data_rep, test_data_rep, removed_input=removed_input, **conf)
        results.append(result)

      if len(job_dfs) == 1:
        _save_results(results, conf['job'], removed_input, conf['run_dir'], method_label, None)
      else:
        idx_width = len(str(num_dfs))
        idx_zero_padded = str(i + 1).zfill(idx_width)
        _save_results(results, conf['job'], removed_input, conf['run_dir'], method_label, idx_zero_padded)

  print("\n  Creating tables and plots")

  summary(conf['run_dir'], job=conf['job'])

  print(f"Finished job '{conf['job']}'\n")


def _prep_config(conf):
  import os
  import copy
  import json

  conf = copy.deepcopy(conf)  # avoid mutating the caller's dict

  known_models = ['ols', 'nn_miso', 'nn_mimo', 'nn_miso_resid', 'nn_mimo_resid']

  defaults = {
    'job': 'job1',
    'lr': 0.001,
    'run_desc': '',
    'removed_inputs': None,
    'models': known_models,
    'train_fraction': 0.8,
    'device': None,
    'num_epochs': 200,
    'batch_size': 256,
    'run_dir': './results',
    'num_boot_reps': 1
  }

  for key in defaults.keys():
    if key not in conf or conf[key] is None:
      conf[key] = defaults[key]

  for key in ['inputs', 'outputs', 'models']:
    if isinstance(conf[key], str):
      conf[key] = [key]

  for model in conf['models']:
    if model not in known_models:
      raise ValueError(f"Model '{model}' not in list of known models: {known_models}")

  os.makedirs(conf['run_dir'], exist_ok=True)

  desc_file = os.path.join(conf['run_dir'], 'description.txt')
  if not os.path.exists(desc_file):
    print(f"  Writing job description to: {desc_file}")
    with open(desc_file, 'w') as f:
      f.write(conf['run_desc'])

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

  config_file = os.path.join(conf['run_dir'], conf['job'], 'config.json')
  print(f"  Writing job configuration to: {config_file}")
  os.makedirs(os.path.dirname(config_file), exist_ok=True)

  with open(config_file, 'w') as f:
    json.dump(conf, f)

  return conf


def _train_and_test_single_rep(train_df, test_df, removed_input=None, **kwargs):

  import time
  import pandas
  import numpy as np
  from .arv import arv
  from .print_metrics import print_metrics

  indent = 10 * " "

  inputs = kwargs['inputs']
  outputs = kwargs['outputs']
  models = kwargs['models']

  # Determine the current input features
  inputs = [inp for inp in inputs if inp != removed_input] if removed_input else inputs

  actual_df = test_df[['datetime'] + outputs].copy()
  results = {'actual': actual_df}

  # Create DataFrames with only time stamps. Data is put in at end.
  for model in kwargs['models']:
    results[model] = {}
    if model != 'ols':
      results[model]['epoch_metrics'] = {}
    results[model]['predicted'] = test_df[['datetime']].copy()

  resid = any(model.endswith('_resid') for model in models)

  if 'ols' in models or resid:
    from sklearn.linear_model import LinearRegression

    # Ordinary linear regression
    print(f"{indent}Performing {len(outputs)}-output/{len(inputs)}-input linear regression")
    print(f"{indent}  Number of fitting parameters: {len(inputs) + 1}")
    n_train = np.prod(train_df[inputs].shape[0])
    n_test = np.prod(test_df[inputs].shape[0])
    print(f"{indent}  Number of training values: {n_train}")
    print(f"{indent}  Number of testing values:  {n_test} (ratio: {n_test/(n_train+n_test):.2f})")

    start = time.time()
    ols_model = LinearRegression()
    ols_model.fit(train_df[inputs], train_df[outputs])
    ols_train_preds = ols_model.predict(train_df[inputs])
    ols_test_preds = ols_model.predict(test_df[inputs])

    arvs = arv(train_df[outputs], ols_train_preds)
    print(f"{indent} {14 * ' '}", end='')
    print_metrics(outputs, arvs, type="train", dt=time.time() - start)

    arvs = arv(test_df[outputs], ols_test_preds)
    print(f"{indent} {14 * ' '}", end='')
    ols_test_string = print_metrics(outputs, arvs, type="test")

    results['ols']['predicted'][outputs] = ols_test_preds


  from .nn import mimo, miso

  for model in models:
    if model not in ['nn_mimo', 'nn_miso', 'nn_mimo_resid', 'nn_miso_resid']:
      continue

    train_inputs = train_df[inputs]
    train_targets = train_df[outputs]
    test_inputs = test_df[inputs]
    test_targets = test_df[outputs]

    if model.endswith('_resid'):
      delta = ols_train_preds
      train_targets = train_targets - delta

    nn_args = [
      train_inputs,
      train_targets,
      test_inputs,
      test_targets,
      outputs,
      indent,
      kwargs
    ]

    if removed_input is None:
      removed_str = "all inputs"
    else:
      removed_str = f"'{removed_input}' input removed"

    if model.startswith('nn_mimo'):

      msg = f"{indent}Training {len(outputs)}-output neural network with"

      if model.endswith('_resid'):
        print(f"{msg} {removed_str} on ols residuals")
      else:
        print(f"{msg} {removed_str}")

      train_preds, test_preds, train_arvs, test_arvs = mimo(*nn_args)

    if model.startswith('nn_miso'):
      msg = f"{indent}Training {len(outputs)} single-output neural networks with"

      if model.endswith('_resid'):
        print(f"{msg} {removed_str} on ols residuals")
      else:
        print(f"{msg} {removed_str}")

      train_preds, test_preds, train_arvs, test_arvs = miso(*nn_args)

    results[model]['epoch_metrics']['train'] = train_arvs
    results[model]['epoch_metrics']['test'] = test_arvs

    if not model.endswith('_resid'):
      results[model]['predicted'][outputs] = test_preds
    else:
      delta = ols_test_preds
      results[model]['predicted'][outputs] = test_preds + delta
      arvs_star = arv(test_df[outputs], test_preds + delta)
      print(f"{indent}   OLS results")
      print(f"{indent} {14 * ' '}{ols_test_string}")
      print(f"{indent}   NN results when residuals are added back")
      print_metrics(outputs, arvs_star, indent=25, type="test")

  return results


def _save_results(results_dict, job, removed_input, run_dir, method, loo_idx):
  """
  If method = 'loo', directory structure is:
    {removed input name}/
      loo/
        loo_1.pkl: [boot_1, boot_2, ...]
        loo_2.pkl: [boot_1, boot_2, ...]
  where
    boot_i = {
      'actual': df,
      'nn1': df,
      'nn3': df,
      'lr': df
    }
  is a bootstrap repetition and df is a DataFrame with cols of timestamp and outputs
  """

  import os
  import pandas

  if loo_idx is None:
    # lno mode
    file_name = f"{method}.pkl"
    sub_dir = '' # Don't create subdir
  else:
    file_name = f"{method}_{loo_idx}.pkl"
    sub_dir = removed_input
    if removed_input is None:
      # TODO: Address potential for a name conflict if a column name is 'None'
      sub_dir = 'None'

  sub_dir = os.path.join(run_dir, job, method, sub_dir)
  os.makedirs(sub_dir, exist_ok=True)
  file_name = os.path.join(sub_dir, file_name)

  pandas.to_pickle(results_dict, file_name)
  print(f"  Saved '{file_name}'")