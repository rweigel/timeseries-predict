def train_and_test(job_dfs, conf):
  import shutil
  import pandas

  from .summary import summary

  print(f"{'-'*shutil.get_terminal_size(fallback=(80, 24)).columns}")

  s = "s" if len(job_dfs) != 1 else ""
  print(f"Starting job '{conf['job']}' with {len(job_dfs)} DataFrame{s}")

  if len(job_dfs) == 0:
    raise ValueError(f"len(job_dfs) = 0 for job '{conf['job']}'.")

  conf = _prep_config(conf)

  for removed_input in conf['removed_inputs']:
    print(f"  Removed input: {removed_input}")
    for i in range(len(job_dfs)):
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
        # TODO: Pass test_data_rep
        #test_data_rep = train_data_rep.drop(train_test_data.index)
        result = _train_and_test_single_rep(train_data_rep, validation_data, removed_input=removed_input, **conf)
        results.append(result)

      _save_results(results, conf['job'], removed_input, conf['run_dir'], method_label, i+1)

  print("\n  Creating tables and plots")

  summary(conf['run_dir'], job=conf['job'])

  print(f"Finished job '{conf['job']}'\n")


def _prep_config(conf):
  import os
  import copy
  import json

  conf = copy.deepcopy(conf)  # avoid mutating the caller's dict

  known_models = ['ols', 'nn_miso', 'nn_mimo', 'nn_miso_resid', 'nn_mimo_resid']

  for model in conf['models']:
    if model not in known_models:
      raise ValueError(f"Model '{model}' not in list of known models: {known_models}")

  defaults = {
    'job': 'job1',
    'lr': 0.001,
    'models': known_models,
    'device': None,
    'num_epochs': 200,
    'batch_size': 256,
    'run_dir': './results',
    'num_boot_reps': 1
  }

  os.makedirs(conf['run_dir'], exist_ok=True)

  desc_file = os.path.join(conf['run_dir'], 'description.txt')
  if not os.path.exists(desc_file):
    print(f"  Writing job description to: {desc_file}")
    with open(desc_file, 'w') as f:
      f.write(conf['run_desc'])

  for key in defaults.keys():
    if key not in conf or conf[key] is None:
      conf[key] = defaults[key]

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

  test_targets = test_df[outputs].values

  # Create DataFrames with only time stamps. Data is put in at end.
  results = {'actual': {'timestamp': test_df['datetime'].values}}
  for output in outputs:
    results['actual'][output] = test_targets[:, outputs.index(output)]
  results['actual'] = pandas.DataFrame(results['actual'])

  for model in kwargs['models']:
    results[model] = {}
    if model != 'ols':
      results[model]['epochs'] = []
    results[model]['predicted'] = pandas.DataFrame({'timestamp': test_df['datetime'].values})

  resid = any(model.endswith('_resid') for model in models)

  if 'ols' in models or resid:
    from sklearn.linear_model import LinearRegression

    # Ordinary linear regression
    print(f"{indent}Performing {len(outputs)}-output/{len(inputs)}-input linear regression")
    print(f"{indent}  Number of fitting parameters: {len(inputs) + 1}")
    print(f"{indent}  Number of training values: {np.prod(train_df[inputs].shape)}")

    ols_model = LinearRegression()
    ols_model.fit(train_df[inputs], train_df[outputs])
    ols_train_preds = ols_model.predict(train_df[inputs])
    ols_test_preds = ols_model.predict(test_df[inputs])

    arvs = arv(train_df[outputs], ols_train_preds)
    print(f"{indent}  Train set", end='')
    print_metrics(outputs, arvs, np.nan)

    arvs = arv(test_df[outputs], ols_test_preds)
    print(f"{indent}  Test set ", end='')
    print_metrics(outputs, arvs, np.nan)

    results['ols']['predicted'][outputs] = ols_test_preds


  from .nn import mimo, miso

  for model in models:
    if model not in ['nn_mimo', 'nn_miso', 'nn_mimo_resid', 'nn_miso_resid']:
      continue

    delta = None
    if model.endswith('_resid'):
      delta = ols_train_preds

    if model.startswith('nn_mimo'):
      msg = f"{indent}Training {len(outputs)}-output neural network with input "
      if model.endswith('_resid'):
        print(f"{msg} '{removed_input}' removed on ols residuals")
      else:
        print(f"{msg} '{removed_input}' removed")

      test_preds, arvs = mimo(train_df[inputs],
                                  train_df[outputs],
                                  test_df[inputs],
                                  delta,
                                  outputs,
                                  indent,
                                  kwargs)

    if model.startswith('nn_miso'):
      msg = f"{indent}Training {len(outputs)} single-output neural networks with input "
      if model.endswith('_resid'):
        print(f"{msg} '{removed_input}' removed on ols residuals")
      else:
        print(f"{msg} '{removed_input}' removed")

      test_preds, arvs = miso(train_df[inputs],
                                  train_df[outputs],
                                  test_df[inputs],
                                  delta,
                                  outputs,
                                  indent,
                                  kwargs)

    arvs = arv(test_df[outputs], test_preds)
    results[model]['epochs'] = arvs

    print(f"{indent}  Test set   ", end='')
    print_metrics(outputs, arvs, np.nan)

    if delta is None:
      results[model]['predicted'][outputs] = test_preds
    else:
      delta = ols_test_preds
      results[model]['predicted'][outputs] = test_preds + delta
      arvs_star = arv(test_df[outputs], test_preds + delta)
      print(f"{indent}  Test set*  ", end='')
      print_metrics(outputs, arvs_star, np.nan)

  return results


def _save_results(results_dict, job, removed_input, run_dir, method, loo_idx):
  import os
  import pandas

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

  if removed_input is None:
    # TODO: Address potential for a name conflict if a column name is 'None'
    removed_input = 'None'

  subdir = os.path.join(run_dir, job, method, removed_input)

  os.makedirs(subdir, exist_ok=True)
  if loo_idx is not None:
    method = f"{method}_{loo_idx}"
  pkl_filepath = os.path.join(subdir, f"{method}.pkl")
  pandas.to_pickle(results_dict, pkl_filepath)
  print(f"  Saved '{pkl_filepath}'")