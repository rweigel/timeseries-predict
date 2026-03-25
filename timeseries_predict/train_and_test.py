def train_and_test(job_dfs, conf):

  import pandas
  import utilrsw

  from .summary import summary

  conf = _prep_config(conf)

  utilrsw.hline()

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
      else:
        print("    Training using single DataFrame")
        method_label = "lno"
        train_test_data = job_dfs[0]

      if conf['lags'] is not None:
        for p, max_lag in conf['lags'].items():
          for lag in range(1, 4):
            p_lag = f'{p}_{lag}'
            train_test_data[p_lag] = train_test_data[p].shift(lag)
            train_test_data = train_test_data.iloc[lag:].reset_index(drop=True)
            conf['inputs'].append(p_lag)

      # Loop for repetitions
      results = []
      for rep in range(conf['n_reps']):
        print(f"        Repetition {rep + 1}/{conf['n_reps']}")

        train_data = train_test_data.sample(frac=conf['train_fraction'], random_state=rep)
        test_data = train_test_data.drop(train_data.index)
        rep = _train_and_test_single_rep(train_data, test_data, removed_input=removed_input, **conf)

        result = {}
        result['inputs'] = [s for s in conf['inputs'] if s != removed_input]
        result['outputs'] = conf['outputs']
        result['indices'] = {
          'train': test_data.index,
          'test': train_data.index
        }
        result['data'] = {
          'train': train_data,
          'test': test_data
        }
        result['models'] = {**rep}
        results.append(result)

        #if method_label == 'loo':
        #  results['indices']['validation'] = validation_indices

      if len(job_dfs) == 1:
        _save_results(results, conf['job'], removed_input, conf['run_dir'], method_label, None)
      else:
        idx_width = len(str(num_dfs))
        idx_zero_padded = str(i + 1).zfill(idx_width)
        _save_results(results, conf['job'], removed_input, conf['run_dir'], method_label, idx_zero_padded)

  summary(conf['run_dir'], job=conf['job'])

  print(f"Finished job '{conf['job']}'\n")


def _prep_config(conf):
  import os
  import yaml

  os.makedirs(conf['run_dir'], exist_ok=True)

  desc_file = os.path.join(conf['run_dir'], 'description.txt')
  if not os.path.exists(desc_file):
    print(f"  Writing job description to: {desc_file}")
    with open(desc_file, 'w') as f:
      f.write(conf['run_desc'])

  config_file = os.path.join(conf['run_dir'], conf['job'], 'config.yaml')
  print(f"  Writing job configuration to: {config_file}")
  os.makedirs(os.path.dirname(config_file), exist_ok=True)

  with open(config_file, 'w') as f:
    yaml.safe_dump(conf, f)

  return conf


def _train_and_test_single_rep(train_df, test_df, removed_input=None, **kwargs):

  import time
  import numpy as np
  from .arv import arv
  from .print_metrics import print_metrics

  indent = 10 * " "

  inputs = kwargs['inputs']
  outputs = kwargs['outputs']
  models = kwargs['models']

  # Determine the current input features
  inputs = [inp for inp in inputs if inp != removed_input] if removed_input else inputs

  results = {}

  # Create DataFrames with only time stamps. Data is put in at end.
  for model in kwargs['models']:
    results[model] = {'predicted': {}, 'metrics': {}}
    if model != 'ols':
      results[model]['epoch_metrics'] = {}
    results[model]['predicted']['test'] = test_df[['datetime']].copy()
    results[model]['predicted']['train'] = train_df[['datetime']].copy()

  resid = any(model.endswith('_resid') for model in models)

  if 'ols' in models or resid:
    from sklearn.linear_model import LinearRegression

    # Ordinary linear regression
    _print_prolog(inputs, outputs, 'ols', removed_input, indent)

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

    arvs_train = arv(train_df[outputs], ols_train_preds)
    print(f"{indent} {14 * ' '}", end='')
    # ols_train_string is used later when printing results for residual models
    ols_train_string = print_metrics(outputs, arvs_train, type="train", dt=time.time() - start)

    arvs_test = arv(test_df[outputs], ols_test_preds)
    print(f"{indent} {14 * ' '}", end='')
    # ols_test_string is used later when printing results for residual models
    ols_test_string = print_metrics(outputs, arvs_test, type="test")

    results['ols']['metrics']['train'] = arvs_train
    results['ols']['metrics']['test'] = arvs_test

    results['ols']['predicted']['train'][outputs] = ols_train_preds
    results['ols']['predicted']['test'][outputs] = ols_test_preds

  from .nn import mimo, miso

  for model in models:

    if model not in ['nn_mimo', 'nn_miso', 'nn_mimo_resid', 'nn_miso_resid']:
      continue

    if model.endswith('_resid'):
      delta = ols_train_preds
      train_df[outputs] = train_df[outputs] - delta

    nn_args = [
      train_df[inputs],
      train_df[outputs],
      test_df[inputs],
      test_df[outputs],
      outputs,
      indent,
      kwargs
    ]

    _print_prolog(inputs, outputs, model, removed_input, indent)

    if model.startswith('nn_mimo'):
      train_preds, test_preds, train_arvs, test_arvs = mimo(*nn_args)

    if model.startswith('nn_miso'):
      train_preds, test_preds, train_arvs, test_arvs = miso(*nn_args)

    results[model]['epoch_metrics']['train'] = train_arvs
    results[model]['epoch_metrics']['test'] = test_arvs

    results[model]['metrics']['train'] = train_arvs[-1, :]
    results[model]['metrics']['test'] = test_arvs[-1, :]

    if not model.endswith('_resid'):
      results[model]['predicted']['test'][outputs] = test_preds
      results[model]['predicted']['train'][outputs] = train_preds
    else:
      results[model]['predicted']['test'][outputs] = test_preds + ols_test_preds
      results[model]['predicted']['train'][outputs] = train_preds + ols_train_preds
      arvs_train_star = arv(train_df[outputs], train_preds + ols_train_preds)
      arvs_test_star = arv(test_df[outputs], test_preds + ols_test_preds)
      results[model]['metrics']['train*'] = arvs_train_star
      results[model]['metrics']['test*'] = arvs_test_star
      print(f"{indent}   OLS results")
      print(f"{indent} {14 * ' '}{ols_train_string}")
      print(f"{indent} {14 * ' '}{ols_test_string}")
      print(f"{indent}   NN results when residuals are added back")
      print_metrics(outputs, arvs_train_star, indent=25, type="train")
      print_metrics(outputs, arvs_test_star, indent=25, type="test")

  return results


def _print_prolog(inputs, outputs, model, removed_input, indent):

  if removed_input is None:
    removed_str = f"all {len(inputs)} input(s)."
  else:
    removed_str = f"'{removed_input}' input removed."

  msg = ""
  if model == 'ols':
    msg = f"{indent}Performing {len(outputs)}-output linear regression with"
  if model.startswith('nn_mimo'):
    msg = f"{indent}Training {len(outputs)}-output neural network with"
  if model.startswith('nn_miso'):
    msg = f"{indent}Training {len(outputs)} single-output neural networks with"

  if model.endswith('_resid'):
    print(f"{msg} {removed_str} on ols residuals")
  else:
    print(f"{msg} {removed_str}")

  print(f"{indent}  Inputs:  {inputs}")
  print(f"{indent}  Outputs: {outputs}")


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
  is a repetition and df is a DataFrame with cols of timestamp and outputs
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