import os
import json

print("Importing torch. ", end="")
import torch
print("Done")

print("Importing sklearn. ", end="")
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
print("Done")

print("Importing numpy and pandas. ", end="")
import numpy as np
import pandas as pd
print("Done")

from .arv import arv
from .summary import summary


def train_and_test(combined_dfs, conf, parallel_jobs=False):

  """
  """

  conf = _prep_config(conf)

  print(f"{conf['tag']} started")
  is_loo = len(combined_dfs) > 1  # Check if leave-one-out is needed
  datasets = combined_dfs if is_loo else [combined_dfs[0]]

  for removed_input in conf['removed_inputs']:
    print(f"  Removed input: {removed_input}")
    for i, test_data in enumerate(datasets):
      if is_loo:
        # For leave-one-out, concatenate training datasets excluding the current test dataset
        print(f"    Training with DataFrame {i + 1}/{len(datasets)} excluded")
        train_data_subset = [df for j, df in enumerate(datasets) if j != i]
        train_data = pd.concat(train_data_subset, ignore_index=True)
        method_label = f"loo_{i + 1}"
      else:
        print(f"    Training using all {len(datasets)} DataFrame(s)")
        train_data = datasets[0]
        method_label = "all"

      # Loop for bootstrap repetitions
      results = []
      for rep in range(conf['num_boot_reps']):
        print(f"        Bootstrap repetition {rep + 1}/{conf['num_boot_reps']}")
        train_boot = train_data.sample(frac=0.8, random_state=rep)
        test_data = datasets[i] if is_loo else datasets[0]  # Test data comes from the current fold
        result = _train_and_test_single_rep(train_boot, test_data, removed_input=removed_input, **conf)
        results.append(result)

      _save(results, conf['tag'], removed_input, conf['results_dir'], method=method_label)

  print("  Creating tables and plots")
  summary(conf['tag'], results_dir=conf['results_dir'])

  print(f"{conf['tag']} finished\n")


def _prep_config(conf):

  known_models = ['ols', 'nn_miso', 'nn_mimo']

  defaults = {
    'tag': 'tag1',
    'lr': 0.001,
    'models': known_models,
    'device': None,
    'num_epochs': 200,
    'batch_size': 256,
    'results_dir': './results',
    'num_boot_reps': 1
  }

  for key in defaults.keys():
    if key not in conf or conf[key] is None:
      conf[key] = defaults[key]

  for model in conf['models']:
    if model not in known_models:
      raise ValueError(f"Model '{model}' not in list of known models: {known_models}")

  removed_inputs = conf['removed_inputs']
  if removed_inputs is None:
    removed_inputs = [None]
  if removed_inputs is True:
    removed_inputs = [None] + conf['inputs']
  conf['removed_inputs'] = removed_inputs

  config_file = os.path.join(conf['results_dir'], conf['tag'], 'config.json')
  print(f"Writing: {config_file}")
  if not os.path.exists(os.path.dirname(config_file)):
    os.makedirs(os.path.dirname(config_file))

  json.dump(conf, open(config_file, 'w'))

  return conf


def _device(device_name):

  if device_name is None:
    return None

  if device_name == 'mps':
    if torch.backends.mps.is_available():
      device = torch.device("mps") # Use MPS on Mac

  if device_name == 'cuda':
    if torch.cuda.is_available():
      device = torch.device("cuda") # Use CUDA on Windows/Linux

  if device_name == 'cpu':
    if torch.cuda.is_available():
      device = torch.device("cpu")

  return device


def _get_activation(activation_name):

  activations = torch.nn.modules.activation.__all__
  if activation_name in activations:
    return getattr(torch.nn, activation_name)
  else:
    raise ValueError(f"Activation '{activation_name}' not found in torch.nn.modules.activation.__all__ = {activations}")

def _get_optimizer(model, optimizer_name, optimizer_kwargs):

  optimizers = torch.optim.__all__
  if optimizer_name in optimizers:
    optim = getattr(torch.optim, optimizer_name)
    return optim(model.parameters(), **optimizer_kwargs)
  else:
    raise ValueError(f"Optimizer '{optimizer_name}' not found in torch.optim.__all__ = {optimizers}")


class _NeuralNetwork(torch.nn.Module):

  def __init__(self, num_inputs, num_outputs, activation="Tanh", hidden_size=32):
    super().__init__()
    activation_class = _get_activation(activation)
    # Default is 
    self.network = torch.nn.Sequential(
      torch.nn.Linear(num_inputs, hidden_size),
      activation_class(), # Default will be to do torch.nn.Tanh()
      torch.nn.Linear(hidden_size, num_outputs)
    )

  def forward(self, x):
    output = self.network(x)
    return output


def _train_and_test_single_rep(train_df, test_df, removed_input=None, **kwargs):

  indent = "          "

  inputs = kwargs['inputs']
  outputs = kwargs['outputs']
  models = kwargs['models']
  num_epochs = kwargs['num_epochs']
  batch_size = kwargs['batch_size']
  hidden_size = kwargs['hidden_size']
  device_name = kwargs['device']
  device = _device(device_name)
  dtype = torch.float32

  if device is None:
    print(f"{indent}Device '{device_name}' is not available. Using 'cpu' instead.")
    device = torch.device("cpu")
  else:
    print(f"{indent}Using device: {device}")

  # Determine the current input features
  current_inputs = [inp for inp in inputs if inp != removed_input] if removed_input else inputs
  num_inputs = len(current_inputs)

  # Fill missing values
  # TODO: Check for NaNs and abort if found. Filling should be done in data
  #       prep function before any nn code called.
  train_df.fillna(train_df.mean(numeric_only=True), inplace=True)
  test_df.fillna(test_df.mean(numeric_only=True), inplace=True)

  # Scale data
  scaler_inputs = MinMaxScaler()
  scaler_targets = MinMaxScaler()

  train_inputs = scaler_inputs.fit_transform(train_df[current_inputs])
  train_targets = scaler_targets.fit_transform(train_df[outputs])
  test_inputs = scaler_inputs.transform(test_df[current_inputs])

  # Convert data to tensors
  train_inputs = torch.tensor(train_inputs, dtype=dtype).to(device)
  train_targets = torch.tensor(train_targets, dtype=dtype).to(device)
  test_inputs = torch.tensor(test_inputs, dtype=dtype).to(device)

  # No scaling or conversion to tensors needed for targets because they
  # are not used for training.
  test_targets = test_df[outputs].values

  # Create DataFrames with only time stamps. Data is put in at end.
  results = {'actual': {'timestamp': test_df['datetime'].values}}
  for output in outputs:
    results['actual'][output] = test_targets[:, outputs.index(output)]
  results['actual'] = pd.DataFrame(results['actual'])

  for model in kwargs['models']:
    results[model] = {}
    if model != 'ols':
      results[model]['epochs'] = []
    results[model]['predicted'] = pd.DataFrame({'timestamp': test_df['datetime'].values})

  def _print_metrics(outputs, arvs, total_loss):
    if isinstance(arvs, list):
      for output, _arv in zip(outputs, arvs):
          print(f" | {output} ARV = {_arv:.3f}", end='')
      print(f" | loss = {total_loss:.4f}")
    else:
      print(f" | {outputs} ARV = {arvs:.3f} | loss = {total_loss:.4f}")

  if 'ols' in models:
    # Ordinary linear regression
    print(f"{indent}Performing linear regression")
    model = LinearRegression()
    model.fit(train_df[current_inputs], train_df[outputs])
    test_preds = model.predict(test_df[current_inputs])
    arvs = arv(test_targets, test_preds)
    print(f"{indent}  Epoch N/A", end='')
    _print_metrics(outputs, arvs, np.nan)
    results['ols']['predicted'][outputs] = test_preds

  if 'nn_mimo' in models:
    # Multi-output neural network
    print(f"{indent}Training {num_inputs}-output neural network w/ input '{removed_input}' removed")
    model = _NeuralNetwork(num_inputs, len(outputs), hidden_size=hidden_size).to(device)
    optimizer = _get_optimizer(model, kwargs['optimizer'], kwargs['optimizer_kwargs'])
    for epoch in range(num_epochs):
      print(f"{indent}  Epoch {epoch + 1}/{num_epochs}", end='')
      arvs, losses = _train_and_test_single_epoch(model, optimizer, train_inputs, train_targets, device, batch_size)
      _print_metrics(outputs, arvs, losses)
      results['nn_mimo']['epochs'].append(arvs)

    model.eval()
    with torch.no_grad():
      test_preds = model(test_inputs).cpu().numpy()  # Multi-output NN predictions

    # Unscale predictions
    test_preds = scaler_targets.inverse_transform(test_preds)
    results['nn_mimo']['predicted'][outputs] = test_preds

  if 'nn_miso' in models:
    # Single-output neural networks, one for each output
    print(f"{indent}Training {num_inputs} single-output neural networks w/ input '{removed_input}' removed")
    test_preds = {}
    for i, output in enumerate(outputs):
      print(f"{indent}  Training single-output neural network for {output}")
      model = _NeuralNetwork(num_inputs, 1, hidden_size=hidden_size).to(device)
      optimizer = _get_optimizer(model, kwargs['optimizer'], kwargs['optimizer_kwargs'])
      train_target = train_targets[:, i:i + 1]

      for epoch in range(num_epochs):
          print(f"{indent}    Epoch {epoch + 1}/{num_epochs}", end='')
          arvs, losses = _train_and_test_single_epoch(model, optimizer, train_inputs, train_target, device, batch_size)
          _print_metrics(output, arvs, losses)
          results['nn_miso']['epochs'].append(arvs)

      model.eval()
      with torch.no_grad():
        test_preds[output] = model(test_inputs).cpu().numpy()

    # Combine outputs results
    test_preds = np.column_stack([test_preds[output] for output in outputs])

    # Unscale predictions
    test_preds = scaler_targets.inverse_transform(test_preds)
    results['nn_miso']['predicted'][outputs] = test_preds

  return results


def _train_and_test_single_epoch(model, optimizer, train_inputs, train_targets, device, batch_size):

  model.train()
  total_loss = 0
  all_predictions = []
  all_targets = []

  DataLoader = torch.utils.data.DataLoader
  TensorDataset = torch.utils.data.TensorDataset

  tds = TensorDataset(train_inputs, train_targets)
  data_loader = DataLoader(tds, batch_size=batch_size, shuffle=True)

  for data, target in data_loader:
    data, target = data.to(device), target.to(device)

    optimizer.zero_grad()
    predictions = model(data)

    # Compute the loss for multi-output
    loss = torch.nn.MSELoss()(predictions, target)
    loss.backward()
    optimizer.step()
    total_loss += loss.item()

    all_predictions.append(predictions.detach().cpu().numpy())
    all_targets.append(target.cpu().numpy())

  # Concatenate predictions and targets
  all_predictions = np.concatenate(all_predictions, axis=0)
  all_targets = np.concatenate(all_targets, axis=0)

  arvs = arv(all_targets, all_predictions)

  return arvs, total_loss


def _save(results_dict, tag, removed_input, results_directory, method):

  # If method = 'loo'
  # removed_input/
  #   loo/
  #     loo_1.pkl: [boot1, boot2, ...]
  #     loo_2.pkl: [boot1, boot2, ...]
  # where
  #   bootN = {actual: df, nn1: df, nn3: df, lr: df}
  # is a bootstrap repetition and
  #   df is a DataFrame with columns of timestamp and outputs

  if removed_input is None:
    removed_input = 'None'

  subdir = os.path.join(results_directory, tag, removed_input)
  if method.startswith('loo'):
    subdir = os.path.join(subdir, 'loo')

  if not os.path.exists(subdir):
    os.makedirs(subdir)

  pkl_filepath = os.path.join(subdir, f"{method}.pkl")
  pd.to_pickle(results_dict, pkl_filepath)
  print(f"  Saved '{pkl_filepath}'")