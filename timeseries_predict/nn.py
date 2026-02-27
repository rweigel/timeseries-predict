import torch


def mimo(train_inputs, train_targets, test_inputs, delta, output_names, indent, kwargs):
  import numpy as np

  from .print_metrics import print_metrics

  dtype = _get_dtype(kwargs['dtype'])
  device = _device(kwargs['device'])

  if delta is not None:
    train_targets = train_targets - delta

  train_inputs, train_targets, test_inputs, scaler_targets = _prep_tensors(train_inputs, train_targets, test_inputs, dtype, device)

  arvs = []
  # Multi-output neural network
  model = _NeuralNetwork(train_inputs.shape[1], train_targets.shape[1], hidden_size=kwargs['hidden_size']).to(device)
  print(f"{indent}  Number of fitting parameters: {_num_params(model)}")
  print(f"{indent}  Number of training values: {np.prod(train_inputs.shape)}")
  optimizer = _get_optimizer(model, kwargs['optimizer'], kwargs['optimizer_kwargs'])

  for epoch in range(kwargs['num_epochs']):
    epoch_str = f"{epoch + 1}/{kwargs['num_epochs']}".ljust(5)
    print(f"{indent}  Epoch {epoch_str}", end='')
    _arvs, losses = _train_single_epoch(model, optimizer, train_inputs, train_targets, device, kwargs['batch_size'])
    print_metrics(output_names, _arvs, losses)
    arvs.append(_arvs)

  model.eval()
  with torch.no_grad():
    test_preds = model(test_inputs).cpu().numpy()  # Multi-output NN predictions

  # Unscale predictions
  test_preds = scaler_targets.inverse_transform(test_preds)

  return test_preds, arvs


def miso(train_inputs, train_targets, test_inputs, delta, output_names, indent, kwargs):
  import numpy as np

  from .print_metrics import print_metrics

  dtype = _get_dtype(kwargs['dtype'])
  device = _device(kwargs['device'])

  train_inputs, train_targets, test_inputs, scaler_targets = _prep_tensors(train_inputs, train_targets, test_inputs, dtype, device)

  test_preds = {}
  arvs = []
  for i, output in enumerate(output_names):
    model = _NeuralNetwork(train_inputs.shape[1], 1, hidden_size=kwargs['hidden_size']).to(device)
    optimizer = _get_optimizer(model, kwargs['optimizer'], kwargs['optimizer_kwargs'])
    if i == 0:
      print(f"{indent}  Number of fitting parameters: {_num_params(model)}")
      print(f"{indent}  Number of training values: {np.prod(train_inputs.shape)}")

    print(f"{indent}  Training single-output neural network for output = '{output}'")
    train_target = train_targets[:, i:i+1]
    for epoch in range(kwargs['num_epochs']):
      print(f"{indent}    Epoch {epoch + 1}/{kwargs['num_epochs']}", end='')
      _arv, loss = _train_single_epoch(model, optimizer, train_inputs, train_target, device, kwargs['batch_size'])
      print_metrics(output, _arv, loss)
      arvs.append(_arv)

    model.eval()
    with torch.no_grad():
      test_preds[output] = model(test_inputs).cpu().numpy()

  # arvs is in form
  # [ out1_arv_epoch1, out1_arv_epoch2, ..., out2_arv_epoch1, out2_arv_epoch2, ... ]
  # Convert to form of mimo
  # [ [out1_arv_epoch1, out2_arv_epoch1, ...], [out1_arv_epoch2, out2_arv_epoch2, ...], ... ]
  arvs = list(np.reshape(arvs, (-1, len(output_names))))

  # Combine outputs results
  test_preds = np.column_stack([test_preds[output] for output in output_names])

  # Unscale predictions
  test_preds = scaler_targets.inverse_transform(test_preds)

  return test_preds, arvs


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


def _train_single_epoch(model, optimizer, train_inputs, train_targets, device, batch_size):
  import numpy as np

  from .arv import arv

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


def _device(device_name):

  if device_name is None:
    return None

  if device_name not in ['cpu', 'cuda', 'mps']:
    raise ValueError(f"Device '{device_name}' not in ['cpu', 'cuda', 'mps']")

  device = None

  if device_name == 'mps':
    if torch.backends.mps.is_available():
      device = torch.device("mps") # Use MPS on Mac

  if device_name == 'cuda':
    if torch.cuda.is_available():
      device = torch.device("cuda") # Use CUDA on Windows/Linux

  if device_name == 'cpu':
    if torch.cuda.is_available():
      device = torch.device("cpu")

  if device is None:
    raise ValueError(f"Device '{device_name}' was requested but is not available.")

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


def _get_dtype(dtype_name):

  dtypes = torch.__dict__.values()
  dtype_names = []
  for dtype_module in dtypes:
    if isinstance(dtype_module, torch.dtype):
      dtype_names.append(dtype_module.__str__().split('.')[-1])
      if dtype_module.__str__().split('.')[-1] == dtype_name:
        return dtype_module

  raise ValueError(f"dtype {dtype_name}' not found in = {dtype_names}")


def _num_params(model):
  return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _prep_tensors(train_inputs, train_targets, test_inputs, dtype, device):
  from sklearn.preprocessing import MinMaxScaler

  # Scale data
  scaler_inputs = MinMaxScaler()
  train_inputs = scaler_inputs.fit_transform(train_inputs)
  test_inputs = scaler_inputs.transform(test_inputs)

  scaler_targets = MinMaxScaler()
  train_targets = scaler_targets.fit_transform(train_targets)

  # Convert data to tensors
  train_inputs = torch.tensor(train_inputs, dtype=dtype).to(device)
  train_targets = torch.tensor(train_targets, dtype=dtype).to(device)
  test_inputs = torch.tensor(test_inputs, dtype=dtype).to(device)

  return train_inputs, train_targets, test_inputs, scaler_targets
