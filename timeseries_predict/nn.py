import torch

# In principle, setting to False should speed up computation when GPU is used.
# However, this does not seem to be the case for the tests run.
compute_test_arvs = True

def mimo(train_inputs, train_targets, test_inputs, test_targets, output_names, indent, kwargs):
  import numpy as np
  import time

  from .arv import arv

  from .print_metrics import print_metrics

  dtype = _get_dtype(kwargs['dtype'])
  device = _device(kwargs['device'])

  tensors = _prep_tensors(train_inputs, train_targets, test_inputs, test_targets, dtype, device)
  train_inputs, train_targets, test_inputs, test_targets, scaler_targets = tensors

  train_arvs = np.empty((kwargs['num_epochs'], len(output_names)))
  test_arvs = np.empty((kwargs['num_epochs'], len(output_names)))

  # Multi-output neural network
  model = _NeuralNetwork(train_inputs.shape[1], train_targets.shape[1], hidden_size=kwargs['hidden_size']).to(device)
  print(f"{indent}  Number of fitting parameters: {_num_params(model)}")
  n_train = np.prod(train_inputs.shape[0])
  n_test = np.prod(test_inputs.shape[0])
  print(f"{indent}  Number of training values: {n_train}")
  print(f"{indent}  Number of testing values:  {n_test} (ratio: {n_test/(n_train+n_test):.2f})")
  optimizer = _get_optimizer(model, kwargs['optimizer'], kwargs['optimizer_kwargs'])

  for epoch in range(kwargs['num_epochs']):
    start = time.time()
    epoch_str = f"{epoch + 1}/{kwargs['num_epochs']}".ljust(5)
    epoch_str = f"{indent}    Epoch {epoch_str}"
    print(epoch_str, end='')
    train_arv, train_rmse = _train_single_epoch(model,
                                        optimizer,
                                        train_inputs,
                                        train_targets,
                                        device,
                                        kwargs['batch_size'])
    train_arvs[epoch, :] = train_arv

    print_metrics(output_names, train_arv, type="train", dt=time.time() - start)

    if compute_test_arvs:
      model.eval()
      with torch.no_grad():
        test_arv = arv(test_targets.detach().cpu().numpy(), model(test_inputs).cpu().numpy())
      model.train()
      test_arvs[epoch, :] = test_arv
      print_metrics(output_names, test_arv, type="test", indent=len(epoch_str))

  model.eval()
  with torch.no_grad():
    train_preds = model(train_inputs).cpu().numpy()
    test_preds = model(test_inputs).cpu().numpy()

  # Unscale predictions
  test_preds = scaler_targets.inverse_transform(test_preds)

  return train_preds, test_preds, train_arvs, test_arvs


def miso(train_inputs, train_targets, test_inputs, test_targets, output_names, indent, kwargs):
  import time
  import numpy as np

  from .arv import arv
  from .print_metrics import print_metrics

  dtype = _get_dtype(kwargs['dtype'])
  device = _device(kwargs['device'])

  tensors = _prep_tensors(train_inputs, train_targets, test_inputs, test_targets, dtype, device)
  train_inputs, train_targets, test_inputs, test_targets, scaler_targets = tensors

  test_preds = {}
  train_preds = {}
  train_arvs = np.empty((kwargs['num_epochs'], len(output_names)))
  test_arvs = np.empty((kwargs['num_epochs'], len(output_names)))
  start = time.time()

  # Compute number of model parameters for each single-output network is
  # approximately 1/len(output_names) of that for the multi-output network
  num_inputs = train_inputs.shape[1]
  n_outputs = len(output_names)
  h = kwargs['hidden_size']
  mimo_params = h * (num_inputs + n_outputs + 1) + n_outputs
  # Solve: miso_h * (num_inputs + 2) + 1 = mimo_params / n_outputs
  miso_hidden_size = max(1, round((mimo_params / n_outputs - 1) / (num_inputs + 2)))

  for i in range(len(output_names)):
    model = _NeuralNetwork(train_inputs.shape[1], 1, hidden_size=miso_hidden_size).to(device)
    optimizer = _get_optimizer(model, kwargs['optimizer'], kwargs['optimizer_kwargs'])
    if i == 0:
      n_train = np.prod(train_inputs.shape[0])
      n_test = np.prod(test_inputs.shape[0])
      print(f"{indent}  Number of training values: {n_train}")
      print(f"{indent}  Number of testing values:  {n_test} (ratio: {n_test/(n_train+n_test):.2f})")

    print(f"{indent}  Training single-output neural network for output = '{output_names[i]}'")
    train_target = train_targets[:, i:i+1]
    for epoch in range(kwargs['num_epochs']):
      start = time.time()
      epoch_str = f"{epoch + 1}/{kwargs['num_epochs']}".ljust(5)
      epoch_str = f"{indent}    Epoch {epoch_str}"
      print(epoch_str, end='')
      train_arv, train_rmse = _train_single_epoch(model, optimizer, train_inputs, train_target, device, kwargs['batch_size'])
      train_arvs[epoch, i] = train_arv[0]

      print_metrics(output_names[i], train_arv, type="train", dt=time.time() - start)
      if compute_test_arvs:
        model.eval()
        with torch.no_grad():
          test_arv = arv(test_targets[:, i:i+1].detach().cpu().numpy(), model(test_inputs).cpu().numpy())
        model.train()
        test_arvs[epoch, i] = test_arv[0]
        print_metrics(output_names[i], test_arv, type="test", indent=len(epoch_str))

    model.eval()
    with torch.no_grad():
      train_preds[output_names[i]] = model(train_inputs).cpu().numpy()
      test_preds[output_names[i]] = model(test_inputs).cpu().numpy()

  # Combine outputs results
  train_preds = np.column_stack([train_preds[output] for output in output_names])
  test_preds = np.column_stack([test_preds[output] for output in output_names])

  # Unscale predictions
  train_preds = scaler_targets.inverse_transform(train_preds)
  test_preds = scaler_targets.inverse_transform(test_preds)

  return train_preds, test_preds, train_arvs, test_arvs


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

    loss = torch.nn.MSELoss()(predictions, target)

    # Verify that the PyTorch loss matches the NumPy loss
    #loss_np = np.mean((target.detach().cpu().numpy() - predictions.detach().cpu().numpy()) ** 2)
    #print(f"loss={loss.item():.6f}, loss_np={loss_np:.6f}")

    loss.backward()
    optimizer.step()

    all_predictions.append(predictions.detach().cpu().numpy())
    all_targets.append(target.cpu().numpy())

  # Concatenate predictions and targets
  all_predictions = np.concatenate(all_predictions, axis=0)
  all_targets = np.concatenate(all_targets, axis=0)
  rmse = np.sqrt(np.mean((all_targets - all_predictions) ** 2))
  arvs = arv(all_targets, all_predictions)
  return arvs, rmse


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


def _prep_tensors(train_inputs, train_targets, test_inputs, test_targets, dtype, device):
  from sklearn.preprocessing import MinMaxScaler

  # Scale data
  scaler_inputs = MinMaxScaler()
  train_inputs = scaler_inputs.fit_transform(train_inputs)
  test_inputs = scaler_inputs.transform(test_inputs)

  scaler_targets = MinMaxScaler()
  train_targets = scaler_targets.fit_transform(train_targets)
  test_targets = scaler_targets.transform(test_targets)

  # Convert data to tensors
  train_inputs = torch.tensor(train_inputs, dtype=dtype).to(device)
  train_targets = torch.tensor(train_targets, dtype=dtype).to(device)
  test_inputs = torch.tensor(test_inputs, dtype=dtype).to(device)
  test_targets = torch.tensor(test_targets, dtype=dtype).to(device)

  return train_inputs, train_targets, test_inputs, test_targets, scaler_targets
