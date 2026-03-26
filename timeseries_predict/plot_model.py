def plot_model(model, input_names, output_names):
  """Render three plots:
    1. A torchviz computation graph      -> file_path.pdf
    2. A visualtorch layered diagram     -> file_path_diagram.png
    3. A custom matplotlib connection diagram -> file_path_diagram_custom.png
  """
  import torch

  # Derive layer info from all Linear layers
  linear_layers = [m for m in model.modules() if isinstance(m, torch.nn.Linear)]
  layer_sizes = [linear_layers[0].in_features] + [l.out_features for l in linear_layers]

  # Detect activation function name from first non-Linear module
  activation_name = None
  for m in model.modules():
    if isinstance(m, torch.nn.Module) and not isinstance(m, (torch.nn.Linear, model.__class__)):
      if m.__class__.__module__ == 'torch.nn.modules.activation':
        activation_name = m.__class__.__name__
        break

  # Count trainable parameters
  n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

  _plot_standard(layer_sizes, input_names, output_names,
                 activation=activation_name,
                 n_params=n_params)


def _build_nn_equation(layer_sizes, input_names=None, output_names=None, activation=None):
  """Build a LaTeX summation equation string describing the network's forward pass."""
  idx_vars = ['i', 'j', 'k', 'm', 'n', 'p', 'q', 'r']
  n_layers = len(layer_sizes)
  n_hidden = n_layers - 2
  parts = []

  act_str = r'\mathrm{' + activation + r'}' if activation else 'f'

  for l in range(1, n_layers):
    in_var  = idx_vars[l - 1]
    out_var = idx_vars[l]
    n_in    = layer_sizes[l - 1]
    is_out  = (l == n_layers - 1)

    # Input symbol for this layer
    if l == 1:
      in_sym = r'x_{' + in_var + r'}'
    elif n_hidden == 1:
      in_sym = r'h_{' + in_var + r'}'
    else:
      in_sym = r'h^{(' + str(l - 1) + r')}_{' + in_var + r'}'

    w = r'w^{(' + str(l) + r')}_{' + in_var + out_var + r'}'
    b = r'b^{(' + str(l) + r')}_{' + out_var + r'}'
    summ = r'\sum\limits_{' + in_var + r'=1}^{' + str(n_in) + r'} ' + w + r'\,' + in_sym + r' + ' + b

    if is_out:
      out_sym = r'\hat{y}_{' + out_var + r'}'
      parts.append(out_sym + r' = ' + summ)
    else:
      out_sym = (r'h_{' + out_var + r'}' if n_hidden == 1
                 else r'h^{(' + str(l) + r')}_{' + out_var + r'}')
      parts.append(out_sym + r' = ' + act_str + r'\!\left(' + summ + r'\right)')

  return parts


def _build_param_block(layer_sizes, n_params):
  """Return a single LaTeX array string with three aligned parameter lines:
    #params = <symbolic>
            = <numeric>
            = <total>
  The '=' signs are aligned via \\begin{array}{rl}.
  """
  n_layers = len(layer_sizes)
  n_hidden = n_layers - 2

  # Symbolic names
  sym = [r'n_{\mathrm{in}}']
  if n_hidden == 1:
    sym.append(r'n_{\mathrm{h}}')
  else:
    for i in range(n_hidden):
      sym.append(r'n_{\mathrm{h}_{' + str(i + 1) + r'}}')
  sym.append(r'n_{\mathrm{out}}')

  sym_terms = [sym[l] + r'(' + sym[l - 1] + r'+1)' for l in range(1, n_layers)]
  num_terms = [str(layer_sizes[l]) + r'(' + str(layer_sizes[l - 1]) + r'+1)'
               for l in range(1, n_layers)]

  row1 = r'\#\,\mathrm{params}&= ' + ' + '.join(sym_terms)
  row2 = r'&= ' + ' + '.join(num_terms)
  row3 = r'&= ' + str(n_params)

  return (r'\begin{array}{r@{\,}l}' +
          row1 + r'\\' +
          row2 + r'\\' +
          row3 +
          r'\end{array}')


def _plot_standard(layer_sizes, input_names=None, output_names=None, activation=None, n_params=None, file_path='model-diagram-standard.png'):
  """Draw a traditional neural network connection diagram using matplotlib.
  Bias nodes (+1) are shown for all layers except the output layer.
  Layers with more than DISPLAY_LIMIT nodes are shown with an ellipsis (first
  N_EACH nodes, a vertical dots marker, then the last N_EACH nodes).
  A summation equation for the network is displayed to the right.
  """
  import matplotlib
  import matplotlib.pyplot as plt
  matplotlib.rcParams['text.usetex'] = True
  matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

  DISPLAY_LIMIT = 16  # layers with more nodes than this use ellipsis
  N_EACH = 7          # nodes shown on each side of the ellipsis

  n_layers = len(layer_sizes)
  # Bias nodes exist on all layers except the output layer
  has_bias = [True] * (n_layers - 1) + [False]

  # Determine which original node indices to display per layer
  layer_shown = []
  for n in layer_sizes:
    if n > DISPLAY_LIMIT:
      shown = list(range(N_EACH)) + list(range(n - N_EACH, n))
    else:
      shown = list(range(n))
    layer_shown.append(shown)
  layer_truncated = [len(layer_shown[i]) < layer_sizes[i] for i in range(n_layers)]

  # Effective vertical slot count per layer (shown nodes + 1 if ellipsis + 1 if bias)
  disp_counts = [len(s) for s in layer_shown]
  eff_counts = [disp_counts[i] + (1 if layer_truncated[i] else 0) + (1 if has_bias[i] else 0)
                for i in range(n_layers)]
  max_eff = max(eff_counts)

  node_radius = 0.25
  x_spacing = 2.0

  # Build node positions, ellipsis positions, and bias positions
  node_positions   = []  # list of lists of (x, y) for displayed circles
  ellipsis_pos     = []  # (x, y) or None per layer
  bias_positions   = []  # (x, y) or None per layer

  for layer_idx in range(n_layers):
    x = layer_idx * x_spacing
    n_shown = disp_counts[layer_idx]
    n_eff = eff_counts[layer_idx]
    top_offset = (max_eff - n_eff) / 2

    if layer_truncated[layer_idx]:
      # Top group: N_EACH nodes
      ys_top = [top_offset + i for i in range(N_EACH)]
      # Ellipsis sits in the slot right after the top group
      ell_y = top_offset + N_EACH
      # Bottom group: N_EACH nodes, one slot below the ellipsis
      ys_bot = [top_offset + N_EACH + 1 + i for i in range(N_EACH)]
      ys = ys_top + ys_bot
      ellipsis_pos.append((x, ell_y))
      bias_y_base = top_offset + N_EACH * 2 + 1
    else:
      ys = [top_offset + i for i in range(n_shown)]
      ellipsis_pos.append(None)
      bias_y_base = top_offset + n_shown

    node_positions.append([(x, y) for y in ys])

    if has_bias[layer_idx]:
      bias_positions.append((x, bias_y_base + 0.5))
    else:
      bias_positions.append(None)

  fig_w = 2 * n_layers
  fig_h = max(3, max_eff * 0.6)
  fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor='white')
  ax.set_aspect('equal')
  ax.axis('off')

  # Draw connections: regular -> next regular
  for layer_idx in range(n_layers - 1):
    for (x0, y0) in node_positions[layer_idx]:
      for (x1, y1) in node_positions[layer_idx + 1]:
        ax.plot([x0, x1], [y0, y1], color='gray', lw=0.5, zorder=1)

  # Draw connections: bias -> next regular
  for layer_idx in range(n_layers - 1):
    bp = bias_positions[layer_idx]
    if bp is not None:
      for (x1, y1) in node_positions[layer_idx + 1]:
        ax.plot([bp[0], x1], [bp[1], y1], color='orange', lw=0.5, zorder=1, linestyle='--')

  # Draw regular nodes, text labels, and side names
  layer_labels = (['Input'] +
                  (['Hidden'] if n_layers - 2 == 1 else [f'Hidden {i+1}' for i in range(n_layers - 2)]) +
                  ['Output'])
  for layer_idx, positions in enumerate(node_positions):
    is_input_layer  = (layer_idx == 0)
    is_output_layer = (layer_idx == n_layers - 1)
    color = '#AEC6E8' if is_input_layer else ('#DD8452' if is_output_layer else '#55A868')

    for pos_i, (x, y) in enumerate(positions):
      orig_i = layer_shown[layer_idx][pos_i]
      circle = plt.Circle((x, y), node_radius, color=color, zorder=2)
      ax.add_patch(circle)
      if is_input_layer:
        ax.text(x, y, f'$x_{{{orig_i+1}}}$', ha='center', va='center', fontsize=6, zorder=3)
        if input_names and orig_i < len(input_names):
          ax.text(x - node_radius - 0.05, y, input_names[orig_i],
                  ha='right', va='center', fontsize=6, zorder=3)
      elif is_output_layer:
        ax.text(x, y, f'$\\hat{{y}}_{{{orig_i+1}}}$', ha='center', va='center', fontsize=6, zorder=3)
        if output_names and orig_i < len(output_names):
          ax.text(x + node_radius + 0.05, y, output_names[orig_i],
                  ha='left', va='center', fontsize=6, zorder=3)

    # Hidden layer label
    if not is_input_layer and not is_output_layer:
      bp = bias_positions[layer_idx]
      label_y = (bp[1] if bp is not None else positions[-1][1]) + node_radius + 0.3
      ax.text(positions[0][0], label_y, f"{layer_labels[layer_idx]}\n({layer_sizes[layer_idx]})",
              ha='center', va='bottom', fontsize=8)

  # Draw ellipsis markers (⋮) for truncated layers
  for ep in ellipsis_pos:
    if ep is not None:
      ax.text(ep[0], ep[1], r'$\vdots$', ha='center', va='center', fontsize=14, zorder=3)

  # Draw bias nodes
  for bp in bias_positions:
    if bp is not None:
      circle = plt.Circle(bp, node_radius, color='#FFD700', ec='black', lw=0.8, zorder=2)
      ax.add_patch(circle)
      ax.text(bp[0], bp[1], '+1', ha='center', va='center', fontsize=6, zorder=3)

  ax.set_xlim(-1.5, (n_layers - 1) * x_spacing + node_radius + 0.5)
  ax.set_ylim(-0.5, max_eff + 1.5)

  # Equation in summation form: placed to the right of the output layer
  parts = _build_nn_equation(layer_sizes, input_names=input_names, output_names=output_names, activation=activation)
  n_parts = len(parts)
  x_eq = (n_layers - 1) * x_spacing + node_radius + 0.7

  # Centre equations vertically around the middle of the output nodes
  out_ys = [y for (x, y) in node_positions[-1]]
  mid_y = (out_ys[0] + out_ys[-1]) / 2
  step = 0.7
  top_y = mid_y + (n_parts - 1) / 2 * step
  for idx, part in enumerate(parts):
    y_eq = top_y - idx * step
    ax.text(x_eq, y_eq, r'$' + part + r'$', ha='left', va='center', fontsize=7, zorder=3)

  if n_params is not None:
    param_block = _build_param_block(layer_sizes, n_params)
    ax.text(x_eq, top_y - (n_parts + 0.5) * step, r'$' + param_block + r'$',
            ha='left', va='top', fontsize=7, zorder=3)

