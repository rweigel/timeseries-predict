def plot(reps, plot_dir, file_base):
  import os
  import numpy
  import matplotlib
  import matplotlib.pyplot as plt

  matplotlib.rcParams['agg.path.chunksize'] = 10000

  def axes(plot_type=None):
    figsize = (8.5, 12)
    sharex = True
    if plot_type == 'scatter':
      figsize = (8.5, 12)
      sharex = False

    plt.figure(figsize=figsize, facecolor='white')

    gs = plt.gcf().add_gridspec(len(outputs))
    axs = gs.subplots(sharex=sharex)

    if len(outputs) == 1:
      axs = [axs]

    return axs

  def savefig(plt, file_name, format=["png"], dpi=300):
    if isinstance(format, str):
      format = [format]
    if len(format) == 1:
      format = format[0]
    else:
      for fmt in format:
        savefig(plt, file_name, format=fmt, dpi=dpi)
      return

    file_path = os.path.join(plot_dir, f"{file_name}.{format}")
    plt.savefig(file_path, bbox_inches="tight", dpi=dpi)
    print(f"        Wrote: {os.path.basename(file_path)}")
    plt.close()

  def scatter(rep, rep_num, model, cat, file_base):
    actual = reps[rep_num]['data'][cat]
    predicted = reps[rep_num]['models'][model]['predicted'][cat]

    # Predicted vs actual scatter plot
    axs = axes('scatter')
    for i, output in enumerate(outputs):
      x = actual[output]
      y = predicted[output]

      # Set limits to be symmetric around zero and encompass all points
      limit_min = min(x.min(), y.min())
      limit_max = max(x.max(), y.max())
      limit_min = -max(abs(limit_min), abs(limit_max))
      limit_max = max(abs(limit_min), abs(limit_max))
      axs[i].set_xlim(limit_min, limit_max)
      axs[i].set_ylim(limit_min, limit_max)
      axs[i].plot([limit_min, limit_max], [limit_min, limit_max], linestyle='--', lw=0.5, color='gray')
      axs[i].set_aspect('equal')

      # Horizontal and vertical lines
      axs[i].plot([limit_min, limit_max], [0, 0], linestyle='-', lw=1, color='black')
      axs[i].plot([0, 0], [limit_min, limit_max], linestyle='-', lw=1, color='black')

      axs[i].scatter(x, y, marker='.', s=1, color='black', label='Predicted')

      if i == 0:
        axs[i].set_title(f"{model} | {cat} data")
      axs[i].set_ylabel('Predicted')
      axs[i].grid(True)
      if i == len(outputs) - 1:
        axs[i].set_xlabel('Actual')

    plt.tight_layout()
    savefig(plt, f"{model}-scatter-{cat}-{file_base}rep_{rep_num}")

  def timeseries(rep, rep_num, model, cat, file_base):
    actual = reps[rep_num]['data'][cat]
    predicted = reps[rep_num]['models'][model]['predicted'][cat]

    kwargs_actual = {'linestyle': '-', 'color': 'black', 'label': 'Actual'}
    kwargs_model = {'linestyle': '-', 'color': 'red', 'label': 'Predicted'}

    axs = axes()
    time_col = 'timestamp' if 'timestamp' in predicted.columns else 'datetime'
    for i, output in enumerate(outputs):
      arv = reps[rep_num]['models'][model]['metrics'][cat][i]
      kwargs_model['label'] = f"Predicted (ARV={arv:.3f})"
      axs[i].plot(actual[time_col], actual[output], **kwargs_actual)
      axs[i].plot(predicted[time_col], predicted[output], **kwargs_model)
      if i == 0:
        axs[i].set_title(f"{model} | {cat} data")
      axs[i].legend(loc="upper right")
      axs[i].set_ylabel(output)
      axs[i].grid(True)
      #datetick()

    plt.tight_layout()
    savefig(plt, f"{model}-timeseries-{cat}-{file_base}rep_{rep_num}")

  def epoch_metrics(rep, rep_num, model, file_base):
    train = numpy.array(reps[rep_num]['models'][model]['epoch_metrics']['train'])
    test = numpy.array(reps[rep_num]['models'][model]['epoch_metrics']['test'])

    style = '-'
    if train.shape[0] == 1:
      # Only one epoch. Need to use a marker so something is rendered.
      style = '.'

    axs = axes()
    for i, output in enumerate(outputs):
      axs[i].plot(train[:, i], style)
      axs[i].plot(test[:, i], style)

      # Only allow integer ticks on the x-axis because epochs are discrete
      axs[i].xaxis.set_major_locator(plt.MaxNLocator(integer=True))

      if i == 0:
        axs[i].set_title(model)
        axs[i].legend(['Train', 'Test'])

      axs[i].grid(True)
      axs[i].set_ylabel(f"{output} ARV")
      if i == len(outputs) - 1:
        axs[i].set_xlabel('Epoch')

    plt.tight_layout()
    savefig(plt, f"{model}-epoch-metrics-{file_base}rep_{rep_num}")

  def _model_diagram(model, model_name, input_names, output_names):
    from .plot_model import plot_model
    plot_model(model, input_names=reps[0]['inputs'], output_names=reps[0]['outputs'])
    savefig(plt, f"{model_name}-model" , format=["png", "svg"])


  rep_num = 0 # Only plot results for the first repetition for now.
  # TODO: Plot average model results across all repetitions and
  # include error bars to show variability across repetitions.


  outputs = reps[rep_num]['outputs']
  model_names = list(reps[rep_num]['models'].keys())

  if file_base.startswith('lno'):
    file_base = ''
  else:
    file_base = f"{file_base}-"

  for model_name in model_names:
    if 'model' in reps[0]['models'][model_name]:
      model = reps[0]['models'][model_name]['model']
      if isinstance(model, list):
        model = model[0]
      _model_diagram(model, model_name, reps[0]['inputs'], reps[0]['outputs'])

    # Epoch metrics plot (if available)
    if 'epoch_metrics' in reps[rep_num]['models'][model_name]:
      epoch_metrics(reps, rep_num, model_name, file_base)

    for cat in ['train', 'test']:
      scatter(reps, rep_num, model_name, cat, file_base)
      timeseries(reps, rep_num, model_name, cat, file_base)
