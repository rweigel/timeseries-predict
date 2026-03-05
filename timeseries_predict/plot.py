def plot(reps, plot_dir, file_base):
  import os
  import numpy
  import matplotlib.pyplot as plt

  def axes():
    plt.figure(figsize=(8.5, 12), facecolor='white')
    gs = plt.gcf().add_gridspec(len(outputs))
    axs = gs.subplots(sharex=True)
    if len(outputs) == 1:
      axs = [axs]
    return axs

  def savefig(plt, file_name):
    os.makedirs(plot_dir, exist_ok=True)

    file_path = os.path.join(plot_dir, f"{file_name}.png")
    plt.savefig(file_path)
    print(f"        Wrote: {os.path.basename(file_path)}")
    plt.close()

  rep_num = 0 # Only plot results for the first repetition for now.
  # TODO: Plot average model results across all repetitions and
  # include error bars to show variability across repetitions.

  kwargs_actual = {'linestyle': '-', 'color': 'black', 'label': 'Actual'}
  kwargs_model = {'linestyle': '-', 'color': 'red', 'label': 'Predicted'}

  actual = reps[rep_num]['data']['test']

  outputs = reps[rep_num]['outputs']
  models = list(reps[rep_num]['models'].keys())

  if file_base.startswith('lno'):
    file_base = ''
  else:
    file_base = f"{file_base}-"

  for model in models:
    axs = axes()
    predicted = reps[rep_num]['models'][model]['predicted']['test']

    time_col = 'timestamp' if 'timestamp' in predicted.columns else 'datetime'
    for i, output in enumerate(outputs):
      arv = reps[rep_num]['models'][model]['metrics']['test'][i]
      kwargs_model['label'] = f"Predicted (ARV={arv:.3f})"
      axs[i].plot(actual[time_col], actual[output], **kwargs_actual)
      axs[i].plot(predicted[time_col], predicted[output], **kwargs_model)
      if i == 0:
        axs[i].set_title(model)
      axs[i].legend(loc="upper right")
      axs[i].set_ylabel(output)
      axs[i].grid(True)
      #datetick()

    plt.tight_layout()
    savefig(plt, f"{model}-timeseries-{file_base}rep_{rep_num}")

    if 'epoch_metrics' in reps[rep_num]['models'][model]:
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

