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
    print(f"        Wrote:   {os.path.basename(file_path)}")
    plt.close()

  rep_num = 0 # Only plot results for the first bootstrap repetition for now.
  # TODO: Plot average model results across all bootstrap repetitions and
  # include error bars to show variability across bootstrap repetitions.

  kwargs_actual = {'linestyle': '-', 'color': 'black', 'label': 'Actual'}
  kwargs_model = {'linestyle': '-', 'color': 'red', 'label': None}

  actual = reps[rep_num]['actual']
  outputs = actual.columns[1:] # Exclude timestamp
  models = list(reps[0].keys())
  models.remove('actual')

  for model in models:
    axs = axes()
    predicted = reps[rep_num][model]['predicted']
    for i, output in enumerate(outputs):
      kwargs_model['label'] = model
      axs[i].plot(actual['datetime'], actual[output], **kwargs_actual)
      axs[i].plot(predicted['datetime'], predicted[output], **kwargs_model)
      if i == 0:
        axs[i].legend()
      axs[i].set_ylabel(output)
      axs[i].grid(True)
      #datetick()

    plt.tight_layout()
    savefig(plt, f"{file_base}_rep_{rep_num}_{model}_timeseries")

    if 'epoch_metrics' in reps[rep_num][model]:
      train = numpy.array(reps[rep_num][model]['epoch_metrics']['train'])
      test = numpy.array(reps[rep_num][model]['epoch_metrics']['test'])
      axs = axes()
      for i, output in enumerate(outputs):
        axs[i].semilogy(train[:, i])
        axs[i].semilogy(test[:, i])

        # Only allow integer ticks on the x-axis because epochs are discrete
        axs[i].xaxis.set_major_locator(plt.MaxNLocator(integer=True))

        if i == 0:
          axs[i].legend(['Train', 'Test'])

        axs[i].grid(True)
        axs[i].set_ylabel(f"{output} ARV")
        if i == len(outputs) - 1:
          axs[i].set_xlabel('Epoch')

    plt.tight_layout()
    savefig(plt, f"{file_base}_rep_{rep_num}_{model}_epoch_metrics")
