def plot(boots, run_dir, plot_subdir, file_base):
  import os
  import matplotlib.pyplot as plt

  def savefig(plt, file_name):
    file_subpath = os.path.join(plot_subdir, f"{file_name}.png")
    file_path = os.path.join(run_dir, file_subpath)
    if not os.path.exists(os.path.dirname(file_path)):
      os.makedirs(os.path.dirname(file_path))

    plt.savefig(f"{file_path}")
    print(f"        Wrote:   {file_subpath}")
    plt.close()

  boot_num = 0 # Only plot results for the first bootstrap repetition for now.
  # TODO: Plot average model results across all bootstrap repetitions and
  # include error bars to show variability across bootstrap repetitions.

  kwargs_actual = {'linestyle': '-', 'color': 'black', 'label': 'Actual'}
  kwargs_model = {'linestyle': '-', 'color': 'red', 'label': None}

  actual = boots[boot_num]['actual']
  outputs = actual.columns[1:] # Exclude timestamp

  models = list(boots[0].keys())
  models.remove('actual')
  for model in models:
    predicted = boots[boot_num][model]['predicted']
    plt.figure(figsize=(8.5, 12), facecolor='white')
    gs = plt.gcf().add_gridspec(len(outputs))
    axs = gs.subplots(sharex=True)
    for i, output in enumerate(outputs):
      kwargs_model['label'] = model
      axs[i].plot(actual['timestamp'], actual[output], **kwargs_actual)
      axs[i].plot(predicted['timestamp'], predicted[output], **kwargs_model)
      if i == 0:
        axs[i].legend()
      axs[i].set_ylabel(output)
      axs[i].grid(True)
      #datetick()

    plt.tight_layout()
    savefig(plt, f"{file_base}_boot_{boot_num}_{model}")

    if 'epochs' in boots[boot_num][model]:
      plt.figure(figsize=(8.5, 8.5), facecolor='white')
      plt.semilogy(boots[boot_num][model]['epochs'])
      # Only allow integer ticks on the x-axis since epochs are discrete
      plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
      plt.grid(True)
      plt.ylabel('ARV')
      plt.xlabel('Epoch')
      plt.tight_layout()
      savefig(plt, f"{file_base}_boot_{boot_num}_{model}_epochs")
