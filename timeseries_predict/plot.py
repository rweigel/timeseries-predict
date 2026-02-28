def plot(reps, plot_dir, file_base):
  import os
  import matplotlib.pyplot as plt

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

    predicted = reps[rep_num][model]['predicted']
    plt.figure(figsize=(8.5, 12), facecolor='white')
    gs = plt.gcf().add_gridspec(len(outputs))
    axs = gs.subplots(sharex=True)
    if len(outputs) == 1:
      axs = [axs]
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
    savefig(plt, f"{file_base}_rep_{rep_num}_{model}")

    if 'epochs' in reps[rep_num][model]:
      plt.figure(figsize=(8.5, 8.5), facecolor='white')
      plt.semilogy(reps[rep_num][model]['epochs'])
      # Only allow integer ticks on the x-axis since epochs are discrete
      plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
      plt.grid(True)
      plt.ylabel('ARV')
      plt.xlabel('Epoch')
      plt.tight_layout()
      savefig(plt, f"{file_base}_rep_{rep_num}_{model}_epochs")
