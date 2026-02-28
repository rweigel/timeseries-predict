def print_metrics(outputs, arvs, total_loss):
  for output, _arv in zip(outputs, arvs):
      print(f" | {output} ARV = {_arv:.3f}", end='')
  print(f" | loss = {total_loss:.4f}")
