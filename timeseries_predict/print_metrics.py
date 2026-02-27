def print_metrics(outputs, arvs, total_loss):
  if isinstance(arvs, list):
    for output, _arv in zip(outputs, arvs):
        print(f" | {output} ARV = {_arv:.3f}", end='')
    print(f" | loss = {total_loss:.4f}")
  else:
    print(f" | {outputs} ARV = {arvs:.3f} | loss = {total_loss:.4f}")
