def print_metrics(outputs, arvs, type=None, indent=0, dt=None):

  if isinstance(outputs, str):
    outputs = [outputs]
  print(indent * ' ', end='')
  for output, _arv in zip(outputs, arvs):
    print(f" | {output} ARV = {_arv:7.3f}", end='')

  timimg = ""
  if dt is not None:
    timimg = f" ({type}; {dt:.3f} s)"
  else:
    timimg = f" ({type})"

  print(f" {timimg}")
