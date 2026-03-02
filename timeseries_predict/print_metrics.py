def print_metrics(outputs, arvs, type=None, indent=0, dt=None):

  msg = ""
  if isinstance(outputs, str):
    outputs = [outputs]
  msg = indent * ' '
  for output, _arv in zip(outputs, arvs):
    msg += f" | {output} ARV = {_arv:7.3f}"

  if dt is not None:
    msg += f" ({type}; {dt:.3f} s)"
  else:
    msg += f" ({type})"

  print(msg)

  return msg
