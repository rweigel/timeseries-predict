def table(stats, directory, **kwargs):
  import os
  import pandas as pd

  columns = ['Removed Input', 'Model']
  for output in kwargs['outputs']:
    columns.append(f"{output} ARV")
    columns.append(f"{output} ARV SE")

  # Check that the number of loo repetitions is consistent across removed inputs
  n_loo = None
  for removed_input in stats.keys():
    if n_loo is not None and len(stats[removed_input]) != n_loo:
      msg = "Number of loo repetitions is not consistent across removed inputs."
      raise ValueError(msg)
    n_loo = len(stats[removed_input])

  for loo_idx in range(n_loo):
    table = []
    for removed_input in stats.keys():
      for model in stats[removed_input][loo_idx][output]:
        row = [removed_input, model]
        for output in stats[removed_input][loo_idx].keys():
          row.append(stats[removed_input][loo_idx][output][model]['mean'])
          row.append(stats[removed_input][loo_idx][output][model]['std'])
        table.append(row)

    # Combine all the summary data into one table
    table = pd.DataFrame(table, columns=columns)

    if n_loo == 1:
      file_name = 'lno.md'
    else:
      idx_width = len(str(n_loo))
      idx_zero_padded = str(loo_idx + 1).zfill(idx_width)
      file_name = f'loo_{idx_zero_padded}.md'

    table_file = os.path.join(directory, file_name)
    table.to_markdown(table_file, index=False, floatfmt=".3f")
    print(f"    Wrote {table_file}")
