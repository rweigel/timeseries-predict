def table(stats, directory, **kwargs):
  import os
  import pandas as pd

  # Check that the number of loo repetitions is consistent across removed inputs
  n_loo = None
  for removed_input in stats.keys():
    if n_loo is not None and len(stats[removed_input]) != n_loo:
      msg = "Number of loo repetitions is not consistent across removed inputs."
      raise ValueError(msg)
    n_loo = len(stats[removed_input])

  if n_loo == 1:
    columns = ['Model', 'Dataset']
  else:
    columns = ['Removed Input', 'Model', 'Dataset']

  for output in kwargs['outputs']:
    columns.append(f"{output} ARV")
    columns.append(f"{output} ARV SE")

  for loo_idx in range(n_loo):
    table_train = []
    table_test = []
    table_all = []
    for removed_input in stats.keys():
      for model in stats[removed_input][loo_idx][output]:
        if n_loo == 1:
          row_train = [model, 'train']
          row_test = [model, 'test']
        else:
          row_train = [removed_input, model]
          row_test = [removed_input, model]
        for output in stats[removed_input][loo_idx].keys():
          row_train.append(stats[removed_input][loo_idx][output][model]['train']['mean'])
          row_train.append(stats[removed_input][loo_idx][output][model]['train']['se'])
          row_test.append(stats[removed_input][loo_idx][output][model]['test']['mean'])
          row_test.append(stats[removed_input][loo_idx][output][model]['test']['se'])

        table_test.append(row_test)
        table_train.append(row_train)
        table_all.append(row_train)
        table_all.append(row_test)

    # Combine all the summary data into one table
    table_test = pd.DataFrame(table_test, columns=columns)
    table_train = pd.DataFrame(table_train, columns=columns)
    table_all = pd.DataFrame(table_all, columns=columns)

    # Drop empty columns (for example SE if std is not available)
    table_train = table_train.dropna(axis=1, how='all')
    table_test = table_test.dropna(axis=1, how='all')
    table_all = table_all.dropna(axis=1, how='all')

    if n_loo == 1:
      file_name = 'lno.md'
    else:
      idx_width = len(str(n_loo))
      idx_zero_padded = str(loo_idx + 1).zfill(idx_width)
      file_name = f'loo_{idx_zero_padded}.md'

    table_file = os.path.join(directory, file_name)
    with open(table_file, 'w') as f:
      f.write('# Training and Test Combined\n')
      table_all.to_markdown(f, index=False, floatfmt=".3f")
      f.write('\n\n')
      f.write('# Training Results\n')
      table_train.to_markdown(f, index=False, floatfmt=".3f")
      f.write('\n\n')
      f.write('# Test Results\n')
      table_test.to_markdown(f, index=False, floatfmt=".3f")

    print(f"    Wrote {table_file}")
