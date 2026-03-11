def table(stats, directory, **kwargs):
  import os
  import pandas

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

  def build_row(metrics, model, row_prefix, label):
    row = row_prefix + [label]
    for output_stats in metrics.values():
      row.append(output_stats[model][label]['mean'])
      row.append(output_stats[model][label]['se'])
    return row

  for loo_idx in range(n_loo):
    table_all = []
    for removed_input in stats.keys():
      for model in stats[removed_input][loo_idx][output]:
        if n_loo == 1:
          row_prefix = [model]
        else:
          row_prefix = [removed_input, model]

        metrics = stats[removed_input][loo_idx]

        row_train = build_row(metrics, model, row_prefix, 'train')
        row_test = build_row(metrics, model, row_prefix, 'test')

        table_all.append(row_train)
        table_all.append(row_test)

        # Optional aggregated rows if available
        if 'train*' in stats[removed_input][loo_idx][output][model]:
          row_train_star = build_row(metrics, model, row_prefix, 'train*')
          table_all.append(row_train_star)
        if 'test*' in stats[removed_input][loo_idx][output][model]:
          row_test_star = build_row(metrics, model, row_prefix, 'test*')
          table_all.append(row_test_star)

    # Combine all the summary data into one table
    table_all = pandas.DataFrame(table_all, columns=columns)

    # Drop empty columns (for example SE if std is not available)
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

    print(f"    Wrote {table_file}")
