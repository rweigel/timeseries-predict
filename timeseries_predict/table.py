"""
Markdown Summary Table Generator

This script generates markdown-formatted tables summarizing the Average Relative 
Variance (ARV) results for different models across leave-one-out (LOO) repetitions 
and input removal scenarios. The tables include mean ARV and standard error (SE) 
per output variable for each model.

"""
import os
import pandas as pd

def table(stats, directory, **kwargs):
    """
    Generate markdown tables summarizing ARV statistics for each LOO repetition.

    For each leave-one-out (LOO) repetition, this function compiles a table that 
    includes the mean and standard error (if available) of ARV values for each 
    model and output variable. The tables are saved in markdown format to the 
    specified directory.

    Parameters:
    ----------
    stats : dict
        A nested dictionary of statistics structured as:
        stats[removed_input][loo_idx][output][model] = {'mean': float, 'std': float or None}

    directory : str
        The output directory where the markdown files will be saved.

    kwargs : dict
        Must include 'outputs' (list of str), the names of output variables to be included in the table.

    Raises:
    -------
    ValueError:
        If the number of LOO repetitions is inconsistent across removed inputs.

    """
    columns = ['Removed Input', 'Model']
    for output in kwargs['outputs']:
        columns.append(f"{output} ARV")
        columns.append(f"{output} ARV SE")

    # Check that the number of loo repetitions is consistent across removed inputs
    n_loo = None
    for removed_input in stats.keys():
        if n_loo is not None and len(stats[removed_input]) != n_loo:
            raise ValueError("Number of loo repetitions is not consistent across removed inputs.")
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

        table_file = os.path.join(directory, f'loo_{loo_idx}.md')
        table.to_markdown(table_file, index=False, floatfmt=".3f")
        print(f"    Wrote {table_file}")

