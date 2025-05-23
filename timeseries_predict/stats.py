"""
Bootstrap Statistics Calculation Script

This script calculates statistical evaluation metrics for multiple models over multiple 
bootstrap repetitions. Specifically, it computes the Average Relative Variance (ARV) 
for each model and output variable, summarizing the results with mean and standard 
deviation across bootstrap runs.

"""
import numpy as np
from .arv import arv

def stats(boots, outputs):
    """
    Compute mean and standard deviation of ARV for each model-output pair across bootstrap runs.

    Parameters:
    ----------
    boots : list of dict
        A list of dictionaries, where each dictionary contains actual and predicted values for 
        multiple models from a single bootstrap run. Each dictionary must include an 'actual' 
        key and one or more model keys (e.g., 'ols', 'nn_miso', 'nn_mimo').

    outputs : list of str
        A list of output variable names (column names) to evaluate.

    Returns:
    -------
    dict
        A nested dictionary structured as:
        stats[output][model] = {'mean': float, 'std': float or None}
        'std' is only computed if there are more than 19 bootstrap samples.
    """
    arvs = {}
    for boot in boots:  # Loop over each bootstrap repetition
        # boot has elements of {actual: df, ols: df, nn_miso: df, nn_mimo: df}
        models = list(boot.keys())
        models.remove('actual')
        for model in models:
            if model not in arvs:
                arvs[model] = {}
            for output in outputs:
                if output not in arvs[model]:
                    arvs[model][output] = []
                arvs[model][output].append(arv(boot['actual'][output], boot[model]['predicted'][output]))

    stats = {}
    for output in arvs[model]:
        stats[output] = {}
        for model in arvs:
            stats[output][model] = {}
            stats[output][model]['mean'] = np.mean(arvs[model][output])
            stats[output][model]['std'] = None
            if len(arvs[model][output]) > 19:
                # Only compute std used for uncertainty if number of bootstrap samples
                # is greater than 19.
                stats[output][model]['std'] = np.std(arvs[model][output], ddof=1)

    return stats
