import numpy as np
import pandas as pd

def arv(A, P):
    if isinstance(A, pd.DataFrame):
        A = A.to_numpy()
    if isinstance(P, pd.DataFrame):
        P = P.to_numpy()

    if A.ndim == 1:
        A, P = np.expand_dims(A, axis=1), np.expand_dims(P, axis=1)

    arvs = []
    # TODO: Outer loop not needed.
    for i in range(A.shape[1]):
        var_A = np.var(A[:, i])
        numerator = np.mean(np.power(A[:, i] - P[:, i], 2), axis=0)
        arv = numerator / var_A if var_A > 0 else np.nan
        arvs.append(arv)

    return arvs if len(arvs) > 1 else arvs[0]
