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
    for i in range(A.shape[1]):
        var_A = np.var(A[:, i])
        numerator = np.mean(np.power(A[:, i] - P[:, i], 2), axis=0)
        _arv = np.nan
        if var_A > 0:
            _arv = numerator / var_A

        arvs.append(_arv)

    return arvs
