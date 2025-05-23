"""
ARV Metric Calculation Script

Provides a function to compute the Average Relative Variance (ARV),
a normalized error metric used to evaluate the accuracy of predicted values 
against actual values. The ARV is particularly useful in time series analysis 
and regression tasks, offering a scale-independent measure of prediction quality.

"""
import numpy as np
import pandas as pd

def arv(A, P):
    """
    Calculate the Average Relative Variance (ARV) between actual and predicted values.

    The ARV is a normalized error metric used to evaluate prediction performance. 
    It is computed as the ratio of the mean squared error (MSE) between actual and 
    predicted values to the variance of the actual values. An ARV of 0 indicates 
    perfect predictions, while an ARV of 1 indicates that the model performs no 
    better than predicting the mean of the actual values.

    Parameters:
    ----------
    A : array-like (numpy.ndarray or pandas.DataFrame)
        Actual values. Can be a 1D or 2D array-like structure.
    
    P : array-like (numpy.ndarray or pandas.DataFrame)
        Predicted values. Must have the same shape as `A`.

    Returns:
    -------
    float or list of floats
        The ARV value(s). Returns a single float if input is 1D, or a list of floats 
        corresponding to each column if input is 2D.

    Notes:
    -----
    - If the variance of a column in `A` is zero, the corresponding ARV is returned as NaN.
    - This function converts pandas DataFrames to NumPy arrays internally.
    
    """
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
