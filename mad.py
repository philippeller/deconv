import numpy as np
def median_absolute_deviation(a, axis=None):
    """Compute the median absolute deviation.

    Returns the median absolute deviation (MAD) of the array elements.
    The MAD is defined as ``median(abs(a - median(a)))``.

    Parameters
    ----------
    a : array-like
        Input array or object that can be converted to an array.
    axis : int, optional
        Axis along which the medians are computed. The default (axis=None)
        is to compute the median along a flattened version of the array.

    Returns
    -------
    median_absolute_deviation : ndarray
        A new array holding the result. If the input contains
        integers, or floats of smaller precision than 64, then the output
        data-type is float64.  Otherwise, the output data-type is the same
        as that of the input.

    Examples
    --------

    This will generate random variates from a Gaussian distribution and return
    the median absolute deviation for that distribution::

        >>> from astropy.stats import median_absolute_deviation
        >>> from numpy.random import randn
        >>> randvar = randn(10000)
        >>> mad = median_absolute_deviation(randvar)

    See Also
    --------
    numpy.median

    """

    # Check if the array has a mask and if so use np.ma.median
    # See https://github.com/numpy/numpy/issues/7330 why using np.ma.median
    # for normal arrays should not be done (summary: np.ma.median always
    # returns an masked array even if the result should be scalar). (#4658)
    if isinstance(a, np.ma.MaskedArray):
        func = np.ma.median
    else:
        func = np.median

    a = np.asanyarray(a)

    a_median = func(a, axis=axis)

    # re-broadcast the output median array to subtract it
    if axis is not None:
        a_median = np.expand_dims(a_median, axis=axis)

    # calculated the median average deviation
    return func(np.abs(a - a_median), axis=axis)

