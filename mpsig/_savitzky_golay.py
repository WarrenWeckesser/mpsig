
from mpmath import mp


def _lstsq(A, b):
    U, S, V = mp.svd(A, compute_uv=True)
    s = [S[k, 0] for k in range(S.rows)]
    s1 = [s[k] for k in range(len(s)) if abs(s[k]) > mp.sqrt(mp.eps)]
    beta = U.transpose() @ mp.matrix(b)
    t1 = [beta[k, 0]/s1[k] for k in range(len(s1))]
    z = mp.matrix(t1 + [mp.zero]*(V.rows - len(t1)))
    return V.transpose() @ z


def savgol_coeffs(window_length, polyorder, deriv=0, delta=None, pos=None,
                  use="conv"):
    """Compute the coefficients for a 1-D Savitzky-Golay FIR filter.

    Parameters
    ----------
    window_length : int
        The length of the filter window (i.e., the number of coefficients).
    polyorder : int
        The order of the polynomial used to fit the samples.
        `polyorder` must be less than `window_length`.
    deriv : int, optional
        The order of the derivative to compute. This must be a
        nonnegative integer. The default is 0, which means to filter
        the data without differentiating.
    delta : float, optional (default is 1)
        The spacing of the samples to which the filter will be applied.
        This is only used if deriv > 0.
    pos : int or None, optional
        If pos is not None, it specifies evaluation position within the
        window. The default is the middle of the window.
    use : str, optional
        Either 'conv' or 'dot'. This argument chooses the order of the
        coefficients. The default is 'conv', which means that the
        coefficients are ordered to be used in a convolution. With
        use='dot', the order is reversed, so the filter is applied by
        dotting the coefficients with the data set.

    Returns
    -------
    coeffs : 1-D ndarray
        The filter coefficients.


    References
    ----------
    A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of Data by
    Simplified Least Squares Procedures. Analytical Chemistry, 1964, 36 (8),
    pp 1627-1639.
    Jianwen Luo, Kui Ying, and Jing Bai. 2005. Savitzky-Golay smoothing and
    differentiation filter for even number data. Signal Process.
    85, 7 (July 2005), 1429-1434.

    Examples
    --------
    >>> from mpmath import mp
    >>> from mpsig import savgol_coeffs

    >>> mp.dps = 25
    >>> savgol_coeffs(5, 2)
    [mpf('-0.08571428571428571428571428571'),
     mpf('0.3428571428571428571428571429'),
     mpf('0.4857142857142857142857142857'),
     mpf('0.3428571428571428571428571429'),
     mpf('-0.08571428571428571428571428571')]
    >>> savgol_coeffs(5, 2, deriv=1)
    [mpf('0.2'), mpf('0.1'), mpf('0.0'), mpf('-0.1'), mpf('-0.2')]

    Note that use='dot' simply reverses the coefficients.

    >>> savgol_coeffs(5, 2, pos=3)
    [mpf('0.2571428571428571428571428571'),
     mpf('0.3714285714285714285714285714'),
     mpf('0.3428571428571428571428571429'),
     mpf('0.1714285714285714285714285714'),
     mpf('-0.1428571428571428571428571429')]
    >>> savgol_coeffs(5, 2, pos=3, use='dot')
    [mpf('-0.1428571428571428571428571429'),
     mpf('0.1714285714285714285714285714'),
     mpf('0.3428571428571428571428571429'),
     mpf('0.3714285714285714285714285714'),
     mpf('0.2571428571428571428571428571')]
    >>> savgol_coeffs(4, 2, pos=3, deriv=1, use='dot')
    [mpf('0.45'), mpf('-0.85'), mpf('-0.65'), mpf('1.05')]

    `x` contains data from the parabola x = t**2, sampled at
    t = -1, 0, 1, 2, 3.  `c` holds the coefficients that will compute the
    derivative at the last position.  When dotted with `x` the result should
    be 6.

    >>> c = savgol_coeffs(5, 2, pos=4, deriv=1, use='dot')
    >>> x = [1, 0, 1, 4, 9]
    >>> sum([c1*x1 for c1, x1 in zip(c, x)])
    mpf('6.0')
    """
    if polyorder >= window_length:
        raise ValueError("polyorder must be less than window_length.")

    halflen, rem = divmod(window_length, 2)

    if pos is None:
        if rem == 0:
            pos = halflen - 0.5
        else:
            pos = halflen

    if not (0 <= pos < window_length):
        raise ValueError("pos must be nonnegative and less than "
                         "window_length.")

    if use not in ['conv', 'dot']:
        raise ValueError("`use` must be 'conv' or 'dot'")

    with mp.extradps(mp.dps):

        if deriv > polyorder:
            coeffs = [mp.zero]*window_length
            return coeffs

        if delta is None:
            delta = mp.one
        else:
            delta = mp.mpf(delta)

        # Form the design matrix A. The columns of A are powers of the integers
        # from -pos to window_length - pos - 1. The powers (i.e., rows) range
        # from 0 to polyorder. (That is, A is a vandermonde matrix, but not
        # necessarily square.)
        x = [k - pos for k in range(window_length)]

        if use == "conv":
            # Reverse so that result can be used in a convolution.
            x = x[::-1]

        order = list(range(polyorder + 1))

        # A = x ** order (element-wise)
        A = mp.matrix(polyorder + 1, window_length)
        for i in range(polyorder + 1):
            for j in range(window_length):
                A[i, j] = mp.mpf(x[j])**mp.mpf(order[i])

        # y determines which order derivative is returned.
        y = [mp.zero]*(polyorder + 1)

        # The coefficient assigned to y[deriv] scales the result to take into
        # account the order of the derivative and the sample spacing.
        y[deriv] = mp.factorial(deriv) / (delta ** deriv)

        # Find the least-squares solution of A*c = y
        c = _lstsq(A, y)
        coeffs = [c[k, 0] for k in range(c.rows)]
        if rem == 1 and pos == halflen and deriv % 2 == 1:
            # window_length is odd, pos is the middle of the
            # window, and deriv is odd.  Under these conditions,
            # the middle coefficient must be 0.
            coeffs[halflen] = mp.zero

        return coeffs
