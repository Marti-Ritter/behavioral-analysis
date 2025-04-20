def find_linear_zero(a, b):
    """Find the zero location of a linear function based on its parameters

    :param a: slope
    :type a: float
    :param b: y-axis intersection
    :type b: float
    :return: Location where the linear function cuts the x-axis
    :rtype: float
    """
    if a == 0:
        raise ValueError("Cannot find zero of a x-independent linear function.")
    else:
        return -b / a
