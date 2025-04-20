from matplotlib_venn import venn2, venn3


def boolean_series_venn(series_list, labels=None):
    """Generates a Venn-plot of 2 or three boolean series

    This function creates a Venn-plot showing the coincidence of True entries in the input Series.
    The index of the Series is used to determine coincidence.

    :param series_list: List or tuple of pd.Series
    :type series_list: list or tuple
    :param labels: Labels for each of the input Series
    :type labels: List or tuple
    :return: matplotlib_venn.VennDiagram showing the coincidence between the boolean Series
    :rtype: matplotlib_venn._common.VennDiagram
    """
    if len(series_list) == 2:
        venn_func = venn2
    elif len(series_list) == 3:
        venn_func = venn3
    else:
        raise ValueError("Can't process more than 3 series! Length of input is {}.".format(len(series_list)))
    set_list = [set(s.index[s.astype(bool)]) for s in series_list]
    return venn_func(subsets=set_list, set_labels=labels)
