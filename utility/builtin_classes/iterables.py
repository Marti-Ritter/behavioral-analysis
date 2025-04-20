from itertools import groupby


def ensure_list(input_object):
    """

    :param input_object:
    :return:
    :rtype: list
    """
    if not isinstance(input_object, list):
        return [input_object]
    else:
        return input_object


def flatten_iterable(iter_obj, seq_types=(list, tuple), inplace=False):
    """
    Flatten a nested iterable into a single list. Taken from https://stackoverflow.com/a/10824086.

    :param iter_obj: A nested iterable object (e.g., list of lists)
    :type iter_obj: list or tuple
    :param seq_types: Types of sequences to flatten (default: (list, tuple))
    :type seq_types: tuple
    :param inplace: Whether to modify the original sequence in place (default: False)
    :type inplace: bool
    :return: Flattened sequence
    :rtype: list

    This function recursively flattens a nested sequence into a single list.
    By default, it returns a modified copy of the input sequence.
    If `inplace` is True, it modifies the input sequence in place and returns None.
    """
    if not inplace:
        iter_obj = iter_obj.copy()
    try:
        for i, x in enumerate(iter_obj):
            while isinstance(x, seq_types):
                iter_obj[i:i + 1] = x
                x = iter_obj[i]
    except IndexError:
        pass
    if inplace:
        return None
    else:
        return iter_obj


def iter_cumsum(input_iter):
    """
    Computes the cumulative sum of an iterable.

    :param input_iter: The iterable to generate the cumulative sum from.
    :type input_iter: collections.Iterable
    :return: A list containing the cumulative sum at each position in the iterable.
    :rtype: list of int or list of float
    """
    cum_sum = []
    total = 0
    for item in input_iter:
        total += item
        cum_sum.append(total)
    return cum_sum


def split_iterable(to_slice, slices):
    slices = [0, *slices, len(to_slice)]
    for i in range(1, len(slices)):
        yield to_slice[slices[i - 1]:slices[i]]


def split_chunks(to_slice, chunk_size):
    slices = range(chunk_size, len(to_slice), chunk_size) if isinstance(chunk_size, int) else iter_cumsum(chunk_size)
    for chunk in split_iterable(to_slice, slices):
        yield chunk


def multi_zip_unequal(*iterables, length_limit_index=None, output_type=list, fill_value=None):
    """
    A utility function to zip multiple iterables of different lengths. The shorter iterables are padded with the
    fill_value to match the length of the iterable indexed by length_limit_index. The output_type parameter determines
    the type of the output. The fill_value parameter determines the value to pad the shorter iterables with.

    A few examples:
    multi_zip_unequal([1, 2, 3], [4, 5, 6, 7], [8, 9], length_limit_index=1) -> [(1, 4, 8), (2, 5, 9), (3, 6, None)]
    multi_zip_unequal([1, 2, 3], [4, 5, 6, 7], [8, 9], length_limit_index=0) -> [(1, 4, 8), (2, 5, 9), (3, 6, None),
    (None, 7, None)]

    :param iterables: The iterables to zip
    :type iterables: collections.Iterable
    :param length_limit_index: The index of the iterable to use as the length limit. If None, the longest iterable is
    used.
    :type length_limit_index: int or None
    :param output_type: The type of the output
    :type output_type: type
    :param fill_value: The value to pad the shorter iterables with. Can be a single value or an iterable of the same
    length as the number of iterables. In the latter case, the fill_value at index i is used to pad the ith iterable.
    :type fill_value: Any
    :return: A zipped iterable of the specified output type
    :rtype: collections.Iterable
    """
    if length_limit_index is None:
        # if None set to index of longest iterable
        length_limit_index = max(range(len(iterables)), key=lambda x: len(iterables[x]))

    # check if fill_value is a single value or an iterable
    if not hasattr(fill_value, "__iter__"):
        fill_value = [fill_value] * len(iterables)

    common_zip = list(zip(*iterables))
    for i in range(len(common_zip), len(iterables[length_limit_index])):
        common_zip.append(
            tuple((fill_value[n] if (i >= len(iterable)) else iterable[i]) for n, iterable in enumerate(iterables)))
    return output_type(common_zip)


def zip_unequal(iter1, iter2, pad1=True, pad2=True, output_type=list, fill_value=None):
    """
    A utility function to zip two iterables of different lengths. The shorter iterable is padded with the fill_value to
    match the length of the longer iterable. The output_type parameter determines the type of the output. The fill_value
    parameter determines the value to pad the shorter iterable with. The pad1 and pad2 parameters determine whether to
    pad the first and second iterable, respectively.

    A few examples:
    zip_unequal([1, 2, 3], [4, 5, 6, 7], pad1=False, pad2=True) -> [(1, 4), (2, 5), (3, 6)]
    zip_unequal([1, 2, 3], [4, 5, 6, 7], pad1=True, pad2=False) -> [(1, 4), (2, 5), (3, 6), (None, 7)]

    :param fill_value: The value to pad the shorter iterable with
    :type fill_value: Any
    :param iter1: The first iterable
    :type iter1: collections.Iterable
    :param iter2: The second iterable
    :type iter2: collections.Iterable
    :param pad1: Whether to pad the first iterable
    :type pad1: bool
    :param pad2: Whether to pad the second iterable
    :type pad2: bool
    :param output_type: The type of the output
    :type output_type: type
    :return: A zipped iterable
    :rtype: collections.Iterable
    """
    common_zip = list(zip(iter1, iter2))
    pad1_zip = [(x, fill_value) for x in iter1[len(iter2):]] if pad1 else []
    pad2_zip = [(fill_value, x) for x in iter2[len(iter1):]] if pad2 else []
    return output_type(common_zip + pad1_zip + pad2_zip)


def filter_consecutive_duplicates(input_iterable, compute_lengths=False):
    """
    Filters consecutive duplicates from an iterable. If compute_lengths is True, the function returns a generator that
    yields tuples of the form (element, count), where count is the number of consecutive occurrences of the element.
    Otherwise, the function returns a generator that yields the unique elements of the iterable.

    :param input_iterable: The input iterable
    :type input_iterable: collections.Iterable
    :param compute_lengths: Whether to compute the lengths of the consecutive duplicates
    :type compute_lengths: bool
    :return: Either a tuple of the form (sequence, lengths) or just the sequence
    :rtype: tuple or collections.Iterable
    """
    if compute_lengths:
        result = [(k, len(list(g))) for k, g in groupby(input_iterable)]
        sequence, lengths = [x[0] for x in result], [x[1] for x in result]
    else:
        result = [k for k, _g in groupby(input_iterable)]
        sequence, lengths = result, None
    return (sequence, lengths) if compute_lengths else sequence
