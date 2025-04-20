import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import mean_squared_error


def pairwise_distances(x, y):
    """
    Calculates the absolute distance between the corresponding points in both arrays in an 1 x n array.

    Args:
    x: numpy array of shape (d, n)
    y: numpy array of shape (d, n)

    Returns:
    dist: numpy array of shape (1, n) containing the absolute distance between the corresponding points in x and y
    """
    dist = np.sum(np.abs(x - y) ** 2, axis=0) ** 0.5
    return dist.reshape(1, -1)


def linear_object_assignment(object_iter1, object_iter2, distance_func, maximize=False, *args, **kwargs):
    """
    Calculates the distances between each possible pair of objects in both iterables and returns the mapping
    {object1: object2} such that the mapping presents the pairing of objects with the smallest cumulative distance
    between all pairs. This is done using the Hungarian algorithm.

    :param object_iter1: A list or tuple of objects
    :type object_iter1: list or tuple
    :param object_iter2: A list or tuple of objects
    :type object_iter2: list or tuple
    :param distance_func: A function that takes two objects and returns a distance between them
    :type distance_func: function
    :param maximize: Whether to maximize or minimize the distance function
    :type maximize: bool
    :param args: Additional arguments to pass to the distance function
    :param kwargs: Additional keyword arguments to pass to the distance function
    :return: A dictionary where the keys are the indices of the objects in object_iter1 and the values are the indices
        of the objects in object_iter2 that form the optimal pairing with respect to the distance between the objects
        in object_iter1 and object_iter2
    :rtype: dict
    """

    # create a matrix to store the pairwise distances between all arrays
    distances = np.zeros((len(object_iter1), len(object_iter2)))

    # calculate the distances between all pairs of arrays
    for i, object1 in enumerate(object_iter1):
        for j, object2 in enumerate(object_iter2):
            distance = distance_func(object1, object2, *args, **kwargs)
            distances[i, j] = distance

    # find the optimal mapping using the Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(distances, maximize=maximize)

    # create a dictionary to store the optimal mapping
    return {i: j for i, j in zip(row_ind, col_ind)}


def linear_array_assignment(array_iter1, array_iter2, distance_func=mean_squared_error, maximize=False,
                            *args, **kwargs):
    """
    This function is a wrapper for linear_object_assignment that takes two iterables of arrays and calculates the
    distance between each pair of arrays using the distance_func. It then returns the optimal mapping of arrays from
    array_iter2 to arrays from array_iter1. The distance_func must take two arrays as input and return a scalar value,
    and defaults to the mean squared error function from sklearn.metrics.
    See the documentation for linear_object_assignment for more details.
    """

    return linear_object_assignment(array_iter1, array_iter2, distance_func=distance_func, maximize=maximize,
                                    *args, **kwargs)


def multiple_linear_object_assignment_with_anchor(anchor_group, *other_groups, distance_func, maximize=False, **kwargs):
    """
    This function takes one "anchor group" and an arbitrary number of other groups of objects. All groups have the same
    number of objects. The function also takes a distance_func that calculates a scalar distance between two objects.

    The function calculates pairwise distances between all objects in the anchor group and each other group. It then
    uses the hungarian algorithm to calculate the optimal alignment of each group's objects to the objects of the anchor
    group. It returns a new set of groups, one for each object in the anchor group, containing that object, as well as
    all objects from the other groups aligned with this object.

    :param anchor_group: A list or tuple of objects
    :type anchor_group: list or tuple
    :param other_groups: An arbitrary number of lists or tuples of objects
    :type other_groups: list or tuple
    :param distance_func: A function that takes two objects and returns a distance between them
    :type distance_func: function
    :param maximize: Whether to maximize or minimize the distance function
    :type maximize: bool
    :param kwargs: Additional keyword arguments to pass to the distance function. Please note that the distance function
        must be able to accept these arguments, and arguments can only be passed as kwargs, as this function is called
        using the *args syntax.
    :return:
    :rtype:
    """
    # Check if all groups have the same number of objects
    n_groups = len(other_groups)
    n_objects = len(anchor_group)
    for i in range(n_groups):
        if len(other_groups[i]) != n_objects:
            raise ValueError("All groups must have the same number of objects")

    # Use linear_object_assignment to match objects from each group to anchor object
    alignments = {}
    for i in range(n_groups):
        alignments[i] = linear_object_assignment(anchor_group, other_groups[i], distance_func=distance_func,
                                                 maximize=maximize, **kwargs)

    # Combine alignments
    combined_alignment = {}
    for i in range(n_objects):
        if i not in combined_alignment.keys():
            combined_alignment[i] = []
        for j in range(n_groups):
            combined_alignment[i].append(alignments[j][i])

    return combined_alignment


def multiple_linear_array_assignment_with_anchor(anchor_array_group, *other_array_groups,
                                                 distance_func=mean_squared_error, maximize=False, **kwargs):
    """
    This function is a wrapper for match_objects_with_anchor that takes two iterables of arrays and calculates the
    distance between each pair of arrays using the distance_func. It then returns the optimal mapping of arrays from
    array_iter2 to arrays from array_iter1. The distance_func must take two arrays as input and return a scalar value,
    and defaults to the mean squared error function from sklearn.metrics.
    See the documentation for match_objects_with_anchor for more details.
    """

    # Use match_objects_with_anchor to match objects from each group to anchor object
    return multiple_linear_object_assignment_with_anchor(anchor_array_group, *other_array_groups,
                                                         distance_func=distance_func, maximize=maximize, **kwargs)


def multi_assign_dataframes(anchor_df_dict, *other_df_dicts, distance_func=mean_squared_error, maximize=False,
                            **kwargs):
    """
    This function takes a dictionary of dataframes, and an arbitrary number of other dictionaries of dataframes. All
    dataframes must have the same shape, and the first dictionary is considered the "anchor" dictionary. The function
    then uses the hungarian algorithm to calculate the optimal alignment of each dataframe in the other dictionaries
    with the dataframes in the anchor dictionary. It returns a new dictionary that maps each key in the anchor
    dictionary to a list of keys from the other dictionaries, where the position of each key in the list corresponds to
    the position of the other dictionary in the function call.

    See the documentation for multiple_linear_object_assignment_with_anchor for more details.

    If you do not want to use dictionaries, you can use the multi_assign_arrays function instead. Or you can create
    dictionaries mapping the id() of each dataframe to the dataframe itself, and use this function.

    :param anchor_df_dict: A dictionary of dataframes
    :type anchor_df_dict: dict of pandas.DataFrame
    :param other_df_dicts: An arbitrary number of dictionaries of dataframes
    :type other_df_dicts: dict of pandas.DataFrame
    :param distance_func: A function that takes two dataframes and returns a distance between them
    :type distance_func: function
    :param maximize: Whether to maximize or minimize the distance function
    :type maximize: bool
    :param kwargs: Additional keyword arguments to pass to the distance function. Please note that the distance function
        must be able to accept these arguments, and arguments can only be passed as kwargs, as this function is called
        using the *args syntax.
    :type kwargs: dict
    :return: A dictionary mapping each key in the anchor dictionary to a list of keys from the other dictionaries
    :rtype: dict of list
    """
    anchor_array_group = [v.values for v in anchor_df_dict.values()]

    other_array_groups = [[v.values for v in df_dict.values()] for df_dict in other_df_dicts]

    alignment_dict = multiple_linear_object_assignment_with_anchor(anchor_array_group, *other_array_groups,
                                                                   distance_func=distance_func, maximize=maximize,
                                                                   **kwargs)

    output_dict = {}
    for anchor_index, alignment_indices in alignment_dict.items():
        output_dict[list(anchor_df_dict)[anchor_index]] = [list(other_df_dicts[i])[j] for i, j in
                                                           enumerate(alignment_indices)]
    return output_dict
