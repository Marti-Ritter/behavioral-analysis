from functools import reduce

from ..general import output_formatter


def get_list_intersection(list_of_sets):
    """

    :param list_of_sets:
    :type list_of_sets: list
    :return:
    :rtype: set
    """
    return list_of_sets.pop().intersection(*list_of_sets)


def get_list_union(list_of_sets):
    """

    :param list_of_sets:
    :type list_of_sets: list
    :return:
    :rtype: set
    """
    return list_of_sets.pop().union(*list_of_sets)


def unified_sets(*args):
    """
    Unifies a list of sets, i.e. merges sets that have an intersection.

    :param args: A list of sets
    :type args: list
    :return: A list of unified sets
    :rtype: list
    """
    unified_list = []

    for input_set in args:
        current_set = set(input_set)

        new_unified_list = []
        while unified_list:
            unified_set = unified_list.pop()
            if current_set.intersection(unified_set):
                current_set.update(unified_set)
            else:
                new_unified_list.append(unified_set)
        unified_list = new_unified_list + [current_set]
    return unified_list


def check_disjoint_sets(*args):
    """
    Checks whether a list of sets is fully separated, i.e. whether no set has an intersection with any other set.

    :param args: A list of sets
    :type args: list
    :return: A boolean indicating whether the sets are disjoint
    :rtype: bool
    """
    args_union = set(reduce(set.union, args))
    args_length_sum = sum(len(arg) for arg in args)
    return len(args_union) == args_length_sum


def max_disjoint_set_old(input_sets, initial_set=None, return_alternatives=False):
    """
    Finds the largest disjoint set from a list of sets by recursively combining all disjoint sets in input_sets with the
    initial_set and returning the largest set, along with the remainder of the input_sets.
    If there are multiple sets with the same size, the first one is returned.
    Any alternatives are returned if return_alternatives is True.

    :param input_sets: A list of sets to find the largest disjoint set from, i.e. the largest combination of sets that
    are disjoint.
    :type input_sets: list of set
    :param initial_set: An initial set to start the search from. Optional.
    :type initial_set: set or None
    :param return_alternatives: A boolean indicating whether to return the alternatives.
    :type return_alternatives: bool
    :return: The largest disjoint set from the input sets, starting from the initial set, along with the remainder of
    the input_sets.
    :rtype: (set, list of set) or (set, list of set, list of (set, list of set))
    """
    initial_set = set() if initial_set is None else initial_set

    disjoint_sets = [s for s in input_sets if s.isdisjoint(initial_set)]
    remainder = []
    alternatives = []

    # return condition
    if not disjoint_sets:
        return output_formatter((initial_set, input_sets, alternatives), (True, True, return_alternatives),
                                unpack_single=True)

    # recursive call
    max_set = initial_set
    for disjoint_set in disjoint_sets:
        new_initial_set = initial_set.union(disjoint_set)
        new_input_sets = [s for s in input_sets if s != disjoint_set]
        (new_max_set, new_remainder) = max_disjoint_set_old(new_input_sets, new_initial_set)
        if len(new_max_set) > len(max_set):
            max_set, remainder = new_max_set, new_remainder
            alternatives = []
        elif (len(new_max_set) == len(max_set)) and (
                (new_max_set, new_remainder) not in alternatives) and (
                new_max_set != max_set):
            alternatives.append((new_max_set, new_remainder))

    return output_formatter((max_set, remainder, alternatives), (True, True, return_alternatives),
                            unpack_single=True)


def max_disjoint_set(input_sets, initial_set=None, return_components=False, return_alternatives=False):
    """
    Finds the largest disjoint set from a list of sets by recursively combining all disjoint sets in input_sets with the
    initial_set and returning the largest set. If there are multiple sets with the same size, the first one is returned.
    Any alternatives are returned if return_alternatives is True.

    :param input_sets: A list of sets to find the largest disjoint set from, i.e. the largest combination of sets that
    are disjoint.
    :type input_sets: list of set
    :param initial_set: An initial set to start the search from. Optional.
    :type initial_set: set or None
    :param return_components: A boolean indicating whether to return the components used to in the largest disjoint set.
    :type return_components: bool
    :param return_alternatives: A boolean indicating whether to return the alternatives.
    :type return_alternatives: bool
    :return: The largest disjoint set from the input sets, starting from the initial set.
    :rtype: set or (set, list of set)
    """
    input_sets = sorted(input_sets, key=len, reverse=True)  # sort by length, the largest first
    initial_set = set() if initial_set is None else initial_set
    disjoint_sets = [s for s in input_sets if s.isdisjoint(initial_set)]

    # default values
    max_set, components, alternatives = initial_set, [], []

    # return condition
    if not disjoint_sets:
        return output_formatter((max_set, components, alternatives), (True, return_components, return_alternatives),
                                unpack_single=True)

    # recursive call
    new_input_sets = disjoint_sets
    for disjoint_set in disjoint_sets:
        new_initial_set = initial_set.union(disjoint_set)
        # The number of available disjoint sets is reduced by any already used sets, to avoid visiting paths twice
        new_input_sets = [s for s in new_input_sets if s != disjoint_set]
        (new_max_set, new_components, new_alternatives) = max_disjoint_set(new_input_sets, new_initial_set,
                                                                           return_components=True,
                                                                           return_alternatives=True)
        if len(new_max_set) > len(max_set):
            max_set, components, alternatives = new_max_set, [disjoint_set] + new_components, new_alternatives
        elif (len(new_max_set) == len(max_set)) and (new_max_set not in [s for s, c in alternatives]) and (
                new_max_set != max_set):
            alternatives.append((new_max_set, new_components))
            alternatives.extend(new_alternatives)
            alternatives = [(s, [disjoint_set] + c) for (s, c) in alternatives]

    # recursion output
    alternatives = alternatives if return_components else [s for s, c in alternatives]
    return output_formatter((max_set, components, alternatives), (True, return_components, return_alternatives),
                            unpack_single=True)


def condense_sets_to_mds(input_sets, allow_alternatives=False):
    """
    Reduces a list of sets to a list of maximal disjoint sets, also referred to as MDS. The MDS are found by
    applying the max_disjoint_set function to the list, removing any sets included into a MDS until all sets have been
    used up. The MDS are returned as a list of lists of sets.
    If allow_alternatives is True, the function will return an additional list of lists of sets, containing all
    alternative solutions per MDS.

    :param input_sets: A list of sets
    :type input_sets: list of set
    :param allow_alternatives: A boolean indicating whether to allow alternative solutions
    :type allow_alternatives: bool
    :return: A list of maximal disjoint sets
    :rtype: list of (list of set)
    """
    available_sets = input_sets
    mds_merges = []
    alternatives_found = []

    while available_sets:
        initial_set = available_sets[0]

        res = max_disjoint_set(available_sets, initial_set=initial_set, return_components=True,
                               return_alternatives=True)
        new_merge = res[1] + [initial_set]
        # TODO: Change this so that every set is considered during the recursion, check if solutions do not overlap!
        available_sets = [s for s in available_sets if s not in new_merge]

        if res[2] and not allow_alternatives:
            raise ValueError("Alternative solutions were found, but allow_alternatives is False.")
        else:
            alt_mds = []
            for alt in res[2]:
                alt_mds.append(alt[1] + [initial_set])
            alternatives_found.append(alt_mds)
        mds_merges.append(new_merge)
    return output_formatter((mds_merges, alternatives_found), (True, allow_alternatives), unpack_single=True)
