from ..general import prepend_append_calls_to_object, prepend_append_calls_to_function

from .funcs import apply_filtered_parameters
from .iterables import ensure_list


# https://stackoverflow.com/a/682052
class AddedCallsMeta(type):
    # https://stackoverflow.com/a/25191150
    @classmethod
    def __prepare__(mcs, name, bases, **kwargs):
        return super(AddedCallsMeta, mcs).__prepare__(name, bases, **kwargs)

    def __new__(mcs, name, bases, namespace, methods_to_modify=None, calls_before=tuple(), calls_after=tuple()):
        new_object = super(AddedCallsMeta, mcs).__new__(mcs, name, bases, namespace)
        for method in methods_to_modify:
            setattr(new_object, method, prepend_append_calls_to_function(getattr(new_object, method),
                                                                         calls_before, calls_after))
        return new_object

    def __init__(cls, name, bases, namespace):
        super(AddedCallsMeta, cls).__init__(name, bases, namespace)


class ListWithAddedCalls(list):
    def __init__(self, *args, methods_to_modify=None, methods_to_exclude=None, calls_before=tuple(),
                 calls_after=tuple(), **kwargs):
        """

        :param args:
        :type args:
        :param methods_to_modify:
        :type methods_to_modify: list of string or tuple of string
        :param methods_to_exclude:
        :type methods_to_exclude: list of string or tuple of string
        :param calls_before:
        :type calls_before: list of callable or tuple of callable
        :param calls_after:
        :type calls_after: list of callable or tuple of callable
        :param kwargs:
        :type kwargs:
        """
        super(ListWithAddedCalls, self).__init__(*args, **kwargs)
        prepend_append_calls_to_object(self, methods_to_modify, methods_to_exclude, calls_before, calls_after)


class DictWithAddedCalls(dict):
    def __init__(self, *args, methods_to_modify=None, methods_to_exclude=None, calls_before=tuple(),
                 calls_after=tuple(), **kwargs):
        """

        :param args:
        :type args:
        :param methods_to_modify:
        :type methods_to_modify: list of string or tuple of string
        :param methods_to_exclude:
        :type methods_to_exclude: list of string or tuple of string
        :param calls_before:
        :type calls_before: list of callable or tuple of callable
        :param calls_after:
        :type calls_after: list of callable or tuple of callable
        :param kwargs:
        :type kwargs:
        """
        super(DictWithAddedCalls, self).__init__(*args, **kwargs)
        prepend_append_calls_to_object(self, methods_to_modify, methods_to_exclude, calls_before, calls_after)


class SelfExtendingDict(dict):
    """
    A dict subclass that allows the attachment of another dictionary containing functions or lists of functions as
    values. If there is an attempt to access a non-existing key in the self-extending dict it will check if that key
    exists in the attached dictionary, and if yes, will call the function(s) behind that key in an attempt to calculate
    the non-existing key in this dict. For lists it will iterate through the functions until one call succeeds.
    Once the non-existing key is successfully calculated it will be permanently included in the dict.
    """
    def __init__(self, extension_funcs=None, *args, **kwargs):
        super(SelfExtendingDict, self).__init__(*args, **kwargs)
        self._extension_funcs = extension_funcs if extension_funcs is not None else {}

    def _attempt_extension(self, item):
        extension_successful = False
        for extension_func in ensure_list(self.extension_funcs[item]):
            try:
                self[item] = apply_filtered_parameters(extension_func, **self)
                extension_successful = True
                break
            except (TypeError, KeyError, ValueError):  # Just a guess at what can go wrong in the supplied functions
                continue
        if not extension_successful:
            raise KeyError("Extension on key {} attempted, but unsuccessful".format(item))

    def __getitem__(self, item):
        if item not in self:
            if item in self.extension_funcs:
                self._attempt_extension(item)
            else:
                raise KeyError("Key \'{}\' not in dict and no extension functions found for that key.".format(item))
        return super(SelfExtendingDict, self).__getitem__(item)

    def uninitialized_keys(self):
        return set(self.extension_funcs.keys()) - set(super(SelfExtendingDict, self).keys())

    def recalculate_items(self, items_to_recalculate=tuple()):
        for item in items_to_recalculate:
            del self[item]
        for item in items_to_recalculate:
            self._attempt_extension(item)

    @property
    def extension_funcs(self):
        return self._extension_funcs

    @extension_funcs.setter
    def extension_funcs(self, extension_funcs_dict):
        assert isinstance(extension_funcs_dict, dict), \
            "extension_funcs must be a dictionary of keys (fields) and functions."
        self._extension_funcs = extension_funcs_dict
