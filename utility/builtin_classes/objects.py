import os
import pickle
from io import BytesIO

from .dicts import update_default_dict


def update_instance_with_defaults(instance_reference, attribute_defaults, passed_attributes=None,
                                  allow_new_attributes=False):
    """
    Updates the attributes of an instance with default values. The default values are taken from the attribute_defaults
    dictionary. If passed_attributes is not None, the default values are updated with the passed attributes. If
    allow_new_attributes is True, new attributes can be added to the instance.

    :param instance_reference: The instance to update
    :type instance_reference: Any
    :param attribute_defaults: A dictionary of default values for the attributes
    :type attribute_defaults: dict
    :param passed_attributes: A dictionary of attributes to update the default values with
    :type passed_attributes: dict or None
    :param allow_new_attributes: Whether to allow new attributes to be added to the instance
    :type allow_new_attributes: bool
    :return: None
    """
    attribute_dict = update_default_dict(attribute_defaults, update_dict=passed_attributes,
                                         allow_new_attributes=allow_new_attributes)
    instance_reference.__dict__.update(attribute_dict)


def save_object(obj, filename, overwrite=False):
    """
    Saves an object to a file using pickle. If the file already exists, it will not be overwritten unless the overwrite
    parameter is set to True.

    :param obj: The object to save
    :type obj: Any
    :param filename: The name of the file to save the object to
    :type filename: str
    :param overwrite: Whether to overwrite the file if it already exists
    :type overwrite: bool
    :return: None
    """
    if not overwrite and os.path.exists(filename):
        raise FileExistsError(f"File {filename} already exists!")
    with open(filename, "wb") as out_file:
        pickle.dump(obj, out_file, pickle.HIGHEST_PROTOCOL)


def load_object(filename):
    """
    Loads an object from a file using pickle.

    :param filename: The name of the file to load the object from
    :type filename: str
    :return: The loaded object
    :rtype: Any
    """
    with open(filename, "rb") as in_file:
        return pickle.load(in_file)


def copy_object(input_object):
    """
    An attempt at an universal copy function. The object is pickled and then unpickled. This should work for most
    objects, but may fail for some. An example of an object that this fails for is a matplotlib figure, as those are all
    registered with the backend and figure manager.
    From https://stackoverflow.com/a/45812071.

    :param input_object: An object
    :type input_object: Any
    :return: A copy of the object
    :rtype: Any
    """
    buffer = BytesIO()
    pickle.dump(input_object, buffer)
    buffer.seek(0)
    return pickle.load(buffer)
