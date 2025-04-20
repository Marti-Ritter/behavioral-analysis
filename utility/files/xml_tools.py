import xml.etree.ElementTree as ET

from ..builtin_classes.dicts import set_dict_value_by_graph, get_dict_value_by_graph, nested_dict_walk, \
    numerize_string_values_in_dict


def ensure_etree(element_or_string):
    if isinstance(element_or_string, str):
        return ET.fromstring(element_or_string)
    elif isinstance(element_or_string, ET.Element):
        return element_or_string
    else:
        raise TypeError("element_or_string must be either an ElementTree element or a string")


def etree_to_dict(element_or_string, add_attributes=True, add_text=True):
    """
    Converts an ElementTree element to a dictionary representation. This representation is not guaranteed to be
    reversible. If add_attributes is True, then the attributes of the element are added to the dictionary. If add_text
    is True, then the text of the element is added to the dictionary. If both add_attributes and add_text are True,
    then the text of the element is added to the dictionary under the key '#text'. If both add_attributes and add_text
    are False, then the dictionary will only contain the children of the element.

    :param element_or_string: An ElementTree element or a string representation of an ElementTree element
    :type element_or_string: ET.Element or str
    :param add_attributes: Whether to add attributes to the dictionary
    :type add_attributes: bool
    :param add_text: Whether to add text to the dictionary
    :type add_text: bool
    :return: A dictionary representation of the element
    :rtype: dict
    """
    element = ensure_etree(element_or_string)
    result = {}

    if add_attributes:
        # Add attributes to the dictionary
        for key, value in element.attrib.items():
            result['@' + key] = value

    # Add children to the dictionary
    for child in element:
        child_dict = etree_to_dict(child, add_attributes=add_attributes, add_text=add_text)
        result.setdefault(child.tag, []).append(child_dict)

    if add_text:
        # Add text content to the dictionary
        if element.text and element.text.strip():
            result['#text'] = element.text.strip()

    return result


def dict_to_etree(d):
    def _to_etree(_d, root):
        if not _d:
            pass
        elif isinstance(_d, str):
            root.text = _d
        elif isinstance(_d, dict):
            for k, v in _d.items():
                assert isinstance(k, str)
                if k.startswith('#'):
                    assert k == '#text' and isinstance(v, str)
                    root.text = v
                elif k.startswith('@'):
                    assert isinstance(v, str)
                    root.set(k[1:], v)
                elif isinstance(v, list):
                    for e in v:
                        _to_etree(e, ET.SubElement(root, k))
                else:
                    _to_etree(v, ET.SubElement(root, k))
        else:
            raise TypeError('invalid type: ' + str(type(_d)))

    assert isinstance(d, dict) and len(d) == 1
    tag, body = next(iter(d.items()))
    node = ET.Element(tag)
    _to_etree(body, node)
    return node


def etree_to_simplified_dict(element_or_string, add_attributes=True, add_text=True, text_to_value=True,
                             transform_types=True):
    """
    Converts an ElementTree element to a simplified dictionary representation. This representation is not guaranteed to
    be reversible. Simplifications include the removal of leading @s and #s from attribute and text keys, respectively,
    and the removal of empty text elements. Single element lists are also converted to their single element. If
    text_to_value is True, then the text of the element is converted to the value of the element, if there is no other
    content in the element. If transform_types is True, then the values of the dictionary are converted to their
    numerical (or boolean) equivalents, if possible.

    :param element_or_string: An ElementTree element or a string representation of an ElementTree element
    :type element_or_string: ET.Element or str
    :param add_attributes: Whether to add attributes to the dictionary
    :type add_attributes: bool
    :param add_text: Whether to add text to the dictionary
    :type add_text: bool
    :param text_to_value: Whether to convert the text of the element to the value of the element, if there is no other
    content in the element
    :type text_to_value: bool
    :param transform_types: Whether to convert the values of the dictionary to their numerical (or boolean) equivalents,
    if possible
    :type transform_types: bool
    :return: A simplified dictionary representation of the element
    :rtype: dict
    """

    standard_dict = etree_to_dict(element_or_string, add_attributes=add_attributes, add_text=add_text)

    def _simplify_dict(val):
        if isinstance(val, list) and len(val) == 1:
            return _simplify_dict(val[0])
        elif isinstance(val, dict):
            return {_key.lstrip('@'): _simplify_dict(_value) for _key, _value in val.items()}
        else:
            return val

    simplified_dict = _simplify_dict(standard_dict)

    if text_to_value:
        for graph, value in nested_dict_walk(simplified_dict, bottom_up=True):
            if (graph[-1] == "#text") and (len(get_dict_value_by_graph(simplified_dict, graph[:-1]).keys()) == 1):
                set_dict_value_by_graph(simplified_dict, graph[:-1], value)

    if transform_types:
        simplified_dict = numerize_string_values_in_dict(simplified_dict)
        for graph, value in nested_dict_walk(simplified_dict, bottom_up=True):
            if isinstance(value, str):
                if value.lower() == "true":
                    set_dict_value_by_graph(simplified_dict, graph, True)
                elif value.lower() == "false":
                    set_dict_value_by_graph(simplified_dict, graph, False)
    return simplified_dict
