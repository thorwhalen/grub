from grub import SearchStore


def ddir(obj):
    """
    List of (dir(obj)) attributes of obj that don't start with an underscore
    :param obj: Any python object
    :return: A list of attribute names
    """
    return [a for a in dir(obj) if not a.startswith('_')]


def search_documented_attributes(obj, obj_to_attrs=ddir, max_results=10):
    """
    Search the documented attributes of a python object
    :param obj: Any python object
    :param obj_to_attrs: The function that gives you attribute names of an object.
    :return: A SearchStore instance to search attributes via their docs

    >>> from inspect import getmodule
    >>> containing_module = getmodule(search_documented_attributes)  # the module of this function
    >>> search_module = search_documented_attributes(containing_module)
    >>> list(search_module('object'))  # if you get an error here, it's probably just be that the docs changed
    ['search_documented_attributes', 'ddir', 'SearchStore']
    >>> list(search_module('obj'))  # if you get an error here, it's probably just be that the docs changed
    ['ddir', 'search_documented_attributes', 'SearchStore']
    """

    def documented_attrs():
        for attr_name in obj_to_attrs(obj):
            attr_val = getattr(obj, attr_name)
            doc = getattr(attr_val, '__doc__', None)
            if isinstance(doc, str):
                yield attr_name, doc

    attrs_store = dict(documented_attrs())
    max_results = min(max_results, len(attrs_store))
    return SearchStore(attrs_store, max_results)
