import re
from grub.base import (
    SearchStore,
    CodeSearcher,
    get_py_files_store,
    camelcase_and_underscore_tokenizer
)


# TODO: Not finished: Continue
class PyCodeSearcherBase(CodeSearcher):
    def __post_init__(self):
        self.search_store = pyobj_semantics_dict(self.search_store)


doctest_line_p = re.compile('\s*>>>')
empty_line = re.compile('\s*$')


def non_doctest_lines(doc):
    r"""Generator of lines of the doc string that are not in a doctest scope.

    >>> def _test_func():
    ...     '''Line 1
    ...     Another
    ...     >>> doctest_1
    ...     >>> doctest_2
    ...     line_after_a_doc_test
    ...     another_line_that_is_in_the_doc_test scope
    ...
    ...     But now we're out of a doctest's scope
    ...
    ...     >>> Oh no, another doctest!
    ...     '''
    >>> from inspect import getdoc
    >>>
    >>> list(non_doctest_lines(getdoc(_test_func)))
    ['Line 1', 'Another', "But now we're out of a doctest's scope", '']

    :param doc:
    :return:
    """
    last_line_was_a_doc_test = False
    for line in doc.splitlines():
        if not doctest_line_p.match(line):
            if not last_line_was_a_doc_test:
                yield line
                last_line_was_a_doc_test = False
            else:
                if empty_line.match(line):
                    last_line_was_a_doc_test = False
        else:
            last_line_was_a_doc_test = True


def call_and_return_none_on_exception(func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except Exception:
        return None


import inspect
from doctest import DocTestFinder
from inspect import signature, getfile, getcomments, getsource, getsourcelines, getdoc

doctest_finder = DocTestFinder(recurse=False)


def argnames(func):
    return ' '.join(inspect.signature(func).parameters)


def tokenize_for_code(string):
    camelcase_and_underscore_tokenizer(string)


_func_info_funcs = {
    'func_name': lambda f: f.__qualname__,
    'arg_names': argnames,
    'comments': getcomments,
    'doc': lambda f: '\n'.join(non_doctest_lines(getdoc(f))),
}


def func_key_info(func) -> dict:
    """Information that points to the function's source"""
    func_src, lineno = getsourcelines(func)
    return getfile(func), lineno


def func_semantic_info(func) -> dict:
    """A dict of semantically relevant infos about a function."""

    def gen():
        for name, info_func in _func_info_funcs.items():
            info = call_and_return_none_on_exception(info_func, func)
            if info is not None:
                yield name, info

    return dict(gen())


def import_module_from_filepath(filepath):
    import importlib.util
    spec = importlib.util.spec_from_file_location("module.name", filepath)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def objs_from_store(store):
    for k in store:
        f = store._prefix + k  # TODO: Revise. Fragile
        m = import_module_from_filepath(f)
        for a in dir(m):
            if not a.startswith('__'):
                yield getattr(m, a)


def gather_objs(objs):
    cumul = set()
    for obj in objs:
        try:
            cumul.add(obj)
        except TypeError:
            pass
    return cumul


def pyobj_semantics_dict(src):
    module_files_store = get_py_files_store(src)
    objs = gather_objs(objs_from_store(module_files_store))

    def gen():
        for obj in objs:
            try:
                yield func_key_info(obj), '\n'.join(func_semantic_info(obj).values())
            except Exception:
                pass

    return dict(gen())


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
    >>> search_module = search_documented_attributes(containing_module, max_results=3)
    >>> list(search_module('documented attributes'))  # if you get an error here, it's probably just be that the docs changed
    ['search_documented_attributes', 'ddir', 'inspect']
    >>> list(search_module('documented objects'))  # if you get an error here, it's probably just be that the docs changed
    ['search_documented_attributes', 'DocTestFinder', 'doctest_finder']
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
