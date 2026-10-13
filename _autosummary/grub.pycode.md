# grub.pycode

Searching python code

### Functions

| `argnames`(func)                                                                          |                                                                          |
|-------------------------------------------------------------------------------------------|--------------------------------------------------------------------------|
| `call_and_return_none_on_exception`(func, ...)                                            |                                                                          |
| [`ddir`](#grub.pycode.ddir)(obj)                                | List of (dir(obj)) attributes of obj that don't start with an underscore |
| [`func_key_info`](#grub.pycode.func_key_info)(func)                      | Information that points to the function's source                         |
| [`func_semantic_info`](#grub.pycode.func_semantic_info)(func)                 | A dict of semantically relevant infos about a function.                  |
| `gather_objs`(objs)                                                                       |                                                                          |
| `import_module_from_filepath`(filepath)                                                   |                                                                          |
| [`non_doctest_lines`](#grub.pycode.non_doctest_lines)(doc)                   | Generator of lines of the doc string that are not in a doctest scope.    |
| `objs_from_store`(store)                                                                  |                                                                          |
| `pyobj_semantics_dict`(src)                                                               |                                                                          |
| [`search_documented_attributes`](#grub.pycode.search_documented_attributes)(obj[, ...]) | Search the documented attributes of a python object                      |
| `tokenize_for_code`(string)                                                               |                                                                          |

### Classes

| [`PyCodeSearcherBase`](#grub.pycode.PyCodeSearcherBase)([search_store, tfidf, knn])   |    |
|---------------------------------------------------------------------------------------------------|----|

### *class* grub.pycode.PyCodeSearcherBase(search_store=<factory>, tfidf='\\\\\\\\w', knn=<factory>)

Bases: [`CodeSearcher`](grub.base.md#grub.base.CodeSearcher)

### grub.pycode.ddir(obj)

List of (dir(obj)) attributes of obj that don’t start with an underscore

* **Parameters:**
  **obj** – Any python object
* **Returns:**
  A list of attribute names

### grub.pycode.func_key_info(func)

Information that points to the function’s source

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

### grub.pycode.func_semantic_info(func)

A dict of semantically relevant infos about a function.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

### grub.pycode.non_doctest_lines(doc)

Generator of lines of the doc string that are not in a doctest scope.

```pycon
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
```

* **Parameters:**
  **doc**
* **Returns:**

### grub.pycode.search_documented_attributes(obj, obj_to_attrs=<function ddir>, max_results=10)

Search the documented attributes of a python object

* **Parameters:**
  * **obj** – Any python object
  * **obj_to_attrs** – The function that gives you attribute names of an object.
* **Returns:**
  A SearchStore instance to search attributes via their docs

```pycon
>>> from inspect import getmodule
>>> containing_module = getmodule(search_documented_attributes)  # the module of this function
>>> search_module = search_documented_attributes(containing_module, max_results=3)
>>> list(search_module('documented attributes'))  # if you get an error here, it's probably just be that the docs changed
['search_documented_attributes', 'ddir', 'inspect']
>>> list(search_module('documented objects'))  # if you get an error here, it's probably just be that the docs changed
['search_documented_attributes', 'DocTestFinder', 'doctest_finder']
```
