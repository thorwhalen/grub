
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

