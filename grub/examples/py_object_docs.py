"""Searching python object docs

You know when you're using a new package, you REPL a few things, and you wonder what
you can do with an object you're holding in your hand.

You can go look online, but it's not always that easy to find the "what can I do with
this particular object" answers.

You can do `help(obj)` and browse it's output.

... or, you can do what we're showing here.


docs_searcher: Search the docs of the attributes of the input python object.

    >>> from grub import TfidfKnnSearcher
    >>> search = docs_searcher(TfidfKnnSearcher, n_results=2)
    >>> len(search)
    6
    >>> keys = search('make index')
    >>> list(keys)
    ['fit', 'fit_knn']

    Those are the names of the top 2 (=n_results) methods matching the query 'make
    index'. You can see the actual docs using `[...]`:

    >>> print(search['fit'])
    Fit all models of pipeline, i.e. making a search index.

Note: There's a similar function in `grub.pycode` called `search_documented_attributes`

"""

from sklearn.neighbors import NearestNeighbors

from grub import TfidfKnnSearcher


def not_dundered(string):
    return not string.startswith('__')


def not_underscored(string):
    return not string.startswith('_')


DFLT_ATTRNAME_FILT = not_underscored
DFLT_ATTR_FILT = callable


def attr_docs(obj, attrname_filt=DFLT_ATTRNAME_FILT, attr_filt=DFLT_ATTR_FILT):
    for attrname in filter(attrname_filt, dir(obj)):
        attr = getattr(obj, attrname)
        if attr_filt(attr):
            doc = getattr(attr, '__doc__', None)
            if doc:
                yield attrname, doc


def docs_searcher(
    obj, n_results=10, attrname_filt=DFLT_ATTRNAME_FILT, attr_filt=DFLT_ATTR_FILT
):
    """Search the docs of the attributes of the input python object.

    >>> from grub import TfidfKnnSearcher
    >>> search = docs_searcher(TfidfKnnSearcher, n_results=2)
    >>> len(search)
    6
    >>> keys = search('make index')
    >>> list(keys)
    ['fit', 'fit_knn']

    Those are the names of the top 2 (=n_results) methods matching the query 'make
    index'. You can see the actual docs using `[...]`:

    >>> print(search['fit'])
    Fit all models of pipeline, i.e. making a search index.

    """
    docs = dict(attr_docs(obj, attrname_filt, attr_filt))
    knn = NearestNeighbors(n_neighbors=n_results, metric='cosine')
    return TfidfKnnSearcher(docs, knn=knn).fit()
