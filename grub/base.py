import re
from typing import Mapping, Union
from types import ModuleType
from dataclasses import dataclass
from inspect import ismodule
import os

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.validation import check_is_fitted

from py2store import LocalTextStore, cached_keys, lazyprop, KvReader


def grub(search_store, query, n=10):
    search_store = get_py_files_store(search_store)
    knn = NearestNeighbors(n_neighbors=n, metric='cosine')
    search = TextFilesSearcher(search_store, knn=knn).fit()
    return search(query)


def get_py_files_store(spec):
    if ismodule(spec):
        spec = os.path.dirname(spec.__file__) + '{}.py'
    if isinstance(spec, str):
        spec = LocalTextStore(spec)
    return spec


camelcase_p = re.compile(r'''
    # Find words in a string. Order matters!
    [A-Z]+(?=[A-Z][a-z]) |  # All upper case before a capitalized word
    [A-Z]?[a-z]+ |  # Capitalized words / all lower case
    [A-Z]+ |  # All upper case
    \d+  # Numbers  TODO: Might want to drop that.
''', re.VERBOSE)


def camelcase_and_underscore_tokenizer(string):
    """
    >>> print(*camelcase_and_underscore_tokenizer('thereIsSomething i_wanted toShow'))
    there is something i wanted to show
    """
    return list(map(str.lower, camelcase_p.findall(string)))


class SearchStoreMixin(KvReader):
    store_attr = 'search_store'

    def __getitem__(self, k):
        return getattr(self, self.store_attr).__getitem__(k)

    def __iter__(self):
        return getattr(self, self.store_attr).__iter__()

    def __contains__(self, k):
        return getattr(self, self.store_attr).__contains__(k)

    def __len__(self):
        return getattr(self, self.store_attr).__len__()


class StoreMixin(SearchStoreMixin):
    store_attr = 'store'


class SearchStore(StoreMixin):
    """Build a search index for anything (that is given a mapping interface with string values).

    A store is anything with a ``collections.Mapping`` interface.
    Typically, a store's backend comes from local files or data-base wrapped into a mapping
    (see ``py2store`` for tools to do that!).
    For testing purposes though, we'll use a ``dict`` here:

    >>> store = {
    ... "Nelson Mandela": "The greatest glory in living lies not in never falling, but in rising every time we fall.",
    ... "Walt Disney": "The way to get started is to quit talking and begin doing.",
    ... "Steve Jobs": "Your time is limited, so don't waste it living someone else's life."
    ... }
    >>> search = SearchStore(store, n_neighbors=2)  # our store is small, so need to restrict our result size to less
    >>> list(search('living'))
    ['Steve Jobs', 'Nelson Mandela']

    A ``SearchStore`` instance is picklable.

    >>> import pickle
    >>> unserialized_search = pickle.loads(pickle.dumps(search))
    >>> list(unserialized_search('living'))
    ['Steve Jobs', 'Nelson Mandela']

    """
    _state_attrs = ('keys_array', 'tfidf', 'knn')

    def __init__(self,
                 store,
                 n_neighbors: int = 10,
                 tokenizer=camelcase_and_underscore_tokenizer
                 ):
        """Index store (values) and provide a search functionality for it.

        :param store: The collection that should be indexed for search.
            A store is anything with a ``collections.Mapping`` interface.
            Typically, a store's backend comes from local files or data-base wrapped into a mapping
            (see ``py2store`` for tools to do that!).
            If given as a string, it will be considered to be a file path and given to
            ``py2store.LocalTextStore`` to make a store.
        :param n_neighbors: The number search results to provide
        :param tokenizer: The function to apply to store (string) values to get an iterable of tokens.
            It is these tokens that are used both as the vocabulary for the queries and to represent searched values.
        """
        if isinstance(store, str):
            store = LocalTextStore(store)
        self.store = store

        self.tfidf = TfidfVectorizer(tokenizer=tokenizer, token_pattern=None)
        self.tfidf.fit(raw_documents=self.store.values())

        self.keys_array = np.array(list(self.store), dtype=object)

        doc_vecs = self.tfidf.fit_transform(self.store.values())
        self.n_neighbors = n_neighbors
        self.knn = NearestNeighbors(n_neighbors=n_neighbors,
                                    metric='cosine').fit(doc_vecs)

    def __getstate__(self):
        return {attr: getattr(self, attr) for attr in self._state_attrs}

    def __setstate__(self, d):
        for attr in self._state_attrs:
            setattr(self, attr, d[attr])

    def __call__(self, query):
        (score, *_), (idx, *_) = self.knn.kneighbors(self.tfidf.transform([query]))
        return self.keys_array[idx]

    def __getitem__(self, k):
        return self.store.__getitem__(k)


class DfltSearchStore(Mapping):
    def __iter__(self):
        yield from ()

    def __len__(self):
        return 0

    def __contains__(self):
        return False

    def __getitem__(self, item):
        raise KeyError("DfltSearchStore is not meant to be used")


# search_store and
@dataclass
class TfidfKnnSearcher(SearchStoreMixin):
    search_store: Mapping = DfltSearchStore()
    tfidf: TfidfVectorizer = TfidfVectorizer()
    knn: NearestNeighbors = NearestNeighbors(n_neighbors=10, metric='cosine')

    def __getattr__(self, attr):
        """Delegate method to wrapped store if not part of wrapper store methods"""
        return getattr(self.search_store, attr)

    @lazyprop
    def keys_array(self):
        return np.array(self.search_store)

    @property
    def n_neighbors(self):
        return self.knn.n_neighbors

    def fvs(self, error_callback=None):
        """The feature vectors for the search_store"""
        if hasattr(self.search_store, 'keys_cache'):
            return self.tfidf
        else:
            keys_cache = []
            search_documents = iterate_values_and_accumulate_non_error_keys(
                self.search_store,
                cache_keys_here=keys_cache,
                error_callback=error_callback
            )
            fvs = self.tfidf.fit_transform(search_documents)
            self.search_store = cached_keys(self.search_store, keys_cache=keys_cache)
            return fvs

    # def fit(self, learn_store, ):
    def __call__(self, query):
        (_, *_), (idx, *_) = self.knn.kneighbors(self.tfidf.transform([query]))
        # # equivalently:
        # fv = self.query_to_fv(query)
        # idx, scores = self.fv_to_idx_and_score(fv)
        return self.keys_array[idx]

    def docs_to_fvs(self, docs):
        self.tfidf.transform(docs)

    def query_to_fv(self, query):
        return self.tfidf.transform([query])[0]

    def fv_to_idx_and_scores(self, fv):
        (scores, *_), (idx, *_) = self.knn.kneighbors([fv])
        return idx, scores

    _state_attrs = ('keys_array', 'tfidf', 'knn')

    def __getstate__(self):
        return {attr: getattr(self, attr) for attr in self._state_attrs}

    def __setstate__(self, d):
        for attr in self._state_attrs:
            setattr(self, attr, d[attr])


@dataclass
class TextFilesSearcherBase(TfidfKnnSearcher):
    search_store: Union[str, Mapping] = DfltSearchStore()
    tfidf: TfidfVectorizer = TfidfVectorizer(token_pattern=camelcase_p)

    def __post_init__(self):
        if isinstance(self.search_store, str):
            self.search_store = LocalTextStore(self.search_store)


@dataclass
class CodeSearcherBase(TextFilesSearcherBase):
    search_store: Union[str, Mapping, ModuleType] = DfltSearchStore()

    def __post_init__(self):
        self.search_store = get_py_files_store(self.search_store)


class TfidfKnnFitMixin(TfidfKnnSearcher):
    def set_search_store(self, search_store):  # TODO: override setter with this function
        self.search_store = search_store
        assert isinstance(self.search_store, Mapping), "Your store isn't a Mapping"
        return self.fit_knn()

    def fit_knn(self):
        self.knn.fit(self.fvs())
        return self

    def fit_tfidf(self, learn_store=None, error_callback=None):
        if learn_store is None:
            learn_store = self.search_store
        keys_fit_on_ = []
        raw_documents = iterate_values_and_accumulate_non_error_keys(
            learn_store,
            cache_keys_here=keys_fit_on_,
            error_callback=error_callback
        )
        self.tfidf.fit(raw_documents=raw_documents)
        self.tfidf.keys_fit_on_ = keys_fit_on_
        return self

    def fit(self, learn_store=None):
        return self.fit_tfidf(learn_store).fit_knn()


def iterate_values_and_accumulate_non_error_keys(
        store,
        cache_keys_here: list,
        errors_caught=Exception,
        error_callback=None
):
    for k in store:
        try:
            v = store[k]
            cache_keys_here.append(k)
            yield v
        except errors_caught as err:
            if error_callback is not None:
                error_callback(store, k, err)


class TextFilesSearcher(TextFilesSearcherBase, TfidfKnnFitMixin):
    """Fittable Text searcher with fitters"""


class CodeSearcher(CodeSearcherBase, TfidfKnnFitMixin):
    """Text searcher with fitters"""
