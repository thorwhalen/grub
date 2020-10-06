import re
from typing import Mapping, Union
from types import ModuleType
from dataclasses import dataclass
from inspect import ismodule
import os

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from py2store import LocalTextStore, cached_keys


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
    \d+  # Numbers
''', re.VERBOSE)


def camelcase_and_underscore_tokenizer(string):
    """
    >>> print(*camelcase_and_underscore_tokenizer('thereIsSomething i_wanted toShow'))
    there is something i wanted to show
    """
    return list(map(str.lower, camelcase_p.findall(string)))


class SearchStore:
    def __init__(self,
                 store,
                 n_neighbors=10,
                 tokenizer=camelcase_and_underscore_tokenizer
                 ):
        if isinstance(store, str):
            store = LocalTextStore(store)
        self.store = store

        self.tfidf = TfidfVectorizer(tokenizer=tokenizer, token_pattern=None)
        self.tfidf.fit(raw_documents=self.store.values())

        self.keys_array = np.array(list(self.store))

        doc_vecs = self.tfidf.fit_transform(self.store.values())
        self.n_neighbors = n_neighbors
        self.knn = NearestNeighbors(n_neighbors=n_neighbors,
                                    metric='cosine').fit(doc_vecs)

    def __call__(self, query):
        (score, *_), (idx, *_) = self.knn.kneighbors(self.tfidf.transform([query]))
        return self.keys_array[idx]


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
class TfidfKnnSearcher:
    search_store: Mapping = DfltSearchStore()
    tfidf: TfidfVectorizer = TfidfVectorizer()
    knn: NearestNeighbors = NearestNeighbors(n_neighbors=10, metric='cosine')

    @property
    def keys_array(self):
        return np.array(self.search_store)

    @property
    def n_neighbors(self):
        return self.knn.n_neighbors

    def fvs(self, error_callback=None):
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
