import re

import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from py2store import LocalTextStore

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
