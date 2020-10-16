from functools import cached_property
import re

import numpy as np

from py2store.slib.s_zipfile import FileStreamsOfZip
from py2store.base import Stream


def line_to_raw_word_vec(line):
    word, vec = line.split(maxsplit=1)
    return word.decode(), vec


class WordVecStream(Stream):
    _obj_of_data = line_to_raw_word_vec


class StreamsOfZip(FileStreamsOfZip):
    def _obj_of_data(self, data):
        return line_to_raw_word_vec(data)


def word_and_vecs(fp):
    #     fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')

    # consume the first line (n_lines, n_dims) not yielded
    n_lines, n_dims = map(int, fp.readline().decode().split())
    for line in fp:
        tok, *vec = line.decode().rstrip().split(' ')
        yield tok, tuple(map(float, vec))


class Search:
    """
    Example:

    ```
    zip_filepath = '/D/Dropbox/_odata/misc/wiki-news-300d-1M-subword.vec.zip'

    import pandas as pd
    df = pd.read_excel('/Users/twhalen/Downloads/pypi package names.xlsx')
    target_words = set(df.word)

    from grub.examples.pypi import Search

    s = Search(zip_filepath, search_words=target_words)
    s.search('search for the right name')
    ```
    """
    tokenizer = re.compile('\w+').findall

    def __init__(self,
                 wordvec_zip_filepath,
                 search_words,
                 wordvec_name_in_zip='wiki-news-300d-1M-subword.vec',
                 n_neighbors=37,
                 verbose=False
                 ):
        self.wordvec_zip_filepath = wordvec_zip_filepath
        self.wordvec_name_in_zip = wordvec_name_in_zip
        self.search_words = set(search_words)
        self.n_neighbors = n_neighbors
        self.verbose = verbose

    @cached_property
    def stream(self):
        return StreamsOfZip(self.wordvec_zip_filepath)

    @cached_property
    def wordvecs(self):
        if self.verbose:
            print('Gathering all the word vecs. This could take a few minutes...')
        with self.stream[self.wordvec_name_in_zip] as fp:
            all_wordvecs = dict(word_and_vecs(fp))
        return all_wordvecs

    def filtered_wordvecs(self, tok_filt):
        with self.stream['wiki-news-300d-1M-subword.vec'] as fp:
            yield from filter(lambda x: tok_filt(x[0]), word_and_vecs(fp))

    def vec_matrix(self, words):
        return np.array([self.wordvecs.get(w, None) for w in words])

    def mean_vec(self, words):
        return np.mean(self.vec_matrix(words), axis=0)

    def query_to_vec(self, query):
        tokens = self.tokenizer(query)
        return self.mean_vec(tokens)

    def query_to_vec_matrix(self, query):
        tokens = self.tokenizer(query)
        return self.vec_matrix(tokens)

    @cached_property
    def knn(self):
        from sklearn.neighbors import NearestNeighbors
        taget_wv = dict(self.filtered_wordvecs(lambda x: x in self.search_words))
        X = np.array(list(taget_wv.values()))

        knn = NearestNeighbors(n_neighbors=self.n_neighbors, metric='cosine').fit(X)
        knn.words = np.array(list(taget_wv.keys()))
        return knn

    def search(self, query):
        query_vec = self.query_to_vec(query)
        r_dist, r_idx = self.knn.kneighbors(query_vec.reshape(1, -1))
        return self.knn.words[r_idx]
