# grub.base

Base functionality

### Functions

| [`camelcase_and_underscore_tokenizer`](#grub.base.camelcase_and_underscore_tokenizer)(string)   |                                                             |
|-----------------------------------------------------------------------------------------------|-------------------------------------------------------------|
| `get_py_files_store`(spec)                                                                    |                                                             |
| `grub`(search_store, query[, n])                                                              |                                                             |
| [`grubber`](#grub.base.grubber)(search_store[, n])                   | Computes a TF-IDF model and a KNN model for a search store. |
| `iterate_values_and_accumulate_non_error_keys`(...)                                           |                                                             |

### Classes

| [`CodeSearcher`](#grub.base.CodeSearcher)([search_store, tfidf, knn])          | Text searcher with fitters                                                                |
|----------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------|
| [`CodeSearcherBase`](#grub.base.CodeSearcherBase)([search_store, tfidf, knn])      |                                                                                           |
| [`DfltSearchStore`](#grub.base.DfltSearchStore)()                                 |                                                                                           |
| [`SearchStore`](#grub.base.SearchStore)(store[, n_neighbors, tokenizer])      | Build a search index for anything (that is given a mapping interface with string values). |
| [`SearchStoreMixin`](#grub.base.SearchStoreMixin)()                                |                                                                                           |
| [`StoreMixin`](#grub.base.StoreMixin)()                                      |                                                                                           |
| [`TextFilesSearcher`](#grub.base.TextFilesSearcher)([search_store, tfidf, knn])     | Fittable Text searcher with fitters                                                       |
| [`TextFilesSearcherBase`](#grub.base.TextFilesSearcherBase)([search_store, tfidf, knn]) |                                                                                           |
| `TfidfKnnFitMixin`()                                                                               |                                                                                           |
| [`TfidfKnnSearcher`](#grub.base.TfidfKnnSearcher)([search_store, tfidf, knn])      |                                                                                           |

### *class* grub.base.CodeSearcher(search_store=<factory>, tfidf='\\\\\\\\w', knn=<factory>)

Bases: [`CodeSearcherBase`](#grub.base.CodeSearcherBase)

Text searcher with fitters

### *class* grub.base.CodeSearcherBase(search_store=<factory>, tfidf='\\\\\\\\w', knn=<factory>)

Bases: [`TextFilesSearcherBase`](#grub.base.TextFilesSearcherBase)

### *class* grub.base.DfltSearchStore

Bases: [`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)

### *class* grub.base.SearchStore(store, n_neighbors=10, tokenizer=<function camelcase_and_underscore_tokenizer>)

Bases: [`StoreMixin`](#grub.base.StoreMixin)

Build a search index for anything (that is given a mapping interface with string values).

A store is anything with a `collections.Mapping` interface.
Typically, a store’s backend comes from local files or data-base wrapped into a mapping
(see `py2store` for tools to do that!).
For testing purposes though, we’ll use a `dict` here:

```pycon
>>> store = {
... "Nelson Mandela": "The greatest glory in living lies not in never falling, but in rising every time we fall.",
... "Walt Disney": "The way to get started is to quit talking and begin doing.",
... "Steve Jobs": "Your time is limited, so don't waste it living someone else's life."
... }
>>> search = SearchStore(store, n_neighbors=2)  # our store is small, so need to restrict our result size to less
>>> list(search('living'))
['Steve Jobs', 'Nelson Mandela']
```

A `SearchStore` instance is picklable.

```pycon
>>> import pickle
>>> unserialized_search = pickle.loads(pickle.dumps(search))
>>> list(unserialized_search('living'))
['Steve Jobs', 'Nelson Mandela']
```

### *class* grub.base.SearchStoreMixin

Bases: `KvReader`

### *class* grub.base.StoreMixin

Bases: [`SearchStoreMixin`](#grub.base.SearchStoreMixin)

### *class* grub.base.TextFilesSearcher(search_store=<factory>, tfidf='\\\\\\\\w', knn=<factory>)

Bases: [`TextFilesSearcherBase`](#grub.base.TextFilesSearcherBase)

Fittable Text searcher with fitters

### *class* grub.base.TextFilesSearcherBase(search_store=<factory>, tfidf='\\\\\\\\w', knn=<factory>)

Bases: [`TfidfKnnSearcher`](#grub.base.TfidfKnnSearcher)

### *class* grub.base.TfidfKnnSearcher(search_store=<factory>, tfidf=<factory>, knn=<factory>)

Bases: [`SearchStoreMixin`](#grub.base.SearchStoreMixin), `TfidfKnnFitMixin`

#### fvs(error_callback=None)

The feature vectors for the search_store

### grub.base.camelcase_and_underscore_tokenizer(string)

```pycon
>>> print(*camelcase_and_underscore_tokenizer('thereIsSomething i_wanted toShow'))
there is something i wanted to show
```

### grub.base.grubber(search_store, n=10)

Computes a TF-IDF model and a KNN model for a search store.
