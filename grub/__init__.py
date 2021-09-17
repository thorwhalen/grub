"""Make a search object in the simplest way possible"""

from grub.base import (
    SearchStore,
    CodeSearcher,
    TfidfKnnSearcher,
    grub,
    camelcase_and_underscore_tokenizer,
)
from grub.pycode import search_documented_attributes
