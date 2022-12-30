"""
Module to hold functions for preprocessing of text prior to LDA.
"""

import string
from typing import Optional, Union

from gensim.models import Phrases
from nltk.stem.wordnet import WordNetLemmatizer
from spacy.lang.en.stop_words import STOP_WORDS

from utils import append_to_txt_file, list_from_text


def clean_text(input_text: str, custom_stopwords: Optional[list[str]] = None) -> str:
    """cleans input text by:
    1. removing punctuation
    2. removing stopwords, including any custom stopwords passed as an argument
    3. lemmanizing words

    Args:
        input_text (_type_): text to be cleaned
        custom_stopwords (Optional[list], optional): Custom stopwordst to be removoved, if any.
        Should be a list of string. Defaults to None.

    Returns:
        _type_: cleaned text
    """
    if custom_stopwords is not None:
        STOP_WORDS.update(set(custom_stopwords))
    lemma = WordNetLemmatizer()
    punc_free = input_text.translate(str.maketrans("", "", string.punctuation))
    stop_free = " ".join(
        [i for i in punc_free.lower().split(" ") if i not in STOP_WORDS]
    )
    normalized = " ".join(lemma.lemmatize(word) for word in stop_free.split())
    return normalized


def generate_bigrams(docs: list[str]) -> list[str]:
    """Appends bigrams to the document dictionary.

    Args:
        docs (list[str]): Document corpus.

    Returns:
        list[str]: document corpus with bigrams
    """
    bigram = Phrases(docs, min_count=20)
    for idx, _ in enumerate(docs):
        for token in bigram[docs[idx]]:
            if "_" in token:
                # Token is a bigram, add to document.
                docs[idx].append(token)
    return docs


def prepare_custom_stopwords(
    stopwords_to_add: Optional[Union[list[str], str]] = None,
    add_word_fillers: bool = True,
    **kwargs
) -> None:
    """Prepares custom stopwords from the full corpus.
        Custom stopwords should be provided either as a list of words,
        or a string with commas separating.
        Standard filler words can be added to the list with the add_word_fillers arg.
        The list is then written to a custom_stopwords.txt file for later use.

    Args:
        stopwords_to_add (Union[list[str],str]): custom stopwords to be included.
        add_word_fillers (bool, optional): Add frequent filler words to stopwords list. Defaults to True.
        **kwargs: Optional keyword arguments for the append_to_txt_file function.

    Returns:
        _type_: None.
    """
    if stopwords_to_add is None:
        stopwords_to_add = []
    if isinstance(stopwords_to_add, str):
        stopwords_to_add = stopwords_to_add.split(",")
    if add_word_fillers:
        filler_words = [
            "like",
            "yeah",
            "um",
            "eh",
            "actually",
            "see",
            "well",
            "er",
            "said",
            "right",
            "he",
        ]  # spoken word fillers
        stopwords_to_add = stopwords_to_add + filler_words
    append_to_txt_file(filler_words, r"custom_stopwords.txt", **kwargs)
    return list_from_text(r"custom_stopwords.txt")
