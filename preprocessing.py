"""
Module to hold functions for preprocessing of text prior to LDA.
"""

import pickle
import string
from typing import Optional, Union
import logging

import gensim
import spacy
from gensim.corpora import Dictionary
from gensim.models import Phrases
from nltk.stem.wordnet import WordNetLemmatizer
from spacy.lang.en.stop_words import STOP_WORDS
from tqdm import tqdm

from utils import (list_from_text, read_toml, read_transcripts,
                   remove_if_substring)


def clean_text(
    input_text: str,
    custom_stopwords: Optional[list[str]] = None,
    normalized: bool = False,
    nouns_only: bool = False,
) -> str:
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
    if nouns_only:
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(normalized)
        tokens = list(doc)
        noun_tokens = [token for token in tokens if token.tag_ in ("NN", "NNP", "NNS")]
        nouns_joined = " ".join([str(i) for i in noun_tokens])
        return nouns_joined
    return normalized


def remove_rare_common_words(
    docs: list[str], no_below: int, no_above: float
) -> gensim.corpora.Dictionary:
    """Removes words that occur in less than no_below documents,
    and/or more than no_above percent of documents.

    Args:
        docs (list[str]): Corpus to be filtered
        no_below (int): Keep tokens which are contained in at least no_below documents.
        no_above (float): Keep tokens which are contained in no more than no_above documents
        (fraction of total corpus size, not an absolute number).

    Returns:
        gensim.corpora.Dictionary: filtered corpus gensim dictionary object
    """

    # Create a dictionary representation of the documents.
    dictionary = Dictionary(docs)

    # Filter out words that occur less than x documents, or more than x% of the documents.
    dictionary.filter_extremes(no_below=no_below, no_above=no_above)
    return dictionary


def generate_bigrams(docs: list[str]) -> list[str]:
    """Appends bigrams to the document dictionary.

    Args:
        docs (list[str]): Document corpus.

    Returns:
        list[str]: document corpus with bigrams
    """
    if isinstance(docs[0], str):
        doc_list = []
        for sublist in docs:
            doc_list.append(sublist.split(" "))
        docs = doc_list.copy()
    bigram = Phrases(docs, min_count=10)
    for idx, _ in enumerate(tqdm(docs)):
        for token in bigram[docs[idx]]:
            if "_" in token:
                # Token is a bigram, add to document.
                docs[idx].append(token)
    return docs


def prepare_custom_stopwords(
    stopwords_to_add: Optional[Union[list[str], str]] = None,
) -> list[str]:
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
        list[str]: List of custom stopwords to be passed to the algorithm.
    """
    if stopwords_to_add is None:
        stopwords_to_add = []
    if isinstance(stopwords_to_add, str):
        stopwords_to_add = stopwords_to_add.split(",")
    from_txt_file = list_from_text("custom_stopwords.txt")
    stopwords_to_add = stopwords_to_add + from_txt_file
    return stopwords_to_add


def preprocess_main(
    num_rows_db: Optional[int] = None,
    save_preprocessed_text: bool = True,
) -> tuple[list[str], gensim.corpora.Dictionary]:
    # TODO Add ability to pass args to inner funcs
    """
    Function to run through all text preprocessing steps.
    1. creates a list of custom stopwords
    2. reads in the episode transcripts from db
    3. generates bigrams from the transcripts
    4. cleans text by removing punctuation, stopwords, and lemmatizing
    5. removes rare and common words

    Args:
        num_rows_db (Optional[int], optional): No. rows (episodes) to grab from database.
        If None, all rows are returned. Defaults to None.
        save_preprocessed_text [bool]: If True, cleaned corpus is pickled in current dir.

    Returns:
        tuple[list[str], gensim.corpora.Dictionary]:
        1. corpus: Documents in avectorized form. Containing the frequency of each word, including the bigrams: list[str],
        2. Index dictionary mapping
    """
    logger = logging.getLogger('dev')
    logger.setLevel(logging.INFO)
    config_dict = read_toml(r"db_info.toml")["database"]  # config dict to access db
    custom_stopwords = prepare_custom_stopwords()
    logging.info("Reading transcripts")
    raw_transcripts = read_transcripts(
        config_dict, row_limit=num_rows_db
    )  # list of strings
    # removing stopwords, punctuation, and lemmatizing
    logging.info("Cleaning docs")
    docs_clean = [
        clean_text(doc, custom_stopwords, nouns_only=True) for doc in tqdm(raw_transcripts)
    ]  # list of list of string
    docs_clean = [x for x in docs_clean if x != []]  # Removing empty transcripts
    substrings_to_remove = ["com"]
    docs_clean = [remove_if_substring(doc, substrings_to_remove) for doc in docs_clean]
    logging.info("Generating bigrams")
    bigrams_clean = generate_bigrams(docs_clean)  # adding bigrams to corpus
    index_dictionary = remove_rare_common_words(
        bigrams_clean, no_below=25, no_above=0.75
    )
    logging.info("Generating bag of words")
    corpus = [index_dictionary.doc2bow(doc) for doc in tqdm(bigrams_clean)]
    logging.info("Text preprocessing complete")
    if save_preprocessed_text:
        with open("cleaned_docs.pkl", "wb") as f:
            pickle.dump(docs_clean, f)
        with open("index_dict.pkl", "wb") as f:
            pickle.dump(index_dictionary, f)
        with open("corpus.pkl", "wb") as f:
            pickle.dump(corpus, f)
    return corpus, index_dictionary
