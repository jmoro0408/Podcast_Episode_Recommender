import string
from collections import Counter
from pprint import pprint
from typing import Optional, Union

import gensim
from gensim import corpora
from gensim.models import Phrases
from nltk.stem.wordnet import WordNetLemmatizer
from spacy.lang.en.stop_words import STOP_WORDS

from preprocessing import (clean_text, generate_bigrams,
                           prepare_custom_stopwords)
from utils import append_to_txt_file, list_from_text, read_from_db, read_toml


def read_transcripts(config_dict: dict, row_limit: Optional[int] = None) -> list[str]:
    """Reads in the podcast transcripts from postgresql db to a list.

    Args:
        config_dict (dict): Config dict with DB parameters.
        row_limit (Optionsal(int)): Number of rows to receive. If None, all
        rows are returned. Defaults to None.

    Returns:
        list[str]: List of transcripts.
    """
    _query = """SELECT * from episodes"""
    if row_limit is not None:
        _query = """SELECT * from episodes LIMIT {}""".format(row_limit)
    df = read_from_db(_query, config_dict=config_dict)
    return df["transcript"].to_list()


def run_LDA(docs: list[str], **kwargs) -> tuple[gensim.models.ldamodel.LdaModel, list]:
    # TODO Split the bigrams appending into a seperate func.
    """Creates and runs the LDA model.

    Args:
        docs (list[str]): List of text (strings) to be used in the analysis.
        **kwargs: Other gensim LDA model arguments/hyperparameters

    Returns:
        gensim.models.ldamodel.LdaModel: Trained LDA model
        doc_term_matrix (list): document term matrix for the corpus
    """
    index_dictionary = corpora.Dictionary(docs)
    doc_term_matrix = [index_dictionary.doc2bow(doc) for doc in docs]
    Lda = gensim.models.ldamodel.LdaModel
    return Lda(doc_term_matrix, id2word=index_dictionary, **kwargs), doc_term_matrix


def main():
    NUM_ROWS = 20
    # Training Parameters
    num_topics = 10
    passes = 20
    iterations = 400
    eval_every = None

    config_dict = read_toml(r"db_info.toml")["database"]  # config dict to access db
    custom_stopwords = prepare_custom_stopwords(
        stopwords_to_add=None, add_word_fillers=True, erase=True
    )
    corpus = read_transcripts(config_dict, row_limit=NUM_ROWS)
    corpus = generate_bigrams(corpus)  # adding bigrams to corpus
    docs_clean = [clean_text(doc, custom_stopwords).split() for doc in corpus]
    ldamodel, doc_term_matrix = run_LDA(
        docs=docs_clean,
        num_topics=num_topics,
        passes=passes,
        iterations=iterations,
        eval_every=eval_every,
        alpha="auto",
        eta="auto",
    )
    top_topics = ldamodel.top_topics(doc_term_matrix)
    pprint(top_topics)


if __name__ == "__main__":
    main()
