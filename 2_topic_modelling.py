import string
from collections import Counter
from typing import Optional, Union
from pprint import pprint

import gensim
from gensim import corpora
from nltk.stem.wordnet import WordNetLemmatizer
from spacy.lang.en.stop_words import STOP_WORDS

from utils import append_to_txt_file, list_from_text, read_from_db, read_toml


def most_frequent_words(docs: list[str], percent: int) -> list[str]:
    """Returns the most frequent x percentage of words in a corpus.
    i.e a percent value of 10 will return the most frequent 10% of words.

    Args:
        docs (list[str]): List of texts (strings) to search through.
        percent (int): most frequent percent to return.

    Returns:
        list[str]: most frequent words
    """
    combined_text = "".join(docs).lower()
    punc_free = combined_text.translate(
        str.maketrans("", "", string.punctuation)
    )  # remove punctuation
    total_unique_words = len(set(punc_free.split()))
    top_x_percent_value = round(total_unique_words * (percent / 100))
    text_list = punc_free.split()
    counters_found = Counter(text_list)
    # Word freuqnecy tuple returns the words with their count. i.e ("the", 86).
    words_frequency_tuple = counters_found.most_common(top_x_percent_value)
    return [x[0] for x in words_frequency_tuple]


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


def run_LDA(docs: list[str], **kwargs) -> tuple[gensim.models.ldamodel.LdaModel, list]:
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


def read_transcripts(config_dict: dict, row_limit:Optional[int] = None) -> list[str]:
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


def prepare_stopwords(
    full_corpus: list[str],
    percent_frequent_words: Union[int, float] = 1.5,
    **kwargs
) -> None:
    """Prepares custom stopwords from the full corpus.
    Uses the full corpus to findt he top x% of words (default at 1.5%) at use these as stopwords.

    Args:
        full_corpus (list[str]): Full list of texts to use to detect stopwords.
        percent_frequent_words (Union[int,float]): % of most frequent words to look for. Defaults to 1.5%
        **kwargs: keyword arguments available to inner append_to_txt_file function.
    Returns:
        List[str]: List of most frequent words.
    """
    if isinstance(full_corpus, str):
        full_corpus = full_corpus.split()
    most_frequent_words_list = most_frequent_words(full_corpus, percent_frequent_words)
    # Appending most frequent 1.5% of words to custom stopwords. 1.5% value has been found through trial and error.
    filler_words = ['like', 'yeah', 'um', 'eh', 'actually', 'see', 'well', 'er',
                    'said', 'right'] #spoken word fillers
    append_to_stopwords = ",".join(most_frequent_words_list + filler_words)
    append_to_txt_file(append_to_stopwords, r"custom_stopwords.txt",**kwargs)
    return list_from_text(r"custom_stopwords.txt")


def main():
    NUM_ROWS = 20
    # Training Parameters
    num_topics = 10
    passes = 20
    iterations = 400
    eval_every = None

    config_dict = read_toml(r"db_info.toml")["database"] #config dict to access db
    texts = read_transcripts(config_dict, row_limit=NUM_ROWS)
    custom_stopwords = prepare_stopwords(full_corpus=texts, erase = True)
    docs_clean = [clean_text(doc, custom_stopwords).split() for doc in texts]
    ldamodel, doc_term_matrix = run_LDA(docs=docs_clean,
                       num_topics=num_topics,
                       passes=passes,
                       iterations = iterations,
                       eval_every = eval_every,
                       alpha = 'auto',
                       eta = 'auto')
    top_topics = ldamodel.top_topics(doc_term_matrix)
    pprint(top_topics)


if __name__ == "__main__":
    main()
