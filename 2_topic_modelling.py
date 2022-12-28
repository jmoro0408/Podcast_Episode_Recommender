import string
from typing import Optional

import gensim
from gensim import corpora
from nltk.stem.wordnet import WordNetLemmatizer
from spacy.lang.en.stop_words import STOP_WORDS

from utils import list_from_text, read_from_db, read_toml


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


def run_LDA(docs: list[str], **kwargs) -> gensim.models.ldamodel.LdaModel:
    """Creates and runs the LDA model.

    Args:
        docs (list[str]): List of text (strings) to be used in the analysis.
        **kwargs: Other gensim LDA model arguments/hyperparameters

    Returns:
        gensim.models.ldamodel.LdaModel: Trained LDA model
    """
    index_dictionary = corpora.Dictionary(docs)
    doc_term_matrix = [index_dictionary.doc2bow(doc) for doc in docs]
    Lda = gensim.models.ldamodel.LdaModel
    return Lda(doc_term_matrix, id2word=index_dictionary, **kwargs)


def main():
    QUERY = """SELECT * from episodes LIMIT 50"""
    NUM_TOPICS = 5
    config_dict = read_toml(r"db_info.toml")["database"]
    df = read_from_db(QUERY, config_dict=config_dict)
    texts = df["transcript"].to_list()
    custom_stopwords = list_from_text(r"custom_stopwords.txt")
    docs_clean = [clean_text(doc, custom_stopwords).split() for doc in texts]
    ldamodel = run_LDA(docs=docs_clean, num_topics=NUM_TOPICS, passes=50)
    print(ldamodel.print_topics(num_topics=NUM_TOPICS, num_words=3))


if __name__ == "__main__":
    main()
