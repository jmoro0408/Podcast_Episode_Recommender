import logging
import pickle

import gensim
import matplotlib.pyplot as plt
from gensim.models import CoherenceModel, LdaModel

from utils import load_lda_model

logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s", level=logging.DEBUG
)


def evaluate_graph(
    dictionary: gensim.corpora.Dictionary,
    corpus: list[list],
    texts: list[str],
    limit: int,
):
    """
    Function to investigate the coherance of a various number of topics.

    Arguments:
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts: Raw texts
    limit : limit of topics to investigate

    Returns:
    -------
    lm_list : List of LDA topic models
    c_v : Coherence values corresponding to the LDA model with respective number of topics
    """
    c_v = []
    lm_list = []
    START = 2
    STEP = 20

    for num_topics in range(START, limit, STEP):
        print(f"-----------Testing {num_topics} topics.-----------")
        lm = LdaModel(
            corpus=corpus,
            num_topics=num_topics,
            id2word=dictionary,
            passes=3,
            iterations=400,
            chunksize=2044,
        )
        lm_list.append(lm)
        cm = CoherenceModel(
            model=lm, texts=texts, dictionary=dictionary, coherence="u_mass"
        )
        c_v.append(cm.get_coherence())

    # Show graph
    x = range(START, limit, STEP)
    plt.plot(x, c_v)
    plt.xlabel("num_topics")
    plt.ylabel("Coherence score")
    plt.legend(("u_mass"), loc="best")
    plt.savefig("Coherance_plot.png", format="png", facecolor="white")
    plt.show()

    return lm_list, c_v


if __name__ == "__main__":
    lda = load_lda_model(
        r"/Users/jamesmoro/Documents/Python/Podcast_Episode_Recommender/Results/model"
    )
    doc_term_matrix = load_lda_model(
        r"/Users/jamesmoro/Documents/Python/Podcast_Episode_Recommender/Results/model.id2word"
    )
    with open("cleaned_docs.pkl", "rb") as f:
        docs = pickle.load(f)
    with open("index_dict.pkl", "rb") as f:
        index_dict = pickle.load(f)

    corpus = [index_dict.doc2bow(doc) for doc in docs]
    lm_list, c_v = evaluate_graph(
        dictionary=index_dict, corpus=corpus, texts=docs, limit=200
    )
