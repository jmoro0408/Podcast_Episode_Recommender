import logging
import pickle

import gensim
import matplotlib.pyplot as plt
from gensim.models import CoherenceModel, LdaModel

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
    u_mass = []
    lm_list = []
    START = 2
    STEP = 20
    _ = dictionary[0]  # This is only to "load" the dictionary.
    id2word = dictionary.id2token

    for num_topics in range(START, limit, STEP):
        print(f"-----------Testing {num_topics} topics.-----------")
        lm = LdaModel(
            corpus=corpus,
            num_topics=num_topics,
            id2word=id2word,
            passes=20,
            iterations=400,
            chunksize=2044,
            alpha="auto",
            eta="auto",
        )
        lm_list.append(lm)
        cm = CoherenceModel(
            model=lm,
            texts=texts,
            dictionary=dictionary,
            coherence="c_v",
        )
        c_v.append(cm.get_coherence())

        u_mass_model = CoherenceModel(
            model=lm,
            texts=texts,
            dictionary=dictionary,
            coherence="u_mass",
        )
        u_mass.append(u_mass_model.get_coherence())

    # Show graph
    x = range(START, limit, STEP)
    # c_v plot
    plt.plot(x, c_v)
    plt.xlabel("num_topics")
    plt.ylabel("Coherence score")
    plt.legend(("C_v"), loc="best")
    plt.savefig("C_v_plot.png", format="png", facecolor="white")
    plt.show()
    # u_mass plot
    plt.plot(x, u_mass)
    plt.xlabel("num_topics")
    plt.ylabel("Coherence score")
    plt.legend(("u_mass"), loc="best")
    plt.savefig("u_mass_plot.png", format="png", facecolor="white")
    plt.show()
    pickle.dump(dict(zip(lm_list, c_v)), open("c_v_optimization_results.pkl", "wb"))
    pickle.dump(
        dict(zip(lm_list, u_mass)), open("u_mass_optimization_results.pkl", "wb")
    )

    return lm_list, c_v, u_mass


if __name__ == "__main__":
    with open("cleaned_docs.pkl", "rb") as f:
        docs = pickle.load(f)
    with open("corpus.pkl", "rb") as f:
        corpus = pickle.load(f)
    with open("index_dict.pkl", "rb") as f:
        index_dict = pickle.load(f)
    # This takes around 7 hrs to run with 500 topics
    lm_list, c_v, u_mass = evaluate_graph(
        dictionary=index_dict, corpus=corpus, texts=docs, limit=500
    )
