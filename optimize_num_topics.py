import logging
import pickle

import gensim
import matplotlib.pyplot as plt
from gensim.models import CoherenceModel, LdaModel

from preprocessing import preprocess_main

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
    START = 10
    STEP = 20
    _ = dictionary[0]  # This is only to "load" the dictionary.
    id2word = dictionary.id2token

    for num_topics in range(START, limit, STEP):
        print(f"-----------Testing {num_topics} topics.-----------")

        passes = 15
        iterations = 200
        eval_every = 10
        alpha = "auto"
        eta = "symmetric"

        lm = LdaModel(
            corpus=corpus,
            id2word=id2word,
            num_topics=num_topics,
            passes=passes,
            iterations=iterations,
            chunksize=2044,
            eval_every=eval_every,
            alpha=alpha,
            eta=eta,
        )

        lm_list.append(lm)
        cm = CoherenceModel(
            model=lm,
            texts=texts,
            dictionary=dictionary,
            coherence="c_v",
            window_size = 100,
        )
        c_v.append(cm.get_coherence())

    # Show graph
    x = range(START, limit, STEP)
    # c_v plot
    plt.plot(x, c_v)
    plt.xlabel("num_topics")
    plt.ylabel("Coherence score")
    plt.legend(("C_v"), loc="best")
    plt.savefig("C_v_plot.png", format="png", facecolor="white")
    plt.show()
    pickle.dump(dict(zip(lm_list, c_v)), open("c_v_optimization_results.pkl", "wb"))
    return lm_list, c_v


if __name__ == "__main__":
    NUM_ROWS = None  # no. rows (episodes) to grab from db
    corpus, index_dictionary = preprocess_main(
        num_rows_db=NUM_ROWS, save_preprocessed_text=True
    )
    with open("cleaned_docs.pkl", "rb") as f:
        docs = pickle.load(f)

    # This takes around 7 hrs to run with 500 topics
    lm_list, c_v, = evaluate_graph(
        dictionary=index_dictionary, corpus=corpus, texts=docs, limit=300
    )