import logging

logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s", level=logging.DEBUG
)

import gensim

from preprocessing import preprocess_main


def run_LDA(
    corpus: list[str], id2word: gensim.corpora.Dictionary, **kwargs
) -> tuple[gensim.models.ldamodel.LdaModel, list]:
    """Creates and runs the LDA model.

    Args:
        corpus (list[str]): List of text (strings) to be used in the analysis.
        id2word (gensim.corpora.Dictionary): index to word dictionary.
        **kwargs: Other gensim LDA model arguments/hyperparameters

    Returns:
        gensim.models.ldamodel.LdaModel: Trained LDA model
    """
    Lda = gensim.models.ldamodel.LdaModel
    return Lda(corpus, id2word=id2word, **kwargs)


def main():
    """Main func to preprocess and subsequently run the LDA model."""
    # Preprocessing text
    NUM_ROWS = None  # no. rows (episodes) to grab from db
    corpus, index_dictionary = preprocess_main(
        num_rows_db=NUM_ROWS, save_preprocessed_text=True
    )
    _ = index_dictionary[0]  # This is only to "load" the dictionary.
    id2word = index_dictionary.id2token
    # Training Parameters
    num_topics = 20
    passes = 10
    iterations = 200
    eval_every = 10

    ldamodel = run_LDA(
        corpus=corpus,
        id2word=id2word,
        num_topics=num_topics,
        passes=passes,
        iterations=iterations,
        chunksize=2044,
        eval_every=eval_every,
        alpha="auto",
        eta="auto",
    )
    ldamodel.save(r"Results/model")


if __name__ == "__main__":
    main()
