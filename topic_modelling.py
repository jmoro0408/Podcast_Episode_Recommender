import logging

logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s", level=logging.DEBUG
)

import gensim

from preprocessing import preprocess_main


def run_LDA(
    docs: list[str], index_dictionary: gensim.corpora.Dictionary, **kwargs
) -> tuple[gensim.models.ldamodel.LdaModel, list]:
    """Creates and runs the LDA model.

    Args:
        docs (list[str]): List of text (strings) to be used in the analysis.
        **kwargs: Other gensim LDA model arguments/hyperparameters

    Returns:
        gensim.models.ldamodel.LdaModel: Trained LDA model
    """
    doc_term_matrix = [index_dictionary.doc2bow(doc) for doc in docs]
    Lda = gensim.models.ldamodel.LdaModel
    return Lda(doc_term_matrix, id2word=index_dictionary, **kwargs)


def main():
    """Main func to preprocess and subsequently run the LDA model."""
    # Preprocessing text
    NUM_ROWS = None  # no. rows (episodes) to grab from db
    docs_clean, index_dictionary = preprocess_main(
        num_rows_db=NUM_ROWS, save_preprocessed_text=True
    )
    # Training Parameters
    num_topics = 150
    passes = 20
    iterations = 400
    eval_every = 10

    ldamodel = run_LDA(
        docs=docs_clean,
        index_dictionary=index_dictionary,
        num_topics=num_topics,
        passes=passes,
        iterations=iterations,
        eval_every=eval_every,
        alpha="auto",
        eta="auto",
    )
    ldamodel.save(r"Results/model")


if __name__ == "__main__":
    main()
