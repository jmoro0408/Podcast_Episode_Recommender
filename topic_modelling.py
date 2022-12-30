from pprint import pprint

import gensim
from preprocessing import preprocess_main


def run_LDA(docs: list[str],
            index_dictionary:gensim.corpora.Dictionary,
            **kwargs) -> tuple[gensim.models.ldamodel.LdaModel, list]:
    """Creates and runs the LDA model.

    Args:
        docs (list[str]): List of text (strings) to be used in the analysis.
        **kwargs: Other gensim LDA model arguments/hyperparameters

    Returns:
        gensim.models.ldamodel.LdaModel: Trained LDA model
        doc_term_matrix (list): document term matrix for the corpus
    """
    doc_term_matrix = [index_dictionary.doc2bow(doc) for doc in docs]
    Lda = gensim.models.ldamodel.LdaModel
    return Lda(doc_term_matrix, id2word=index_dictionary, **kwargs), doc_term_matrix


def main():
    # Preprocessing text
    NUM_ROWS = 20 #no. rows (episodes) to grab from db
    docs_clean, index_dictionary = preprocess_main(num_rows_db = NUM_ROWS)
    # Training Parameters
    num_topics = 10
    passes = 20
    iterations = 400
    eval_every = None

    ldamodel, doc_term_matrix = run_LDA(
        docs=docs_clean,
        index_dictionary=index_dictionary,
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
