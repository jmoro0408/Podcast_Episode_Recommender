import logging

logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s", level=logging.DEBUG
)
import pickle
from pprint import pprint
from typing import Union

from gensim.matutils import cossim
from utils import load_lda_model, read_titles, read_toml

MODEL_DIR = (
    r"/Users/jamesmoro/Documents/Python/Podcast_Episode_Recommender/Results/model"
)
ID_2_WORD_DIR = r"/Users/jamesmoro/Documents/Python/Podcast_Episode_Recommender/Results/model.id2word"


def get_cosine_similarity(lda_model, doc1: list[tuple], doc2: list[tuple]) -> float:
    """Calculate cosine distance between two docs.
    Input docs should be bag of words i.e
    corpus = [index_dict.doc2bow(doc) for doc in docs]
    doc1 = corpus[0]

    Args:
        doc1 (list): first doc to compare. BoW representation.
        doc2 (list): second doc to compare. BoW representation.

    Returns:
        float: Similarity between docs. Higher = more similiar.
    """
    doc1 = lda_model.get_document_topics(doc1)
    doc2 = lda_model.get_document_topics(doc2)
    return cossim(doc1, doc2)


def find_similar_episodes(
    saved_lda_model_dir: str,
    episode_to_compare: Union[int, str],
    corpus: list[tuple],
    raw_titles: list[str],
    top_n: int = 5,
) -> dict:
    """
    Compares a selected episode to all in corpus and returns the top_n most similar by
    cosine similarity.

    Args:
        episode_to_compare (Union[int, str]): Title of episode or row of index of episode within corpus.
        If supplied as a string, spelling must match the db record exactly.
        corpus (list[tuple]): Bag of words corpus representation.
        i.e [index_dict.doc2bow(doc) for doc in docs]
        raw_titles (list[str]): List of corpus episode titles.
        top_n (int, optional): Number of most similar documents to return. Defaults to 5.

    Returns:
        dict: Dictionary containing most similar episodes and their similarity score.
    """
    if isinstance(episode_to_compare, str):
        try:
            episode_to_compare = raw_titles.index(episode_to_compare)
        except ValueError:
            print("Episode not found, please ensure the spelling is totally correct.")
            return None
    similarity_dict = {}
    lda = load_lda_model(saved_lda_model_dir)
    for idx, doc in enumerate(corpus):
        if idx % 10 == 0:
            print(idx)
        similarity_dict[idx] = get_cosine_similarity(lda, corpus[episode_to_compare], doc)
    sorted_similarity = dict(
        sorted(similarity_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
    )
    similarity_values_list = list(sorted_similarity.values())
    similarity_episodes_list = list(sorted_similarity.keys())
    episode_titles = {}
    for episode, value in zip(similarity_episodes_list, similarity_values_list):
        episode_titles[raw_titles[episode]] = value
    print(f"{raw_titles[episode_to_compare]}:")
    return episode_titles


if __name__ == "__main__":
    EPISODE_TITLE = 'Cleopatra: Ms. Understood'
    # EPISODE_TITLE = 25
    with open("cleaned_docs.pkl", "rb") as f:
        docs = pickle.load(f)
    with open("corpus.pkl", "rb") as f:
        corpus = pickle.load(f)

    config_dict = read_toml(r"db_info.toml")["database"]

    raw_titles = read_titles(config_dict, None)

    pprint(
        find_similar_episodes(
            saved_lda_model_dir=MODEL_DIR,
            episode_to_compare=EPISODE_TITLE,
            corpus=corpus,
            raw_titles=raw_titles,
            top_n=10,
        ),
        sort_dicts=False,
    )
