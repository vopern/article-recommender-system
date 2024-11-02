import joblib
import numpy as np
from sentence_transformers import SentenceTransformer


def load_scoring_model(model_path):
    """
    Load sentence transformer model
    """
    model = joblib.load(model_path)
    return model


def load_model(model_path):
    """
    Load sentence transformer model
    """
    model = SentenceTransformer(model_path)
    return model


def similarity(l1, l2):
    """
    Pointwise, between two lists of embeddings
    """
    l1 = np.array(l1)
    if l1.shape[0] == 1:
        l1 = np.repeat(l1, l2.shape[0], axis=0)
    l2 = np.array(l2)
    sim = (l1 * l2).sum(axis=1)
    return sim.tolist()


class RecommendationModel:

    def __init__(self, embedding_model, scoring_model):
        self.embedding_model = embedding_model
        self.scoring_model = scoring_model

    def rerank_list(self, query, ids, titles):
        """
        Rerank document ids by embedding similarity between query and title.
        A personalized ranking could be implemented as well.
            E.g. use the recommender model to score known ids. Features of new documents would have to be imputed,
            and the query be incorporated as an additional feature.
        """
        query_embedding = self.embedding_model.encode([query])
        title_embeddings = self.embedding_model.encode(titles)
        similarities = similarity(query_embedding, title_embeddings)
        sorted_ids = [id for _, id in sorted(zip(similarities, ids), reverse=True)]
        return sorted_ids

    def score(self, document_features, user_features):
        """
        Return score of a document for a user. This is a pointwise model scoring each document individually.
        """
        document_embeddings = np.array(document_features['title_embeddings'].values.tolist())

        query_embeddings = self.embedding_model.encode([user_features['query']])
        document_features['qe_score'] = similarity(document_embeddings, query_embeddings)

        title_embeddings = self.embedding_model.encode([user_features['title']])
        document_features['te_score'] = similarity(document_embeddings, title_embeddings)

        document_features['score'] = self.scoring_model.predict_proba(document_features[['qe_score', 'te_score']])[:, 1]
        document_features.sort_values(by='score', ascending=False, inplace=True)
        return document_features