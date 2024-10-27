"""
Fast text application
- load configuration file
- load a sentence transformer model from folder specified in configuration file
- load document preprocessing from parquet files in a folder specified in the configuration file
- serve the model in an endpoint. The endpoint accepts as a parameter a query string, and will return the loaded documents
  sorted by similarity of the embeddings with the query string
"""

import uvicorn
import os
import sys
from typing import List, Optional
import pandas as pd

from fastapi import FastAPI, Query
from pydantic import BaseModel
import datetime
from feast import FeatureStore

sys.path.append('..')
import utils
from feature_store import feature_definitions as fd
from scoring_model.model_utils import RecommendationModel, load_model, load_scoring_model


class Document(BaseModel):
    entry_id: str
    updated: Optional[str] = None
    published: Optional[str] = None
    title: str
    authors: Optional[List[str]] = None
    categories: Optional[List[str]] = None
    comment: Optional[str] = None
    summary: Optional[str] = None
    primary_category: Optional[str] = None
    doi: Optional[str] = None
    submitted: Optional[str] = None
    journal_ref: Optional[str] = None


class QueryRequest(BaseModel):
    query: str


class RerankRequest(BaseModel):
    result_ids: list[str]
    titles: list[str]
    query: str


class FeatureHandler:

    def __init__(self, config):
        self.config = config
        self.init_feature_store()
        self.load_documents()

    def load_documents(self):
        """
        Load all documents from disc, including precomputed embeddings.
        This might be handled by the feature store in combination with a preprocessing service, to preselect
        candidates for recommendations in a more advanced setup.
        """
        self.documents = pd.read_parquet(fd.local_file_paths['documents'])

    def document_features(self):
        return self.documents.copy()

    def init_feature_store(self):
        """
        create and update feast feature store
        """
        fs_base = os.path.join(utils.base_folder(), 'feature_store')
        self.fstore = FeatureStore(repo_path=fs_base)
        self.fstore.apply(objects=[fd.user, fd.user_stats_fv, fd.user_stats_source])
        #  Load features into online store
        self.fstore.materialize_incremental(end_date=datetime.datetime.now())

    def user_features_from_store(self, user_id):
        entity_rows = [
            {
                "user_id": user_id,
            },
        ]
        features_to_fetch = [
            "user_daily_stats:query",
            "user_daily_stats:title",
        ]
        returned_features = self.fstore.get_online_features(
            features=features_to_fetch,
            entity_rows=entity_rows,
        ).to_dict()
        defaults = {'query': '', 'title': ''}
        for key, val in returned_features.items():
            returned_features[key] = val[0] if val[0] else defaults[key]
        return returned_features


app = FastAPI()
config = utils.load_config()

feature_handler = FeatureHandler(config['ranking_service'])
embedding_model = load_model(os.path.join(utils.base_folder(), config['ranking_service']["model_path"]))
scoring_model = load_scoring_model(os.path.join(utils.base_folder(), config['ranking_service']["scoring_model_path"]))

recommender = RecommendationModel(embedding_model, scoring_model)


@app.post("/rerank")
def rerank(request: RerankRequest):
    result_ids = request.result_ids
    titles = request.titles
    query = request.query
    result_ids = recommender.rerank_list(query, result_ids, titles)

    return result_ids


@app.get("/recommendations")
def prepare_recommended_documents(user_id: str = Query(...)):
    sorted_documents = recommender.score(feature_handler.document_features(), feature_handler.user_features_from_store(user_id))
    sorted_documents = sorted_documents.to_dict(orient='records')[0:10]
    sorted_documents = [Document.model_validate(doc) for doc in sorted_documents]
    return sorted_documents


# Run the app
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
