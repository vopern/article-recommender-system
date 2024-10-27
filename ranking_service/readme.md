# Backand API for recommendations

Build with FastAPI, this service 
- provides reranking for search results from the arxiv api (not personalized) based on semantic similarity
of embeddings
- personalized recommendations for a user_id

In particular
- A scoring model is served
- features and candidate documents are loaded
- for the model, features are retrieved from the online feature store or calculated on the fly.
