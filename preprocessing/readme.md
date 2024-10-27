# Prepare data for serving and training

This script

- Fetches article metadata from the arxiv as document recommendations.
- Parses user logs to aggregate user and user-document interaction features
- Stores results to be used in the feature store for serving and as historical features for model training.

It should be run daily to update the recommendation results as well as the features used by the 
recommender model.
