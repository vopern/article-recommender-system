# Scoring model

## Description
Script to retrain the scoring model and a class to load it in the backend for serving.
The model should score a set of candidate documents for a user based on document features, user features
and user-document features.

For a new model
- Prepare features from the logs and the documents. 
- Run the training script
This step could also be containerized and scheduled (with a cron job or another workflow tool).

The model here is a very simple content-based model. Since there is no user data here, it doesn't 
really make sense to implement a more complex one, but if it were to be extended, one could go
for a multi-task model to predict click-outs and detail views. 
It mostly has to recommend new articles, so a purely collaborative filtering approach might not work well. 




