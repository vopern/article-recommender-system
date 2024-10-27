# This is an example feature definition file

from datetime import timedelta
import os
import sys
sys.path.append('..')
import utils

from feast import (
    Entity,
    FeatureView,
    Field,
    FileSource,
)
from feast.types import Float32, Float64, Int64, String

local_file_paths = {'user': '../feature_store/data/latest_user_features.parquet',
                    'documents': '../feature_store/data/latest_document_features.parquet'}

# Define an entity for the driver. You can think of an entity as a primary key used to
# fetch features.
user = Entity(name="user", join_keys=["user_id"])
document = Entity(name="document", join_keys=["entry_id"])

# Read data from parquet files. Parquet is convenient for local development mode. For
# production, you can use your favorite DWH, such as BigQuery. See Feast documentation
# for more info.
user_stats_source = FileSource(
    name="user_stats_source",
    path=local_file_paths['user'],
    timestamp_field="date_ts"
)

document_stats_source = FileSource(
    name="document_stats_source",
    path=local_file_paths['user'],
    timestamp_field="published_ts"
)

# Our parquet files contain sample data that includes a user_id column, and
# feature columns. Here we define a Feature View that will allow us to serve this
# data to our model online.
user_stats_fv = FeatureView(
    # The unique name of this feature view. Two feature views in a single
    # project cannot have the same name
    name="user_daily_stats",
    entities=[user],
    ttl=timedelta(days=1),
    # The list of features defined below act as a schema to both define features
    # for both materialization of features into a store, and are used as references
    # during preprocessing for building a scoring_model dataset or serving features
    schema=[
        Field(name="query", dtype=String, description="Searched queries"),
        Field(name="title", dtype=String, description="Results with interactions"),
    ],
    online=True,
    source=user_stats_source
)

document_stats_fv = FeatureView(
    name="document_daily_stats",
    entities=[document],
    ttl=timedelta(days=10),
    schema=[
        Field(name="click", dtype=Int64, description="Total user clickouts"),
        Field(name="expand", dtype=Int64, description="Total user detail views"),
        Field(name="title", dtype=String, description="Article title"),
    ],
    online=True,
    source=document_stats_source
)