"""
Train a model for ranking recommendations based on user interactions and document features.
Options would be an NN-based multitask model, or pair-wise/list-wise ranking models
or even prompt-based prediction to an LLM API.

Here this is just a dummy-model for testing the system.
"""

import joblib
import os

import pandas as pd


from sklearn.utils import resample
from sklearn.linear_model import LogisticRegression
import utils

from model_utils import similarity


def upsample(train_df):
    """
    Upsample results with user interactions to remove class imbalance.
    """
    df_majority = train_df[train_df['label'] == 0]
    df_minority = train_df[train_df['label'] == 1]

    df_minority_upsampled = resample(df_minority,
                                     replace=True,
                                     n_samples=len(df_majority),
                                     random_state=42)

    return pd.concat([df_majority, df_minority_upsampled])


def trainings_data(num_days):
    """
    Prepare trainings data from logs and historicl user features and document features.
    Features should be the same used for inference in the backend and might be prepared in the feature store
    alternatively.
    """
    folder = os.path.join(utils.base_folder(), 'data', 'preprocessing')
    train_df = []
    for ds in utils.last_date_strings(num_days):
        try:
            df = pd.read_parquet(os.path.join(folder, f'{ds}_document_features.parquet'))
            uf = pd.read_parquet(os.path.join(folder, f'{ds}_user_features.parquet'))
            logs = pd.read_parquet(os.path.join(folder, f'{ds}_logs.parquet'))
        except FileNotFoundError as e:
            print(f'No data for {ds}, skipping')
            continue

        tdf = pd.merge(logs, uf, on='user_id', how='left')
        tdf = pd.merge(tdf, df[['entry_id', 'title_embeddings']], left_on='result', right_on='entry_id', how='left', )
        tdf['label'] = tdf['action'].isin(['click', 'expand']).astype(int)
        tdf.dropna(subset=['user_id', 'entry_id'], inplace=True)
        tdf['qe_score'] = similarity(tdf['user_query_embeddings'].values.tolist(), tdf['title_embeddings'].values.tolist())
        tdf['te_score'] = similarity(tdf['user_title_embeddings'].values.tolist(), tdf['title_embeddings'].values.tolist())
        train_df.append(tdf)

    train_df = pd.concat(train_df)
    train_df = upsample(train_df)
    X = train_df[['qe_score', 'te_score']]
    y = train_df['label']
    return X, y


def train_model(X, y):
    """
    Train a dummy model to score documents based on user-document interaction features.
    """
    logistic_model = LogisticRegression()
    logistic_model.fit(X, y)
    joblib.dump(logistic_model, os.path.join(utils.base_folder(), 'data', 'training', f'scoring_model_{utils.today_str()}.pkl'))


if __name__ == '__main__':
    print('Retraining model')
    X, y = trainings_data(num_days=5)
    train_model(X, y)

