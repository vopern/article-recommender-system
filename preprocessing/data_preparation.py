import os
import sys
import datetime
import arxiv
import pandas as pd
import re
import json

sys.path.append('..')
import utils

OUTPUT_DIR = os.path.join(utils.base_folder(), "data", "preprocessing")
FS_DIR = os.path.join(utils.base_folder(), "feature_store", "data")
LOG_DIR = os.path.join(utils.base_folder(), "data", "frontend")


def fetch(client, topic, day):
    """
    Fetch article metadata from the arxiv
    """
    qdate = day.strftime("%Y%m%d")
    date = day.strftime("%Y-%m-%d")

    all_records = []
    search = arxiv.Search(
        query=f"cat:{topic} AND submittedDate:[{qdate}0000 TO {qdate}2359]",
        max_results=1000,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending
    )
    results = client.results(search)
    for result in results:
        record = {
            "entry_id": result.entry_id,
            "updated": result.updated.strftime("%Y-%m-%d"),
            "published_ts": result.published,
            "published": result.published.strftime("%Y-%m-%d"),
            "title": result.title,
            "authors": [author.name for author in result.authors],
            "categories": result.categories,
            "comment": result.comment,
            "primary_category": result.primary_category,
            "journal_ref": result.journal_ref,
            "summary": result.summary,
            "doi": result.doi,
            "submitted": date,
        }
        all_records.append(record)
    df = pd.DataFrame(all_records)
    return df


def parse_logs(day):
    """
    Parse dictionary part in lines in log files to a dataframe.
    Neglect duplicates from streamlit reloading the site.
    """
    log_files = [os.path.join(LOG_DIR, f) for f in os.listdir(LOG_DIR) if f.endswith('.log') and day in f]
    LOG_PATTERN = r'\{.+?\}'
    logs = []
    for log_file in log_files:
        with open(log_file, 'r') as f:
            for line in f:
                match = re.search(LOG_PATTERN, line)
                if match:
                    data = json.loads(match.group())
                    logs.append(data)
    if logs:
        return pd.DataFrame(logs).drop_duplicates()
    return


def prepare_raw_data(download_all, config, date_str: str):
    """
    Process user interaction logs and download document metadata from the arxiv.
    """
    print('Preparing processed logs and arxiv metadata')
    date = utils.str_to_date(date_str)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    topics = config['preprocessing']['topics']
    num_days = config['preprocessing']['history_days']
    for i in range(num_days):
        day_str = utils.date_to_str(date - datetime.timedelta(days=i))
        log_data = parse_logs(day_str)
        if log_data is not None:
            output_file = os.path.join(OUTPUT_DIR, f"{day_str}_logs.parquet")
            log_data.to_parquet(output_file)

    client = arxiv.Client()
    for i in range(num_days):
        day = date - datetime.timedelta(days=i)
        output_file = os.path.join(OUTPUT_DIR, f"{utils.date_to_str(day)}_arxiv.parquet")
        if os.path.exists(output_file) and not download_all:
            print(f'Data for {day} already downloaded')
            continue
        dfs = []
        for topic in topics:
            print(f"Fetching documents for topic: {topic} on day {day}")
            df = fetch(client, topic, day)
            dfs.append(df)
        dfs = pd.concat(dfs)

        dfs.to_parquet(output_file, engine="pyarrow")
        print(f"Loaded {len(dfs)} records for date {day}")
        print(f"Data saved to {output_file}")


def load_files(num_days, fid, date_str, include_date=True):
    """
    Load data from the last few days from preprocessed files
    """
    data = []
    files = os.listdir(OUTPUT_DIR)
    files.sort()
    if include_date:
        files = [f for f in files if f[0:10] <= date_str and fid in f]
        files = [f for f in files[-num_days + 1:]]
    else:
        files = [f for f in files if f[0:10] < date_str and fid in f]
        files = [f for f in files[-num_days:]]

    print(f'Loading from files {files}')
    for f in files:
        fname = os.path.join(OUTPUT_DIR, f)
        df = pd.read_parquet(fname)
        df['date'] = f[0:10]
        data.append(df)
    data = pd.concat(data)
    return data


def prepare_features(config, date_str):
    """
    Prepare today's user and document features based on recent history.
    """
    history_days = config['preprocessing']['history_days']
    print('Preparing features')
    logs = load_files(history_days, 'logs', date_str, include_date=False)
    logs['click'] = logs['action'] == 'click'
    logs['expand'] = logs['action'] == 'expand'
    logs['impress'] = logs['action'] == 'impress'
    log_doc_features = logs.groupby(['result']).agg({'click':'sum', 'expand': 'sum', 'impress': 'sum'}).reset_index()
    documents = load_files(history_days, 'arxiv', date_str)

    print('Preparing document features')
    model = utils.load_embedding_model(config)

    def column_to_embeddings(column_list, model):
        embeddings = model.encode(column_list)
        embeddings = [embeddings[i, :].tolist() for i in range(len(column_list))]
        return embeddings

    documents['title_embeddings'] = column_to_embeddings(documents["title"].tolist(), model)
    documents = pd.merge(documents, log_doc_features, left_on='entry_id', right_on='result', how='left')
    documents[['click', 'expand', 'impress']] = documents[['click', 'expand', 'impress']].fillna(0)

    documents.to_parquet(os.path.join(OUTPUT_DIR, f'{date_str}_document_features.parquet'))
    documents.to_parquet(os.path.join(FS_DIR, f'latest_document_features.parquet'))
    print('Preparing user features')
    logs = pd.merge(logs, documents[['entry_id', 'title']], left_on='result', right_on='entry_id', how='left')
    user_features = logs.groupby(['user_id']).agg({'query':list, 'title': list}).reset_index()

    def compress_list(l):
        if l is None:
            return ''
        l = set(l)
        l = [i for i in l if type(i) == str and len(l) > 0]
        s = ','.join(l)
        return s

    user_features['query'] = user_features['query'].apply(compress_list)
    user_features['user_query_embeddings'] = column_to_embeddings(user_features["query"].tolist(), model)
    user_features['title'] = user_features['title'].apply(compress_list)
    user_features['user_title_embeddings'] = column_to_embeddings(user_features["title"].tolist(), model)
    user_features['date_ts'] = datetime.datetime.strptime(date_str, '%Y-%m-%d')
    user_features.to_parquet(os.path.join(OUTPUT_DIR, f'{date_str}_user_features.parquet'))
    user_features.to_parquet(os.path.join(FS_DIR, f'latest_user_features.parquet'))


def main(date, download_all=False):
    """
    Prepare features for date
    Consider logs up until yesterday and documents including today
    """
    config = utils.load_config()
    prepare_raw_data(download_all, config, date)
    prepare_features(config, date)


if __name__ == "__main__":
    main(date=datetime.date.today().strftime('%Y-%m-%d'))