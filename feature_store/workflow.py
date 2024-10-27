import subprocess
from datetime import datetime

from feast import FeatureStore


def fetch_online_features(store):
    entity_rows = [
        {
            "entry_id": "http://arxiv.org/abs/2410.16579v1",
        },
        {
            "entry_id": "http://arxiv.org/abs/2410.16574v1",
        },
        {"entry_id": "http://arxiv.org/abs/2410.16270v1"},
    ]
    features_to_fetch = [
        "document_daily_stats:click",
        "document_daily_stats:expand",
        "document_daily_stats:title",
    ]
    returned_features = store.get_online_features(
        features=features_to_fetch,
        entity_rows=entity_rows,
    ).to_dict()
    for key, value in sorted(returned_features.items()):
        print(key, " : ", value)


def run_demo():
    store = FeatureStore(repo_path=".")
    print("\n--- Run feast apply ---")
    subprocess.run(["feast", "apply"])

    print("\n--- Load features into online store ---")
    store.materialize_incremental(end_date=datetime.now())

    print("\n--- Online features ---")
    fetch_online_features(store)

if __name__ == "__main__":
    run_demo()