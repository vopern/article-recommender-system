import yaml
import os
from sentence_transformers import SentenceTransformer
import datetime


def date_to_str(date: datetime.date) -> str:
    return date.strftime('%Y-%m-%d')


def today_str() -> str:
    return date_to_str(datetime.date.today())


def str_to_date(date: str) -> datetime.date:
    return datetime.datetime.strptime(date, '%Y-%m-%d').date()


def last_date_strings(num_days: int) -> list[str]:
    return [date_to_str(datetime.date.today() - datetime.timedelta(days=i)) for i in range(num_days)]


def base_folder() -> str:
    current_dir = os.path.dirname(__file__)
    return current_dir


def load_config() -> dict:
    current_dir = os.path.dirname(__file__)
    with open(os.path.join(current_dir, "config.yaml"), "r") as f:
        config = yaml.safe_load(f)
    return config


def load_embedding_model(config: dict) -> SentenceTransformer:
    hf_model_name = config['hf_model_name']
    cache_folder = os.path.join(base_folder(), config['model_cache_folder'])
    print(f'Loading embedding model {hf_model_name}')
    model = SentenceTransformer(hf_model_name, cache_folder=cache_folder)
    print(f'...done')
    return model