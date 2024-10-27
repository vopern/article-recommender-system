import os
import json
import streamlit as st
import arxiv
import requests
from types import SimpleNamespace
from streamlit.logger import get_logger
import webbrowser

import logging
import datetime


@st.cache_resource
def get_my_logger(fpath):
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")

    logger = get_logger(__name__)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    fname = os.path.join(fpath, f"{current_date}.log")
    file_handler = logging.FileHandler(fname, mode='a')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def search_arxiv(query):
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=10,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending
    )
    results = client.results(search)
    results = list(results)
    return results


def get_recommendation(backend, user_id):
    endpoint = f"{backend}/recommendations?"
    params = {'user_id':user_id}
    results = requests.get(endpoint, params=params).json()
    result_objects = [SimpleNamespace(**result) for result in results]
    return result_objects


def rerank(backend, results, query):
    endpoint = f"{backend}/rerank"
    result_map = {result.entry_id: result for result in results}
    payload = {'result_ids': [result.entry_id for result in results],
               'titles': [result.title for result in results],
               'query': query}
    response = requests.post(endpoint, json=payload)
    reranked_ids = response.json()
    reranked_results = [result_map[rid] for rid in reranked_ids]
    return reranked_results


def track(user_id, action, query=None, result=None):
    info = {'user_id': user_id, 'action':action}
    if result:
        info['result'] = result
    if query:
        info['query'] = query
    logger.info(json.dumps(info))


def display_result(result, user_id, query):
    author_names = [str(author) for author in result.authors]
    col1, col2 = st.columns([1, 6])
    with col1:
        on = st.toggle(label='Details', key=result.entry_id, label_visibility="collapsed")
    with col2:
        st.write(f"{', '.join(author_names)}:\n\r **{result.title}**")
    st.divider()

    if on:
        track(user_id=user_id, action='expand', query=query, result=result.entry_id)
        with st.expander(result.title, expanded=True):
            st.write(f"**Authors:** {', '.join([str(author) for author in result.authors])}")
            published = result.published if type(result.published) == str else result.published.strftime('%Y-%m-%d')
            st.write(f"**Published:** {published}")
            updated = result.updated if type(result.updated) == str else result.updated.strftime('%Y-%m-%d')
            st.write(f"**Updated:** {updated}")
            st.write(f"**Primary Category:** {result.primary_category}")
            st.write(f"**Summary:**\n{result.summary}")
            clickout = st.button(f"**URI:**\n{result.entry_id}")
            if clickout:
                track(user_id=user_id, action='click', query=query, result=result.entry_id)
                webbrowser.open_new_tab(result.entry_id)
            if result.comment:
                st.write(f"**Comment:** {result.comment}")
            if result.journal_ref:
                st.write(f"**Journal Reference:** {result.journal_ref}")
            if result.doi:
                st.write(f"**DOI:** {result.doi}")
    else:
        track(user_id=user_id, action='impress', query=query, result=result.entry_id)

logger = get_my_logger(fpath='../data/frontend')

if os.environ.get('RUNNING_IN_DOCKER'):
    backend = 'http://ranking-service:8000'
else:
    backend = 'http://localhost:8000'

with st.sidebar:
    user_id = st.text_input("Enter your user id for recommendations")
    query = st.text_input("(Optional) Search Query")
    go = st.button('Recommend or Search')

recommend = len(user_id) > 0 and len(query) == 0
search = len(query) > 0

if recommend:
    st.title('Recommended arXiv Publications')
    st.subheader(f'For {user_id}')
    results = get_recommendation(backend, user_id)
    track(user_id=user_id, action='recommendations')
    if results:
        st.subheader("Latest Papers:")
        for result in results:
            display_result(result, user_id, query)
    else:
        st.warning(f"No recommendations for user {user_id}.")

elif search:
    st.title('Results from arXiv Publications')
    st.subheader(f'Searched for: {query}')
    results = search_arxiv(query)
    track(user_id=user_id, action='search', query=query)
    if results:
        results = rerank(backend, results, query)
        st.subheader("Latest Papers:")
        for result in results:
            display_result(result, user_id, query)
    else:
        st.warning("No results found for the given query.")

else:
    st.text('Enter User-Id or Search Query')