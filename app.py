from flask import Flask, request, jsonify, render_template
from elasticsearch import Elasticsearch
import pandas as pd
import time
import re
from IndexerTFIDF import IndexerTFIDF, highlight_and_trim
from IndexerTFIDF import Pr

app = Flask(__name__, template_folder='.')
app.es_client = Elasticsearch(
    "https://localhost:9200",
    basic_auth=("elastic", "M8OIu*UpfhnRm10hmp*X"),
    ca_certs="C:/Users/acer/All SE/IR/M2/http_ca.crt"
)

indexer = IndexerTFIDF(is_reset=False)

import re

def highlight_query_terms(text, query):
    """
    Highlights query terms in the text using <b> tags.
    """
    query_terms = query.split()
    for term in query_terms:
        text = re.sub(rf"\b({re.escape(term)})\b", r"<b>\1</b>", text, flags=re.IGNORECASE)
    return text

def extract_surrounding_text(text, query, max_sentences=5):
    """
    Extracts up to max_sentences surrounding the first occurrence of the query term.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text)  # Split text into sentences while keeping punctuation
    query_terms = query.lower().split()

    # Find the sentence index that contains any query term
    for i, sentence in enumerate(sentences):
        if any(term in sentence.lower() for term in query_terms):
            start = max(0, i - 1)  # Start from the sentence before, if possible
            end = min(len(sentences), i + 2)  # Include one after
            selected_sentences = sentences[start:end]
            break
    else:
        selected_sentences = sentences[:max_sentences]  # Default to first few if no match

    return ' '.join(selected_sentences)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/search', methods=['GET'])
def search():
    query_term = request.args.get('query')
    start_bm25 = time.time()

    # BM25 + PageRank search
    bm25_results = app.es_client.search(
        index='simple',
        source_excludes=['url_lists'],
        size=100,
        query={
            "script_score": {
                "query": {"match": {"text": query_term}},
                "script": {"source": "_score * doc['pagerank'].value"}
            }
        },
        highlight={
            "fields": {
                "text": {}  # This will highlight the "text" field in the results
            },
            "pre_tags": ["<em>"],  # Tags to wrap around highlighted query terms
            "post_tags": ["</em>"]  # Closing tag for the highlighted terms
        }
    )

    bm25_results_list = []
    for hit in bm25_results['hits']['hits']:
        # Get the text and apply custom highlight and trim
        text = hit["_source"]['text']
        highlighted_text = highlight_and_trim(text, query_term)

        bm25_results_list.append({
            'title': hit["_source"]['title'],
            'url': hit["_source"]['url'],
            'text': highlighted_text
        })

    end_bm25 = time.time()
    start_tfidf = time.time()

    # Custom TF-IDF + PageRank search
    tfidf_results = indexer.query(query_term)
    tfidf_results_list = []
    for _, row in tfidf_results.iterrows():
        text = extract_surrounding_text(row['text'], query_term)  # Keep this for TF-IDF results
        text = highlight_query_terms(text, query_term)
        tfidf_results_list.append({
            'title': row['title'],
            'url': row['url'],
            'text': text
        })

    end_tfidf = time.time()
    total_hit_bm25 = len(bm25_results['hits']['hits'])
    total_hit_tfidf = len(tfidf_results)
    elapse_bm25 = end_bm25 - start_bm25
    elapse_tfidf = end_tfidf - start_tfidf

    return jsonify({
        'query': query_term,
        'total_hit_bm25': total_hit_bm25,
        'total_hit_tfidf': total_hit_tfidf,
        'elapse_bm25': elapse_bm25,
        'elapse_tfidf': elapse_tfidf,
        'bm25_results': bm25_results_list,
        'tfidf_results': tfidf_results_list
    })

if __name__ == '__main__':
    app.run(debug=True)