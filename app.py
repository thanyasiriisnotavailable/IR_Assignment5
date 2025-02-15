from flask import Flask, request, jsonify, render_template
from elasticsearch import Elasticsearch
import pandas as pd
import time
import re
from IndexerTFIDF import IndexerTFIDF
from IndexerTFIDF import Pr

app = Flask(__name__, template_folder='C:/Users/acer/All SE/IR/Search_Engine/templates')
app.es_client = Elasticsearch(
    "https://localhost:9200",
    basic_auth=("elastic", "M8OIu*UpfhnRm10hmp*X"),
    ca_certs="C:/Users/acer/All SE/IR/M2/http_ca.crt"
)

indexer = IndexerTFIDF(is_reset=False)

def highlight_query_terms(text, query):
    """
    Highlights query terms in the text using <b> tags.
    """
    for term in query.split():
        text = re.sub(f"({term})", r"<b>\1</b>", text, flags=re.IGNORECASE)
    return text

def extract_surrounding_text(text, query, max_sentences=2):
    """
    Extracts two or three sentences surrounding the query term.
    """
    sentences = re.split(r'(?<=[.!?]) +', text)
    for i, sentence in enumerate(sentences):
        if query.lower() in sentence.lower():
            start = max(0, i - 1)
            end = min(len(sentences), i + 2)
            return ' '.join(sentences[start:end])
    return ' '.join(sentences[:max_sentences])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['GET'])
def search():
    query_term = request.args.get('query')
    start = time.time()

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
        }
    )
    bm25_results_list = []
    for hit in bm25_results['hits']['hits']:
        text = extract_surrounding_text(hit["_source"]['text'], query_term)
        text = highlight_query_terms(text, query_term)
        bm25_results_list.append({
            'title': hit["_source"]['title'],
            'url': hit["_source"]['url'],
            'text': text
        })

    # Custom TF-IDF + PageRank search
    tfidf_results = indexer.query(query_term)
    tfidf_results_list = []
    for _, row in tfidf_results.iterrows():
        text = extract_surrounding_text(row['text'], query_term)
        text = highlight_query_terms(text, query_term)
        tfidf_results_list.append({
            'title': row['title'],
            'url': row['url'],
            'text': text
        })

    end = time.time()
    total_hit_bm25 = len(bm25_results['hits']['hits'])
    total_hit_tfidf = len(tfidf_results)
    elapse = end - start

    return jsonify({
        'query': query_term,
        'total_hit_bm25': total_hit_bm25,
        'total_hit_tfidf': total_hit_tfidf,
        'elapse': elapse,
        'bm25_results': bm25_results_list,
        'tfidf_results': tfidf_results_list
    })

if __name__ == '__main__':
    app.run(debug=True)