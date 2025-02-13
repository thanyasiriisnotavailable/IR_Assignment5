from flask import Flask, request, render_template
from elasticsearch import Elasticsearch
import pandas as pd
import time
from IndexerTFIDF import IndexerTFIDF
from IndexerTFIDF import Pr

app = Flask(__name__, template_folder='C:/Users/acer/All SE/IR/Search_Engine')
app.es_client = Elasticsearch(
    "https://localhost:9200",
    basic_auth=("elastic", "M8OIu*UpfhnRm10hmp*X"),
    ca_certs="C:/Users/acer/All SE/IR/M2/http_ca.crt"
)

indexer = IndexerTFIDF(is_reset=False)

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
    bm25_results_df = pd.DataFrame(
        [
            [hit["_source"]['title'], hit["_source"]['url'], hit["_source"]['text'][:100], hit["_score"]]
            for hit in bm25_results['hits']['hits']
        ],
        columns=['title', 'url', 'text', 'score']
    )

    # Custom TF-IDF + PageRank search
    tfidf_results = indexer.query(query_term)
    tfidf_results['text'] = tfidf_results['text'].apply(lambda x: ' '.join(x.split()[:20]) + ('...' if len(x.split()) > 20 else ''))

    end = time.time()
    total_hit = len(bm25_results['hits']['hits']) + len(tfidf_results)
    elapse = end - start

    return render_template('search.html', 
                           query=query_term, 
                           total_hit=total_hit, 
                           elapse=elapse, 
                           bm25_results=bm25_results_df.to_dict('records'), 
                           tfidf_results=tfidf_results[['title', 'url', 'text', 'final_score']].to_dict('records'))

if __name__ == '__main__':
    app.run(debug=True)