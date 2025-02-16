from pathlib import Path
import os
import pickle
import json
import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from m4 import preProcess
from m4 import preprocess_stopwords

class Pr:
    def __init__(self, alpha):
        self.pr_result = None
        self.crawled_folder = Path(os.path.abspath('')).parent / 'crawled/'
        self.alpha = alpha

    def url_extractor(self):
        url_maps = {}
        all_urls = set()

        for file in os.listdir(self.crawled_folder):
            if file.endswith(".txt"):
                with open(os.path.join(self.crawled_folder, file), encoding='utf-8') as f:
                    j = json.load(f)
                all_urls.add(j['url'])
                for s in j['url_lists']:
                    all_urls.add(s)
                url_maps[j['url']] = list(set(j['url_lists']))
        all_urls = list(all_urls)
        return url_maps, all_urls

    def pr_calc(self):
        url_maps, all_urls = self.url_extractor()
        url_matrix = pd.DataFrame(columns=all_urls, index=all_urls)

        for url in url_maps:
            if len(url_maps[url]) > 0 and len(all_urls) > 0:
                url_matrix.loc[url] = (1 - self.alpha) * (1 / len(all_urls))
                url_matrix.loc[url, url_maps[url]] = url_matrix.loc[url, url_maps[url]] + (
                            self.alpha * (1 / len(url_maps[url])))

        url_matrix.loc[url_matrix.isnull().all(axis=1), :] = (1 / len(all_urls))

        x0 = np.matrix([1 / len(all_urls)] * len(all_urls))
        P = np.asmatrix(url_matrix.values)

        prev_Px = x0
        Px = x0 * P
        i = 0
        while any(abs(np.asarray(prev_Px).flatten() - np.asarray(Px).flatten()) > 1e-8):
            i += 1
            prev_Px = Px
            Px = Px * P

        print('Converged in {0} iterations: {1}'.format(i, np.around(np.asarray(Px).flatten().astype(float), 5)))

        self.pr_result = pd.DataFrame(Px, columns=url_matrix.index, index=['score']).T.loc[list(url_maps.keys())]


class IndexerTFIDF:
    def __init__(self, is_reset=False):
        self.documents = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.crawled_folder = Path(os.path.abspath('')).parent / 'crawled/'
        self.stored_file = 'resources/manual_indexer.pkl'
        self.pr = Pr(alpha=0.85)

        if not is_reset and os.path.isfile(self.stored_file):
            with open(self.stored_file, 'rb') as f:
                cached_dict = pickle.load(f)
                self.__dict__.update(cached_dict)
        else:
            self.run_indexer()

    def run_indexer(self):
        documents = []

        # Load each document into memory
        for file in os.listdir(self.crawled_folder):
            if file.endswith(".txt"):
                with open(os.path.join(self.crawled_folder, file), 'r', encoding='utf-8') as f:
                    j = json.load(f)
                    documents.append(j)

        self.documents = pd.DataFrame.from_dict(documents)
        self.pr.pr_calc()
        self.documents['pagerank'] = self.documents['url'].map(self.pr.pr_result['score'])

        # Create and fit the TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(preprocessor=preProcess, stop_words=preprocess_stopwords(stopwords.words('english')))
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(
            self.documents.apply(lambda s: ' '.join(s[['title', 'text']]), axis=1))

        with open(self.stored_file, 'wb') as f:
            pickle.dump(self.__dict__, f)

    import re

    def query(self, q):
        # Transform the query using the fitted TF-IDF vectorizer
        transformed_query = self.tfidf_vectorizer.transform([q])

        # Compute cosine similarity between query and document vectors
        tfidf_scores = cosine_similarity(transformed_query, self.tfidf_matrix).flatten()

        # Rank documents based on tfidf * pagerank
        final_scores = tfidf_scores * self.documents['pagerank'].values
        rank = final_scores.argsort()[::-1]

        # Retrieve and return ranked results
        results = self.documents.iloc[rank].copy().reset_index(drop=True)
        results['tfidf_score'] = tfidf_scores[rank]
        results['final_score'] = final_scores[rank]

        # Filter out documents with a final score of 0.0
        results = results[results['final_score'] > 0.0].reset_index(drop=True)

        def highlight_and_trim(text, query, max_length=200):
            query_words = query.split()  # Handle multi-word queries
            text_lower = text.lower()

            # Find first occurrence of any whole-word query match
            query_positions = [
                match.start() for word in query_words
                for match in re.finditer(rf'\b{re.escape(word)}\b', text_lower)  # Exact whole word match
            ]

            start_idx = min(query_positions) if query_positions else 0  # Default to 0 if no match found

            # Define snippet range: Try centering the query
            snippet_start = max(0, start_idx - max_length // 4)  # Shift start back a little for context
            snippet_end = min(len(text), snippet_start + max_length)

            # Extract the snippet
            preview_text = text[snippet_start:snippet_end]

            # Ensure it starts and ends at word boundaries
            if snippet_start > 0 and ' ' in preview_text:
                preview_text = '...' + preview_text[preview_text.index(' '):]  # Avoid cutting first word
            if snippet_end < len(text) and ' ' in preview_text:
                preview_text = preview_text[:preview_text.rindex(' ')] + '...'  # Avoid cutting last word

            # Highlight query terms with whole word matching
            for word in query_words:
                preview_text = re.sub(rf'\b{re.escape(word)}\b', r'<b>\g<0></b>', preview_text, flags=re.IGNORECASE)

            return preview_text

        # Apply transformation
        results['text'] = results['text'].apply(lambda x: highlight_and_trim(x, q))

        # Remove duplicate rows based on 'text' column
        results = results.drop_duplicates(subset=['text']).reset_index(drop=True)
        results = results.drop_duplicates(subset=['title']).reset_index(drop=True)
        results = results.drop_duplicates(subset=['url']).reset_index(drop=True)

        return results