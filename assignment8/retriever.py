import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class DataRetriever:
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.text_chunks = self._preprocess()
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.vectorizer.fit_transform(self.text_chunks)

    def _preprocess(self):
        chunks = []
        for _, row in self.df.iterrows():
            chunk = " ".join([f"{col}: {row[col]}" for col in self.df.columns])
            chunks.append(chunk)
        return chunks

    def get_relevant_chunks(self, query, top_k=3):
        query_vec = self.vectorizer.transform([query])
        sim_scores = cosine_similarity(query_vec, self.tfidf_matrix)[0]
        top_indices = sim_scores.argsort()[-top_k:][::-1]
        return [self.text_chunks[i] for i in top_indices]
