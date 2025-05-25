import math
from collections import defaultdict


class TFIDF:
    def __init__(self, inverted_index, total_docs):
        self.inverted_index = inverted_index
        self.total_docs = total_docs
        self.idf_cache = {}

    def idf(self, term):
        if term in self.idf_cache:
            return self.idf_cache[term]

        df = len(self.inverted_index.get(term, []))
        if df == 0:
            idf = 0
        else:
            idf = math.log(self.total_docs / df)

        self.idf_cache[term] = idf
        return idf

    def tf(self, freq):
        return 1 + math.log(freq) if freq > 0 else 0

    def score(self, query_terms):
        # Coleta todos os documentos que contêm todos os termos (conjunctive DAAT)
        doc_postings = []
        for term in query_terms:
            postings = self.inverted_index.get(term, [])
            doc_postings.append(set(doc_id for doc_id, _ in postings))

        if not doc_postings:
            return {}

        # Interseção dos postings (documentos que têm TODOS os termos)
        candidate_docs = set.intersection(*doc_postings) if len(doc_postings) > 1 else doc_postings[0]

        scores = defaultdict(float)

        for term in query_terms:
            postings = self.inverted_index.get(term, [])
            posting_dict = {doc_id: freq for doc_id, freq in postings}

            idf_value = self.idf(term)

            for doc_id in candidate_docs:
                tf_value = self.tf(posting_dict.get(doc_id, 0))
                scores[doc_id] += tf_value * idf_value

        return scores
