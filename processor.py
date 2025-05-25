import argparse
import json
from parser import Parser

class Processor:
    def __init__(self, ranker, index_path, threads):
        self.ranker = ranker
        self.index_path = index_path
        self.threads = threads
        self.index = {}
        self.parser = Parser()

        self.load_index()

    def process_query(self, query: str):
        # Preprocess the query
        tokens = self.parser.parse_text(query)
        # Implement the ranking logic based on the selected ranker
        if self.ranker == "TFIDF":
            return self.TFIDF(tokens)
        elif self.ranker == "BM25":
            return self.BM25(tokens)
        else:
            raise ValueError("Invalid ranker specified. Use 'TFIDF' or 'BM25'.")

    def load_index(self):
        # Load the index from the specified path
        with open(self.index_path, 'r') as f:
            self.index = json.load(f)
        

def main():
    parser = argparse.ArgumentParser(description="Web Crawler")
    parser.add_argument(
        "-r", dest='ranker', type=str, required=True, help="A string informing the ranking function (either “TFIDF” or “BM25”) to be used to score documents for each query."
    )
    parser.add_argument(
        "-q", dest='queries', type=str, required=True, help="he path to a file with the list of queries to process."
    )
    parser.add_argument(
        "-i", dest='index_path', type=str, required=False, help="The path to an index file."
    )
    parser.add_argument(
        "-t", dest='threads', type=int, required=False, default=1, help="The number of threads to use for indexing."
    )

    args = parser.parse_args()
    

if __name__ == "__main__":
    main()