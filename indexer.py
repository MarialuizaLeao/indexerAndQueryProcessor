import argparse
from collections import defaultdict, Counter
import time
import os
import json
import re
import psutil
import threading
import queue
import datetime
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk import download

# Configurações de pré-processamento
download('stopwords')
download('punkt')
download('punkt_tab')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

BATCH_SIZE = 100000

def tokenize_with_hyphen(text):
    tokens = re.findall(r'\b\w+(?:-\w+)*\b', text.lower())
    expanded = []
    for token in tokens:
        expanded.append(token)
        if '-' in token:
            expanded.extend(token.split('-'))
    return expanded

# Função de pré-processamento (tokenização, remoção de stopwords e stemming)
def preprocess(text):
    tokens = word_tokenize(text.lower())
    filtered = [stemmer.stem(w) for w in tokens if w.isalnum() and w not in stop_words]
    for token in tokens:
        if '-' in token:
            filtered.append(stemmer.stem(token))
            expended = tokenize_with_hyphen(token)
            for word in expended:
                if word.isalnum() and word not in stop_words:
                    filtered.append(stemmer.stem(word))
        elif token.isalnum() and token not in stop_words:
            filtered.append(stemmer.stem(token))
    return filtered

# Função para monitorar uso de memória
def get_memory_usage_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

class Indexer:
    def __init__(self, memory_limit: int, corpus_path: str, index_path: str, threads: int):
        self.threads = threads
        self.memory_limit = memory_limit
        self.corpus_path = corpus_path
        self.index_path = index_path
        # self.inverted_index = defaultdict(list)  # termo: [(doc_id, freq)]
        # self.doc_index = {}  # doc_id: metadata
        # self.lexicon = set()

        self.doc_queue = queue.Queue(maxsize=1000)
        self.partial_indexes = []
        self.partial_index_count = 0
        self.lock = threading.Lock()
        self.memory_alert_event = threading.Event()
        self.barrier = threading.Barrier(threads)

    def run(self):
        start_time = time.time()
        os.makedirs(self.index_path, exist_ok=True)

        start_time = time.time()

        threads = []
        for _ in range(self.threads):
            t = threading.Thread(target=self.worker)
            t.start()
            threads.append(t)

        with open(self.corpus_path, 'r', encoding='utf-8') as f:
            current_batch = []
            for line in f:
                doc = json.loads(line)
                doc_id = doc.get('id')
                content = f"{doc.get('title', '')} {doc.get('text', '')} {' '.join(doc.get('keywords', []))}"
                current_batch.append((doc_id, content))

                if len(current_batch) >= BATCH_SIZE:
                    for item in current_batch:
                        self.doc_queue.put(item)

                    for _ in range(self.threads):  # -2 para não contar a thread principal e a de log
                        self.doc_queue.put('BATCH_DONE')

                    self.doc_queue.join()
                    print(f"[Log {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Batch processed: {len(current_batch)} documents.")
                    current_batch = []

            # Processa batch final se não estiver vazio
            if current_batch:
                for item in current_batch:
                    self.doc_queue.put(item)

                for _ in range(self.threads):  # -2 para não contar a thread principal e a de log
                    self.doc_queue.put('BATCH_DONE')

                self.doc_queue.join()

        # Sinaliza finalização
        for _ in range(self.threads):
            self.doc_queue.put(None)

        print("Finalizando threads...")

        for t in threads:
            print(f"Thread {t.name} finalizando...")
            t.join()

        # Faz merge dos parciais
        inverted_index, doc_index = self.merge_partial_indexes()

        # Salva índices finais
        with open(os.path.join(self.index_path, 'inverted_index.json'), 'w', encoding='utf-8') as f:
            json.dump(inverted_index, f)

        with open(os.path.join(self.index_path, 'doc_index.json'), 'w', encoding='utf-8') as f:
            json.dump(doc_index, f)

        with open(os.path.join(self.index_path, 'lexicon.json'), 'w', encoding='utf-8') as f:
            json.dump(list(inverted_index.keys()), f)

        elapsed_time = int(time.time() - start_time)

        # Estatísticas
        num_lists = len(inverted_index)
        total_postings = sum(len(v) for v in inverted_index.values())
        avg_list_size = total_postings / num_lists if num_lists > 0 else 0

        stats = {
            "Index Size": int(get_memory_usage_mb()),
            "Elapsed Time": elapsed_time,
            "Number of Lists": num_lists,
            "Average List Size": avg_list_size
        }

        print(json.dumps(stats, indent=4))

    def worker(self):
        inverted_index = defaultdict(list)
        doc_index = {}  # doc_id: metadata
        lexicon = set()

        while True:
            item = self.doc_queue.get()
            if item is None:
                print(f"[Thread] Exiting thread {threading.current_thread().name}")
                break

            if item == 'BATCH_DONE':
                # 🔥 Batch terminou: faz flush, limpa memória, espera outras threads
                if inverted_index:
                    self.flush_partial_index(inverted_index, doc_index, lexicon)

                inverted_index = defaultdict(list)
                doc_index = {}  # doc_id: metadata
                lexicon = set()

                self.doc_queue.task_done()
                continue

            doc_id, content = item
            terms = preprocess(content)
            
            term_freq = Counter(terms)

            for term, freq in term_freq.items():
                inverted_index[term].append((doc_id, freq))
                lexicon.add(term)

            doc_index[doc_id] = {
                'content': content,
            }

            self.doc_queue.task_done()

            if get_memory_usage_mb() > self.memory_limit * 0.8:
                self.memory_alert_event.set()
            
            if self.memory_alert_event.is_set():
                self.flush_partial_index(inverted_index, doc_index, lexicon)
                inverted_index = defaultdict(list)
                doc_index = {}  # doc_id: metadata
                lexicon = set()
                print(f"Memory after flushing: {get_memory_usage_mb()} MB.")

                self.barrier.wait()

                if self.barrier.parties - self.barrier.n_waiting == 1:
                    self.memory_alert_event.clear()

        # Ao final da thread, salva o que ficou
        if inverted_index:
            self.flush_partial_index(inverted_index, doc_index, lexicon)

        self.doc_queue.task_done()

    def index_document(self, doc):
        doc_id = doc.get('id')
        text = f"{doc.get('title', '')} {doc.get('text', '')} {' '.join(doc.get('keywords', []))}"
        terms = preprocess(text)
        
        term_freq = Counter(terms)

        for term, freq in term_freq.items():
            self.inverted_index[term].append((doc_id, freq))
            self.lexicon.add(term)

        self.doc_index[doc_id] = {
                'title': doc.get('title'),
                'keywords': doc.get('keywords', [])
            }

    def flush_partial_index(self, inverted_index, doc_index, lexicon):
        with self.lock:
            filename = os.path.join(self.index_path, f'partial_index_{self.partial_index_count}.json')
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump({
                    'inverted_index': {k: v for k, v in inverted_index.items()},
                    'doc_index': doc_index,
                    'lexicon': list(lexicon)
                }, f)
            self.partial_index_count += 1

    def merge_partial_indexes(self):
        final_inverted_index = defaultdict(list)
        final_doc_index = {}

        for i in range(self.partial_index_count):
            filename = os.path.join(self.index_path, f'partial_index_{i}.json')
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
                inv = data['inverted_index']
                doc = data['doc_index']

                for term, postings in inv.items():
                    final_inverted_index[term].extend(postings)

                final_doc_index.update(doc)

            os.remove(filename)  # Remove parcial após merge

        return final_inverted_index, final_doc_index

def main():
    # Configuracao do analisador de argumentos
    parser = argparse.ArgumentParser(description="Web Crawler")
    parser.add_argument(
        "-m", dest='memory_limit', type=int, required=True, help="The memory available to the indexer in megabytes."
    )
    parser.add_argument(
        "-c", dest='corpus_path', type=str, required=True, help="The path to the corpus file to be indexed."
    )
    parser.add_argument(
        "-i", dest='index_path', type=str, required=False, help="The path to the directory where indexes should be written."
    )
    parser.add_argument(
        "-t", dest='threads', type=int, required=False, default=1, help="The number of threads to use for indexing."
    )

    args = parser.parse_args()
    # Initiate thread log
    # log_thread = threading.Thread(target=log_thread_status, daemon=True)
    # log_thread.start()
    indexer = Indexer(args.memory_limit, args.corpus_path, args.index_path, args.threads)
    indexer.run()

if __name__ == "__main__":
    main()