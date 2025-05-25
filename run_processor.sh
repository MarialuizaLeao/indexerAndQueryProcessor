#!/bin/bash

python3 -m venv pa2
source pa2/bin/activate
pip3 install -r requirements.txt

python3 processor.py -q queries.txt -i _index_/inverted_index.json -r TFIDF