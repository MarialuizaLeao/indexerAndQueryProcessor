#!/bin/bash

python3 -m venv pa2
source pa2/bin/activate
pip3 install -r requirements.txt

python3 indexer.py -m 1024 -c reduced_corpus.jsonl -i _index_ -t 100