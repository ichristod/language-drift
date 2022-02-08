#!/bin/bash

# Download German DTA Corpus data
wget https://www.ims.uni-stuttgart.de/documents/ressourcen/korpora/wocc/dta18.txt.gz
wget https://www.ims.uni-stuttgart.de/documents/ressourcen/korpora/wocc/dta19.txt.gz

# Move Corpora
mkdir -p data/dta_corpus/corpus1
mkdir -p data/dta_corpus/corpus2

mv dta18.txt.gz data/dta_corpus/corpus1/dta18.txt.gz
mv dta19.txt.gz data/dta_corpus/corpus2/dta19.txt.gz