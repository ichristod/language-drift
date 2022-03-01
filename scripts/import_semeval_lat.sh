#!/bin/bash

# Download Latin SemEval data
wget https://zenodo.org/record/3992738/files/semeval2020_ulscd_lat.zip
unzip semeval2020_ulscd_lat.zip
rm semeval2020_ulscd_lat.zip

# Move Corpora
mkdir -p data/lat_semeval/corpus1
mkdir -p data/lat_semeval/corpus2

mv semeval2020_ulscd_lat/corpus1/lemma/*txt.gz data/lat_semeval/corpus1/lemma.txt.gz
mv semeval2020_ulscd_lat/corpus2/lemma/*txt.gz data/lat_semeval/corpus2/lemma.txt.gz

mv semeval2020_ulscd_lat/corpus1/token/*txt.gz data/lat_semeval/corpus1/token.txt.gz
mv semeval2020_ulscd_lat/corpus2/token/*txt.gz data/lat_semeval/corpus2/token.txt.gz

# Move targets
mkdir -p data/lat_semeval/targets
mv semeval2020_ulscd_lat/targets.txt data/lat_semeval/targets/targets.tsv

# Move gold data
mkdir -p data/lat_semeval/truth
mv semeval2020_ulscd_lat/truth/binary.txt data/lat_semeval/truth/binary.tsv
mv semeval2020_ulscd_lat/truth/graded.txt data/lat_semeval/truth/graded.tsv

# Clean up
rm -r semeval2020_ulscd_lat

