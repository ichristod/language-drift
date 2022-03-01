#!/bin/bash

# Download Swedish SemEval data
wget https://zenodo.org/record/3730550/files/semeval2020_ulscd_swe.zip
unzip semeval2020_ulscd_swe.zip
rm semeval2020_ulscd_swe.zip

# Move Corpora
mkdir -p data/swe_semeval/corpus1
mkdir -p data/swe_semeval/corpus2

mv semeval2020_ulscd_swe/corpus1/lemma/*txt.gz data/swe_semeval/corpus1/lemma.txt.gz
mv semeval2020_ulscd_swe/corpus2/lemma/*txt.gz data/swe_semeval/corpus2/lemma.txt.gz

mv semeval2020_ulscd_swe/corpus1/token/*txt.gz data/swe_semeval/corpus1/token.txt.gz
mv semeval2020_ulscd_swe/corpus2/token/*txt.gz data/swe_semeval/corpus2/token.txt.gz

# Move targets
mkdir -p data/swe_semeval/targets
mv semeval2020_ulscd_swe/targets.txt data/swe_semeval/targets/targets.tsv

# Move gold data
mkdir -p data/swe_semeval/truth
mv semeval2020_ulscd_swe/truth/binary.txt data/swe_semeval/truth/binary.tsv
mv semeval2020_ulscd_swe/truth/graded.txt data/swe_semeval/truth/graded.tsv

# Clean up
rm -r semeval2020_ulscd_swe

