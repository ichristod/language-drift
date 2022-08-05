#!/bin/bash
# load all execution parameters
. executions-configuration/exec_lda2vec.conf

if $prepare_lda2vec; then
    echo "aaaaaaaaaaaaa"
    python static/prepare_lda2vec_data.py data/${dataset_id}/corpus_1/lemma.txt.gz data/${dataset_id}/corpus_2/lemma.txt.gz
fi

