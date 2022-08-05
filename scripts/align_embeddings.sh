#!/bin/bash

name=$0
mapping=$1
outdir=$2
resdir=$3
data_set_id=$4
traindir=$5
top_neighbors=$6
w2vec_method=$7
language=$8

if [[ $mapping == "procrustes" ]]
  then
    if [[ $w2vec_method == "lda2vec" ]]
      then
        # Create word2vec format files from wordvectors of pytorch lda2vec
        python modules/numpy_to_word2vec_format.py data/${data_set_id}/corpus1/lemma_docids.json ${traindir}
        python modules/numpy_to_word2vec_format.py data/${data_set_id}/corpus2/lemma_docids.json ${traindir}
    fi
    echo "    Align with orthogonal procrustes method"
    # Length-normalize, meanc-center and align with OP
    python modules/map_embeddings.py --normalize unit center --init_identical --orthogonal ${traindir}/mat1 ${traindir}/mat2 ${outdir}/mat1ca ${outdir}/mat2ca
    # Measure CD for every word in the intersection
    python measures/cd.py ${outdir}/mat1ca ${outdir}/mat2ca ${resdir}/distances_intersection.tsv
    # Measure CD for every target word
    python measures/cd.py ${outdir}/mat1ca ${outdir}/mat2ca data/${data_set_id}/targets/targets.tsv ${resdir}/distances_targets.tsv
    # Calculate local neighborhood measure for target words
    python measures/local_neighborhood.py ${traindir}/mat1 ${traindir}/mat2 data/${data_set_id}/targets/targets.tsv ${top_neighbors} ${resdir}/local_neighborhood_distances.tsv ${language}

elif [[ "$mapping" == "incremental"  || "$mapping" == "twec" ]]
  then
    echo "    Alignment is not necessary in cases of "${mapping}"."
    # Length-normalize, meanc-center
    python modules/normalization.py ${traindir}/mat1 ${traindir}/mat2
    # Measure CD for every word in the intersection
    python measures/cd.py ${traindir}/mat1 ${traindir}/mat2 ${resdir}/distances_intersection.tsv
    # Measure CD for every target word
    python measures/cd.py ${traindir}/mat1 ${traindir}/mat2 data/${data_set_id}/targets/targets.tsv ${resdir}/distances_targets.tsv
    # Calculate jaccard index for target words' topN neighbors
    # python measures/jaccard_neighbors.py ${outdir}/mat1 ${outdir}/mat2 data/${data_set_id}/targets/targets.tsv ${window_size} ${resdir}/local_neighborhood_distances.csv
    # Calculate local neighborhood measure for target words
    python measures/local_neighborhood.py ${traindir}/mat1 ${traindir}/mat2 data/${data_set_id}/targets/targets.tsv ${top_neighbors} ${resdir}/local_neighborhood_distances.tsv
fi