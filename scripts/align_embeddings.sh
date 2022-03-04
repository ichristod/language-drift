#!/bin/bash

name=$0
mapping=$1
outdir=$2
resdir=$3
data_set_id=$4
traindir=$5

if [[ $mapping == "procrustes" ]]
  then
    echo "Non incremental learning"
    # Length-normalize, meanc-center and align with OP
    python modules/map_embeddings.py --init_identical --orthogonal ${traindir}/mat1 ${traindir}/mat2 ${outdir}/mat1ca ${outdir}/mat2ca
    # Measure CD for every word in the intersection
    python measures/cd.py ${outdir}/mat1ca ${outdir}/mat2ca ${resdir}/distances_intersection.tsv
    # Measure CD for every target word
    python measures/cd.py ${outdir}/mat1ca ${outdir}/mat2ca data/${data_set_id}/targets/targets.tsv ${resdir}/distances_targets.tsv
    #Calculate jaccard index for target words' topN neighbors
    #python measures/jaccard_neighbors.py ${outdir}/mat1ca ${outdir}/mat2ca data/${data_set_id}/targets/targets.tsv ${window_size} ${resdir}/jaccard_index_report.csv

elif [[ "$mapping" == "incremental" ]]
  then
    echo "Alignment is not necessary in cases of incremental_learning"
    # Measure CD for every word in the intersection
    python measures/cd.py ${traindir}/mat1 ${traindir}/mat2 ${resdir}/distances_intersection.tsv
    # Measure CD for every word in the intersection of the vocabularies
    python measures/cd.py ${traindir}/mat1 ${traindir}/mat2 data/${data_set_id}/targets/targets.tsv ${resdir}/distances_targets.tsv
    # Calculate jaccard index for target words' topN neighbors
    # python measures/jaccard_neighbors.py ${outdir}/mat1 ${outdir}/mat2 data/${data_set_id}/targets/targets.tsv ${window_size} ${resdir}/jaccard_index_report.csv
fi