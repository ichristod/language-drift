#!/bin/bash
name=$0
data_set_id=$1
window_size=$2
dim=$3
k=$4
s=$5
min_count1=$6
min_count2=$7
itera=$8
t=$9
incremental=${10}
w2vec_algorithm=${11}
pretrained=${12}
path_pretrained=${13}

function usage {
    echo "For a set of target words, decide which words lost or gained sense(s) between C1 and C2."
    echo ""
    echo "  Usage:" 
    echo "      classify_sgns.sh <data_set_id> <window_size> <dim> <k> <s> <min_count1> <min_count2> <itera> <t>" 
    echo ""
    echo "      <data_set_id>           = data set identifier"
    echo "      <window_size>           = the linear distance of context words to consider in each direction"
    echo "      <dim>                   = dimensionality of embeddings"
    echo "      <k>                     = number of negative samples parameter (equivalent to shifting parameter for PPMI)"
    echo "      <s>                     = threshold for subsampling"
    echo "      <min_count1>            = number of occurrences for a word to be included in the vocabulary (corpus1)"
    echo "      <min_count2>            = number of occurrences for a word to be included in the vocabulary (corpus2)"
    echo "      <itera>                 = number of iterations"
    echo "      <t>                     = threshold = mean + t * standard error"
    echo "      <incremental>           = indicates incremental learning (incremental)"
    echo "      <w2vec_algorithm>       = cbow | sgns"
    echo "      <pretrained>            = option of pretrained embeddings (None, glove)"
    echo "      <path_pretrained>       = path to pretrained embeddings directory with txt files"
    echo ""
}

if [ $# -ne 11 ] && [ $# -ne 13 ]
	then
		usage
		exit 1
fi

if [[ ( $1 == "--help") ||  $1 == "-h" ]] 
	then 
		usage
		exit 0
fi

param_id=${w2vec_algorithm}_win${window_size}_dim${dim}_k${k}_s${s}_mc${min_count1}_mc${min_count2}_i${itera}_${incremental}

if [[ $incremental == true ]]
  then
    param_id+=_"incremental"
fi


if [ $# -eq 13 ]
  then
    param_id+=_${pretrained}
fi

outdir=output/${data_set_id}/${param_id}

mkdir -p ${outdir}
mkdir -p ${resdir}


# Generate word embeddins with SGNS
python static/sgns.py --len data/${data_set_id}/corpus1/lemma.txt.gz ${outdir}/mat1 ${window_size} ${dim} ${k} ${s} ${min_count1} ${itera} ${mapping} ${w2vec_algorithm} --pretrained ${pretrained} --path_pretrained ${path_pretrained}
python static/sgns.py --len data/${data_set_id}/corpus2/lemma.txt.gz ${outdir}/mat2 ${window_size} ${dim} ${k} ${s} ${min_count2} ${itera} ${mapping} ${w2vec_algorithm} --pretrained ${pretrained} --path_pretrained ${path_pretrained}

echo ${data_set_id} ${resdir} ${mapping} ${w2vec_algorithm} ${pretrained} ${window_size} ${dim} ${t} ${data_set_id} ${language}




