#!/bin/bash
function pause(){
 read -s -n 1 -p "Press any key to continue . . ."
 echo ""
}
STR="cbow_win10_dim100_k5_s0.001_mc3_mc3_i5_incremental cbow_win10_dim100_k5_s0.001_mc3_mc3_i5_twec"
SUB='twec'

for tade in $STR; do
  if [[ "$tade" =~ .*"$SUB".* ]]; then
    echo ${tade}
  fi
done


pause




'''


# concatenate lemma version of coprus into one file (twec)
mkdir -p ../data/de_4.0.0/corpus_concat
cat ../data/de_4.0.0/corpus1/token.txt.gz ../data/de_4.0.0/corpus2/token.txt.gz > ../data/de_4.0.0/corpus_concat/token.txt.gz

files="../data/de_4.0.0/corpus1/token.txt.gz ../data/de_4.0.0/corpus2/token.txt.gz ../data/de_4.0.0/corpus_concat/token.txt.gz"

for file in $files; do
  FILENAME=${file}
  FILESIZE=$(stat -c%s "$FILENAME")
  echo "Size of $FILENAME = $FILESIZE bytes."
done







python evaluation/class_metrics.py data/${dataset_id}/truth/binary.tsv ${outdir}/scores_targets.tsv \n
${outdir}/pickled_classification_res.pkl ${mapping} ${w2vec_method} ${pretrained_embed} ${window_size} \n
${dim} ${thres_percentage} ${dataset_id} ${language}

param_id="cbow_win10_dim50_k5_s0.001_mc3_mc3_i5_incremental"

params=$(echo $param_id | tr "_" "\n")

for param in $params
do
    echo "$param"
    if
    NUMBER=$(echo "I am 999 years old." | tr -dc '0-9')
done

read pass1 pass2 pass3 <<< $(bash ./child.sh)
echo ${pass1} ${pass2}
'''