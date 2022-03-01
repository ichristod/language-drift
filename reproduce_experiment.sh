#!/bin/bash

#Declare languages of experiment
languages=("en de lat swe")
#en de en lat
download_datasets=true
download_pretrained=true
prepare_datasets=true

thresholds="1.0"
mappings="alignment incremental"
w2vec_methods="cbow sgns"
version="1.0.0"

# hardcoded paths
identify_appropriate_pretrained () {
  if [ "$language" == "en" ]
    then
      pretrained_embed=glove
      pretrained_path=./pretrained_embed/glove.6B
  elif [ "$language" == "de" ]
    then
      pretrained_embed=dewiki
      pretrained_path=./pretrained_embed/dewiki
  elif [ "$language" == "lat" ]
    then
      pretrained_embed=lat_conll17
      pretrained_path=./pretrained_embed/lat_conll17
  elif [ "$language" == "swe" ]
    then
      pretrained_embed=swe_conll17
      pretrained_path=./pretrained_embed/swe_conll17
  fi
}

for language in ${languages[*]}; do
  # download datasets if needed
  if $download_datasets ; then
    import_script="import_semeval"_${language}
    bash ./scripts/${import_script}.sh
  fi

  if $download_pretrained ; then
    bash ./scripts/import_pretrained_embeddings.sh ${language}
  fi

  # flag dataset with language and version
  dataset_id=${language}_${version}

  # create the appropriate folder structure if needed
  if $prepare_datasets ; then
    bash ./scripts/prepare_data.sh ${dataset_id} ./data/${language}_semeval/corpus1/lemma.txt.gz ./data/${language}_semeval/corpus2/lemma.txt.gz ./data/${language}_semeval/corpus1/token.txt.gz ./data/${language}_semeval/corpus2/token.txt.gz ./data/${language}_semeval/targets/targets.tsv ./data/${language}_semeval/truth/binary.tsv ./data/${language}_semeval/truth/graded.tsv
  fi

  identify_appropriate_pretrained

  # finally execute the experiment :-)
  # 1.0 1.5 2.0
  for threshold in $thresholds; do
    # alignment incremental
    for mapping in $mappings; do
      # sgns cbow
      for w2vec_method in $w2vec_methods; do
        #
        if [ "$language" != "swe" ]
          then
            bash ./scripts/classify_sgns.sh ${dataset_id} 10 100 5 0.001 3 3 5 ${threshold} ${mapping} ${w2vec_method} ${language} ${pretrained_embed} ${pretrained_path}
        fi
        bash ./scripts/classify_sgns.sh ${dataset_id} 10 100 5 0.001 3 3 5 ${threshold} ${mapping} ${w2vec_method} ${language} None None
      done
    done
  done
  # delete previously created models due to limited storage
  # rm -rf ./output/${dataset_id}/*
done
