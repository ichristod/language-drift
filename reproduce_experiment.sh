#!/bin/bash

# load all execution parameters
. executions-configuration/exec.conf

# TODO
# Create a kind of dictionary type in configuration file
identify_appropriate_pretrained () {
  if [ "$language" == "en" ]
    then
      pretrained_embed=glove
      pretrained_path=./pretrained_embed/glove.6B
  elif [ "$language" == "de" ]
    then
      pretrained_embed=dewiki
      pretrained_path=./pretrained_embed/${pretrained_embed}
  elif [ "$language" == "lat" ]
    then
      pretrained_embed=latconll17
      pretrained_path=./pretrained_embed/${pretrained_embed}
  elif [ "$language" == "swe" ]
    then
      pretrained_embed=sweconll17
      pretrained_path=./pretrained_embed/${pretrained_embed}
  fi
}

function retrieve_parameters_from_path(){
  #param_id=cbow_win10_dim100_k5_s0.001_mc3_mc3_i5_alignment_glove
  param_id=$1

  # split path with to different elements of array
  # our delimeter is '_'
  IFS='_' read -ra params <<< "$param_id"

  # set pretrained embeddings name
  if [[ ${#params[@]} == 10 ]]
    then
      pretrained_embed="${params[9]}"
  else
    pretrained_embed="None"
  fi

  for i in "${params[@]}"; do
    value=$(echo "${i}" | tr -dc '0-9.')
    name=$(echo "${i}" | tr -dc 'a-z')

    # check categorical parameters
    if [ -z "${value}" ];
      then
        case ${name} in
          "cbow"|"sgns")
            w2vec_method=${name}
            ;;

          "incremental"|"procrustes")
            mapping=${name}
            ;;

          *)
            ;;
        esac
    else
      case ${name} in
        "win")
        window_size=${value}
        ;;

        "dim")
        dim=${value}
        ;;

        *)
        ;;
      esac
    fi
  done

  echo ${window_size} ${dim} ${w2vec_method} ${mapping} ${pretrained_embed}
}

# check task
if $binary_classification; then
  for language in ${languages[*]}; do

    # download datasets if needed
    if $download_datasets ; then
      import_script="import_semeval"_${language}
      bash ./scripts/${import_script}.sh
    fi

    # download datasets if needed
    if $download_pretrained ; then
      bash ./scripts/import_pretrained_embeddings.sh ${language}
    fi

    # flag dataset with language and version
    dataset_id=${language}_${version}

    # create the appropriate folder structure if needed
    if $prepare_datasets ; then
      bash ./scripts/prepare_data.sh ${dataset_id} ./data/${language}_semeval/corpus1/lemma.txt.gz ./data/${language}_semeval/corpus2/lemma.txt.gz ./data/${language}_semeval/corpus1/token.txt.gz ./data/${language}_semeval/corpus2/token.txt.gz ./data/${language}_semeval/targets/targets.tsv ./data/${language}_semeval/truth/binary.tsv ./data/${language}_semeval/truth/graded.tsv
    fi

    # executions' output folder
    exec_dir=output/${dataset_id}/*/

    # finally execute the experiment :-)
    for action in $actions; do

      # create embeddings for every different algorithm
      if [[ $action == "train" ]]; then
        for w2vec_method in $w2vec_methods; do
          echo "w2vec_method: "${w2vec_method}

          for mapping in $mappings; do

            param_id=${w2vec_method}_win${window_size}_dim${dim}_k${k}_s${s}_mc${min_count1}_mc${min_count2}_i${epochs}_${mapping}

            if $use_pretrained; then
              # define pretrained embeddings path
              identify_appropriate_pretrained

              param_id_embed=${param_id}_${pretrained_embed}
              # parameter to indicate execution's folder name

              # output folder of models with pretrained weights initialization
              outdir=output/${dataset_id}/${param_id_embed}/trained_models
              mkdir -p ${outdir}

              # Generate word embeddings with pretrained weights initialization
              python static/sgns.py --len data/${dataset_id}/corpus1/lemma.txt.gz ${outdir}/mat1 ${window_size} ${dim} ${k} ${s} ${min_count1} ${epochs} ${mapping} ${w2vec_method} --pretrained ${pretrained_embed} --path_pretrained ${pretrained_path}
              python static/sgns.py --len data/${dataset_id}/corpus2/lemma.txt.gz ${outdir}/mat2 ${window_size} ${dim} ${k} ${s} ${min_count2} ${epochs} ${mapping} ${w2vec_method} --pretrained ${pretrained_embed} --path_pretrained ${pretrained_path}

            fi

            # output folder of models with stochastic weights initialization
            outdir=output/${dataset_id}/${param_id}/trained_models
            mkdir -p ${outdir}

            # Generate word embeddings with stochastic weights initialization
            python static/sgns.py --len data/${dataset_id}/corpus1/lemma.txt.gz ${outdir}/mat1 ${window_size} ${dim} ${k} ${s} ${min_count1} ${epochs} ${mapping} ${w2vec_method} --pretrained None --path_pretrained None
            python static/sgns.py --len data/${dataset_id}/corpus2/lemma.txt.gz ${outdir}/mat2 ${window_size} ${dim} ${k} ${s} ${min_count2} ${epochs} ${mapping} ${w2vec_method} --pretrained None --path_pretrained None
          done
        done
      fi

      # create comparable objects - bring them in common space
      if [[ $action == "align" ]]; then
        for mapping in $mappings; do

          # for each model in datasetId
          for dir in $(ls -d ${exec_dir}); do
            dir="${dir%/*}"

            # parameter to indicate execution's folder name
            # cbow_win10_dim100_k5_s0.001_mc3_mc3_i5_alignment_glove
            param_id="${dir##*/}"

            # folder of trained models
            traindir=output/${dataset_id}/${param_id}/trained_models
            # output folder of common space models
            outdir=output/${dataset_id}/${param_id}/common_space_models
            # results folder
            resdir=output/${dataset_id}/${param_id}/distances
            # create output and results folder
            mkdir -p ${outdir}
            mkdir -p ${resdir}

            # actual creation of comparable objects
            bash ./scripts/align_embeddings.sh ${mapping} ${outdir} ${resdir} ${dataset_id} ${traindir}
          done
        done
      fi

      # create final results
      if [[ $action == "evaluate" ]]; then

        # apply threshold criteria for binary classification
        for thres_percentage in $thres_percentages; do

           # for each model in datasetId
          for dir in $(ls -d ${exec_dir}); do
            dir="${dir%/*}"

            # parameter to indicate execution's folder name
            # cbow_win10_dim100_k5_s0.001_mc3_mc3_i5_alignment_glove
            param_id="${dir##*/}"

            # calculated distances folder
            distdir=output/${dataset_id}/${param_id}/distances

            # results folder
            outdir=output/${dataset_id}/${param_id}/results/t${thres_percentage}
            mkdir -p ${outdir}

            # Compute binary scores for targets
            python measures/binary.py ${distdir}/distances_intersection.tsv ${distdir}/distances_targets.tsv ${outdir}/scores_targets.tsv " ${thres_percentage} "

            retrieve_parameters_from_path ${param_id}

            # Calculate classification performance
            python evaluation/class_metrics.py data/${dataset_id}/truth/binary.tsv ${outdir}/scores_targets.tsv ${outdir}/pickled_classification_res.pkl ${mapping} ${w2vec_method} ${pretrained_embed} ${window_size} ${dim} ${thres_percentage} ${dataset_id} ${language}

          done
        done
      fi
    done
  done
fi