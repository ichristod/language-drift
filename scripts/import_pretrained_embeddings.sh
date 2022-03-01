#!/bin/bash
name=$0
language=$1

if [ "$language" == "lat" ]
  then
    # Download Latin conll 17 pretrained embeddings
    wget http://vectors.nlpl.eu/repository/20/57.zip
    mkdir -p pretrained_embed/lat_conll17
    unzip 57.zip -d pretrained_embed/lat_conll17
    rm 57.zip
elif [ "$language" == "de" ]
  then
    # Download German conll 17 pretrained embeddings
    wget http://vectors.nlpl.eu/repository/20/45.zip
    mkdir -p pretrained_embed/de_conll17
    unzip 45.zip -d pretrained_embed/de_conll17
    rm 45.zip
elif [ "$language" == "spa" ]
  then
    # Download Spanish conll 17 pretrained embeddings
    wget http://vectors.nlpl.eu/repository/20/68.zip
    mkdir -p pretrained_embed/spa_conll17
    unzip 68.zip -d pretrained_embed/spa_conll17
    rm 68.zip
elif [ "$language" == "swe" ]
  then
    # Download Swedish conll 17 pretrained embeddings
    wget http://vectors.nlpl.eu/repository/20/69.zip
    mkdir -p pretrained_embed/swe_conll17
    unzip 69.zip -d pretrained_embed/swe_conll17
    rm 69.zip
fi