# create new env
conda create --name language-drift python=3.9.1

# install pip
conda install pip

# install requirements
pip install -r requirements.txt


# give permissions in project folders
sudo chmod a+w language-drift

# download semeval test data
sudo ./scripts/import_semeval_en.sh

# create the appropriate folder structure
bash ./scripts/prepare_data.sh 1st-try ./data/en_semeval/corpus1/lemma.txt.gz ./data/en_semeval/corpus2/lemma.txt.gz ./data/en_semeval/corpus1/token.txt.gz ./data/en_semeval/corpus2/token.txt.gz ./data/en_semeval/targets/targets.tsv ./data/en_semeval/truth/binary.tsv ./data/en_semeval/truth/graded.tsv

# download spacy to use for detection in whether a word is a NOUN, VERB or ADJ.
python -m spacy download en_core_web_sm

# run SGNS without pretrained embeddings
bash ./scripts/discover_sgns.sh 1st-try 10 50 5 0.001 3 3 5 1.0 en None None

# run SGNS with Glove pretrained embeddings
bash ./scripts/discover_sgns.sh 1st-try 10 50 5 0.001 3 3 5 1.0 en Glove ./pretrained_embed/glove.6B


./data/en_semeval/targets/targets.tsv
