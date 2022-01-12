# language-drift


[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

> Implementation of my thesis as part of the IIT-UOP M.Sc. in Data Science (2020-2022). 

This project aims to compare different distributional text representations to determine whether semantic drift occurs and how it impacts the different text representations.

[]: # 'TODO' Give credits to https://github.com/seinan9/LSCDiscovery and other projects that it was based on.

## Table of Contents

- [Install](#install)
- [API](#api)
- [Tests](#tests)
- [Support](#support)
- [Contributing](#contributing)
- [License](#license)

## Install

```
# create new env
conda create --name language-drift python=3.9.1

# install pip
conda install pip

# install requirements
pip install -r requirements.txt

# download spacy to use for detection in whether a word is a NOUN, VERB or ADJ.
python -m spacy download en_core_web_sm

# folder permissions
sudo chmod a+w language-drift

# download semeval test data
sudo ./scripts/import_semeval_en.sh

#TODO add - glove mkdir+wget 
```

## API

```

```

## Tests

```
# create the appropriate folder structure for "1st-trial"
bash ./scripts/prepare_data.sh 1st-trial ./data/en_semeval/corpus1/lemma.txt.gz ./data/en_semeval/corpus2/lemma.txt.gz ./data/en_semeval/corpus1/token.txt.gz ./data/en_semeval/corpus2/token.txt.gz ./data/en_semeval/targets/targets.tsv ./data/en_semeval/truth/binary.tsv ./data/en_semeval/truth/graded.tsv

# run SGNS without pretrained embeddings
bash ./scripts/discover_sgns.sh 1st-trial 10 50 5 0.001 3 3 5 1.0 en None None

# run SGNS with Glove pretrained embeddings
bash ./scripts/discover_sgns.sh 1st-trial 10 50 5 0.001 3 3 5 1.0 en Glove ./pretrained_embed/glove.6B
```

## Support


## Contributing



## License
The project is licensed under the Apache 2.0 license.
