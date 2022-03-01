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

###TODO add - glove mkdir+wget 

# reproduce results
bash ./reproduce_experiment.sh


```

## API
The implementation handles the task of **binary classification** for the detection of semantic change on targeted words.

These words are annotated and provided from the [SemEval-2020 Task 1: Unsupervised Lexical Semantic Change Detection](https://aclanthology.org/2020.semeval-1.1) (Schlechtweg et al., SemEval 2020).

In order to identify whether a word lost, gained or kept it's initial sense we use the following threshold on the cosine distance of each word (W) in corpora C1 from the same word in corpora C2.

```
Distance(Wc1,Wc2) > threshold = mean(list_of_distances_of_target_words) + standardError(list_of_distances_of_target_words)
```

### - reproduce_experiment script
The main parameters of the reproduce_experiment script are:
- languages - **en**, **de**, **lat** and **swe**.
- word2vec algorithm
  - skip-gram with negative-sampling - **sgns**
  - continuous bag of words - **cbow**
- mappings
  - **alignment** with orthogonal procrustes
  - **incremental** fine-tuning approach
- threshold multiplier - e.g. **1.0**, **1.5**, **2.0**
- version of the execution - e.g. **1.0.0**

There are also available some functionalities for the first execution:
1. download_datasets for the supported languages   - **true**/**false**
2. download_pretrained embeddings (except swedish) - **true**/**false**
3. prepare_datasets folder structure - **true**/**false**

####TODO: Create configuration file
### - word2vec parameters
Word2vec parameters are described in **classify_sgns.sh**. 
An example of that script is presented below. 
```
bash ./scripts/classify_sgns.sh ${dataset_id} 10 100 5 0.001 3 3 5 ${threshold} ${mapping} ${w2vec_method} ${language} ${pretrained_embed} ${pretrained_path}
```
However, its recommended usage is through **reproduce_experiment.sh**  

### - results (under investigation)
So far results can be found on **language_drift_results.csv**


## Usage
After completing the installation steps, the appropriate plots are produced from the execution of **create_visualizations.py**
```
language = 'en' | 'de' | 'lat' | 'swe'
embed = 'glove' | 'dewiki' | 'lat_conll17' | ''
path = './results/**/**/classification'
```

## Support

## Contributing


## License
The project is licensed under the Apache 2.0 license.
