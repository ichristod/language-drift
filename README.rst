=========================================================
The impact of different representations in the presence of language drift
=========================================================


This repo contains Python code and scripts to handle with the task of **Diachronic Semantic Change Detection**.
Given the diachronic nature of the task, we compare at least two different text representations, one for each of the earlier and later corpora.
For the creation of text representation we utilized word (**Word2vec**), topic(**LDA2vec**) and pretrained embeddings (**Glove**, **Wikipedia2Vec**).
To obtain comparable text representations, we either trained them in a common space (**TWEC**, **Embeddings Initialization**) from the beginning or
to used an alignment method (**Orthogonal Procrustes**) to map the independently trained representations in that common space.
Finally, in order to quantify the degree of semantic change we applied **Cosine Distance** and **Local Neighborhood Measure**.

We have adopted a decoupled approach, separating the algorithmic implementations of the models, the evaluation measures,
and the actions that can be chosen in our pipeline. This approach contributes to a quick and efficient reproduction of the experiments via bash scripts.
It allows easy adaptations and extensions with new models for calculating word embeddings and new evaluation measures. In other words, we have created a pool of
methods and steps that can be combined into a single pipeline of actions. These actions depend on whether we want to train embeddings, use pre-trained ones,
use alignment methods, or use a different approach that learns representations in a common semantic space.

The implemented decoupled pipeline is described by the following diagram.

.. image:: /visualizations/images/pipeline.png


Reference
---------

This work is based on the following papers

+ Di Carlo et al.(2019). **Training Temporal Word Embeddings with a Compass**. Proceedings of the AAAI 2019 Conference on Artificial Intelligence, 33(01), https://doi.org/10.1609/aaai.v33i01.33016326
+ Kim et al.(2014). **Temporal Analysis of Language through Neural Language Models**. Proceedings of the ACL 2014 Workshop on Language Technologies and Computational Social Science, https://aclanthology.org/W14-2517
+ Hamilton et al.(2018). **Diachronic Word Embeddings Reveal Statistical Laws of Semantic Change**. https://arxiv.org/abs/1605.09096
+ Moody (2016). **Mixing Dirichlet Topic Models and Word Embeddings to Make lda2vec**. https://arxiv.org/abs/1605.02019
+ Hamilton et al.(2018). **Cultural Shift or Linguistic Drift? Comparing Two Computational Measures of Semantic Change**. Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing, https://aclanthology.org/D16-1229/

Abstract
--------

Natural language inherently contains an interpretation of the world in the form of vocabulary and the different meanings of words.
Language changes can reflect sociocultural evolution; therefore, their systematical exploration is a valuable tool to social and humanities sciences researchers.
In this thesis, we examine the detection of semantic changes between two time periods t1, t2. For the empirical study, we use datasets of four different languages
(English, German, Latin, and Swedish) provided from the SemEval-2020 Task 1. The whole set of our experiments is evaluated against a binary classification task,
depending on whether a word's sense changes or not. For that purpose, we explore a set of different approaches including methods that have not been previously
submitted in the SemEval-2020 Task 1. Furthermore, we create an extensible system which decouples each stage of the diachronic semantic change detection workflow
from the actual implementations. This approach contributes to a quick and efficient reproduction of the experiments, aiming to facilitate research in the domain
of semantic change. Based on the results of our empirical study, we answer three different questions.
The first is related to identifying the most suitable alignment method for the word embeddings Wt1, Wt2. The methods under investigation are the Orthogonal Procrustes,
the Embeddings Initialization, and the Temporal Word Embeddings with a Compass. The next question refers to the performance of the Word2vec pre-trained embeddings
compared to others whose weights had not been prior initialized. Finally, through the application of LDA2vec, we explore whether the LDA (Latent Dirichlet Allocation)
topics overcome the performance of the SGNS (Skip-gram with Negative Sampling) or not.


Installing
----------

* clone the repository
* :code:`conda create --name language-drift python=3.9.1`
* :code:`conda install pip`
* :code:`pip install -r requirements.txt`
* :code:`sudo chmod a+w language-drift`
* add manually glove in pretrained_embed/glove.6B/
* :code:`bash ./reproduce.sh`


Guide
-----

* The implementation handles the task of **binary classification** for the detection of semantic change on targeted words.

* These words are annotated and provided from the [SemEval-2020 Task 1: Unsupervised Lexical Semantic Change Detection](https://aclanthology.org/2020.semeval-1.1).

* In order to identify whether a word lost, gained or kept it's initial sense we use the following threshold on the cosine distance of each word (W) in corpora C1 from the same word in corpora C2.
 :code:`Distance(Wc1,Wc2) > threshold = mean(list_of_distances_of_target_words) + standardError(list_of_distances_of_target_words)`

How To Use
----------

The main parameters of the reproduce script are provided though the configuration files of **exec.conf** and **exec_lda2vec.conf**.

Below you can find a description of the parameters:

* languages --> [en, de, lat, swe]
* algorithms --> [sgns, cbow, lda2vec]
* mappings --> [alignment, incremental, twec]
* top neighbors of local neighborhood measure --> e.g. 5, 10
* threshold multiplier --> e.g. 1.0, 1.5, 2.0
* version of the execution --> e.g. 1.0.0

There are also available some functionalities for the first execution:

* download_datasets for the supported languages --> true/false
* download_pretrained embeddings --> true/false
* prepare_datasets folder structure --> true/false


**Word2vec** parameters are described in **classify_sgns.sh**.

An example of that script is presented below.

:code:`bash ./scripts/classify_sgns.sh ${dataset_id} 10 100 5 0.001 3 3 5 ${threshold} ${mapping} ${w2vec_method} ${language} ${pretrained_embed} ${pretrained_path}`


After completing the installation steps, the relevant results can be  plots are produced from the execution of **statistic_tests.ipynb**

Results
------
Results are presented in the following table and can also be found in **language_drift_results.csv** file.

    .. image:: /visualizations/images/results.png

+ **Which vectors’ alignment method performs better?**

            The number of samples were

    .. image:: /visualizations/images/opsamples.png

            **F1 Scores per representation model**

    .. image:: /visualizations/images/experiment1.png

+ **Do pre-trained embeddings improve performance?**

            The number of samples were

    .. image:: /visualizations/images/pretrainedsamples.png

            **F1 Scores per representation model**

    .. image:: /visualizations/images/experiment2.png

+ **Do the representations of LDA2vec and Word2vec perform the same**

            The number of samples were

    .. image:: /visualizations/images/lda2vecsaples.png

            **F1 Scores per representation model**

    .. image:: /visualizations/images/experiment3.png

+ **LDA2vec and Local Neighborhood Measure?**

            **F1 Scores per similarity measure**

    .. image:: /visualizations/images/experiment4.png



Credits
-------

Parts of the code rely on [LSCDetection](https://github.com/Garrafao/LSCDetection), [fuzzywuzzy](https://github.com/seatgeek/fuzzywuzzy), [gensim](https://github.com/rare-technologies/gensim), [numpy](https://pypi.org/project/numpy/), [scikit-learn](https://pypi.org/project/scikit-learn/), [scipy](https://pypi.org/project/scipy/), [VecMap](https://github.com/artetxem/vecmap), [TWEC](https://github.com/valedica/twec) and [LDA2vec](https://github.com/cemoody/lda2vec).
