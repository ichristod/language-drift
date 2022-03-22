import csv
import logging
import time
import pandas as pd

from docopt import docopt
from gensim.models import KeyedVectors
from scipy.spatial.distance import cosine as cosine_distance

import numpy as np


def main():
    """
    Compute local neighborhood measure for target words' topN neighbors.
    """

    # Get the arguments
    args = docopt("""Compute local neighborhood measure for target words' topN neighbors.

    Usage:
        local_neighborhood.py <path_src_embeddings> <path_trg_embeddings> <path_targets> <top_neighbors> <path_output> 

        <path_src_embeddings>    = path to file A containing embeddings
        <path_trg_embeddings>    = path to file B containing embeddings
        <path_targets>           = path to target words file
        <top_neighbors>          = number of top similar neighbor words
        <path_output>            = output path for result file

    """)

    path_src_embeddings = args['<path_src_embeddings>']
    path_trg_embeddings = args['<path_trg_embeddings>']

    # path to file containing words to be examined
    path_targets = args['<path_targets>']
    top_neighbors = int(args['<top_neighbors>'])
    path_output = args['<path_output>']

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    logging.info(__file__.upper())
    start_time = time.time()

    # load embeddings matrices
    model_src = KeyedVectors.load_word2vec_format(path_src_embeddings, binary=False)
    model_trg = KeyedVectors.load_word2vec_format(path_trg_embeddings, binary=False)

    # calculate averages for each document
    avg_src = np.average(model_src[model_src.vocab], axis=0)
    avg_trg = np.average(model_trg[model_trg.vocab], axis=0)

    df = pd.DataFrame()
    distances = {}

    with open(path_targets, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE, strict=True)
        for row in reader:
            try:
                top_src_similar = list(set([i[0] for i in model_src.most_similar(positive=[row[0]],topn=top_neighbors)]))
                top_trg_similar = list(set([i[0] for i in model_trg.most_similar(positive=[row[0]],topn=top_neighbors)]))

                all_words = set(top_src_similar + top_trg_similar)

                vec_1 = []
                vec_2 = []

                for inner_word in all_words:
                    if inner_word in model_src.vocab:
                        vec_1.append(1 - cosine_distance(model_src[row[0]], model_src[inner_word]))
                    else:
                        vec_1.append(1 - cosine_distance(model_src[row[0]], avg_src))

                for inner_word in all_words:
                    if inner_word in model_trg.vocab:
                        vec_2.append(1 - cosine_distance(model_trg[row[0]], model_trg[inner_word]))
                    else:
                        vec_2.append(1 - cosine_distance(model_trg[row[0]], avg_trg))

                distances[row[0]] = 1 - cosine_distance(vec_1, vec_2)

                # Write output to <path_output>
                with open(path_output, 'w', encoding='utf-8') as f:
                    for key in distances:
                        f.write(key + '\t' + str(distances[key]) + '\n')

            except ValueError:
                print("error in Local Neighborhood Measure")

    logging.info("--- %s seconds ---" % (time.time() - start_time))
    print("")


if __name__ == '__main__':
    main()
