import csv
import logging
import time
import pandas as pd

from docopt import docopt
from gensim.models import KeyedVectors
import numpy as np


def main():
    """
    Compute jaccard index for target words' topN neighbors.
    """

    # Get the arguments
    args = docopt("""Compute jaccard index for target words' topN neighbors.

    Usage:
        jaccard_neighbors.py <path_src_embeddings> <path_trg_embeddings> <path_targets> <top_neighbors> <path_output> 

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

    model_src = KeyedVectors.load_word2vec_format(path_src_embeddings, binary=False)
    model_trg = KeyedVectors.load_word2vec_format(path_trg_embeddings, binary=False)

    df = pd.DataFrame()

    with open(path_targets, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE, strict=True)
        for row in reader:
            try:
                topn_src_similar = [i[0] for i in model_src.most_similar(positive=[row[0]],topn=top_neighbors)]
                topn_trg_similar = [i[0] for i in model_trg.most_similar(positive=[row[0]],topn=top_neighbors)]

                shared_items = [i for i, j in zip(topn_src_similar, topn_trg_similar) if i == j]

                df2 = pd.DataFrame({'target_words': [row[0]], 'jaccard_index': [len(shared_items) / top_neighbors],
                                    'shared_words': [shared_items]})
                df = pd.concat([df, df2], ignore_index=True, axis=0)

                df.to_csv(path_output, index=False)

            except ValueError:
                print("error in Jaccard Index calculation")


if __name__ == '__main__':
    main()
