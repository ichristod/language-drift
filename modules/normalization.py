import logging
import time

import numpy as np

import embeddings
from docopt import docopt


def main():
    # Get the argument
    args = docopt("""Normalize embeddings
    
    Usage:
        normalization.py <src_input> <trg_input>
        
        <src_input>     = the input source embeddings
        <trg_input>     = the input target embeddings
       
    """)

    src_input = args['<src_input>']
    trg_input = args['<trg_input>']


    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    logging.info(__file__.upper())

    start_time = time.time()

    # Read input embeddings
    srcfile = open(src_input, errors='surrogateescape')
    trgfile = open(trg_input, errors='surrogateescape')

    src_words, x = embeddings.read(srcfile, dtype='float64')
    trg_words, z = embeddings.read(trgfile, dtype='float64')

    # Allocate memory
    xw = np.empty_like(x)
    zw = np.empty_like(z)

    # STEP 0: Normalization
    embeddings.normalize(x, ['unit','center'])
    embeddings.normalize(z, ['unit','center'])

    logging.info("--- %s seconds ---" % (time.time() - start_time))
    print("")


if __name__ == '__main__':
    main()
