import logging
import sys
sys.path.append('./modules/')
import time

from docopt import docopt
import gensim
from gensim.models.word2vec import PathLineSentences
from gensim.models import KeyedVectors

def main():
    """
    Make embedding vector space with Negative Sampling from corpus.
    """

    # Get the arguments
    args = docopt("""Make embedding vector space with Skip-Gram with Negative Sampling from corpus.

    Usage:
        sgns.py [-l] <path_corpus> <path_output> <window_size> <dim> <k> <s> <min_count> <itera> <embed_pretrained> <path_pretrained>
         
    Arguments:
       
        <path_corpus>      = path to corpus directory with zipped files
        <path_output>      = output path for vectors
        <window_size>      = the linear distance of context words to consider in each direction
        <dim>              = dimensionality of embeddings
        <k>                = number of negative samples parameter (equivalent to shifting parameter for PPMI)
        <s>                = threshold for subsampling
        <min_count>        = number of occurrences for a word to be included in the vocabulary
        <itera>            = number of iterations
        <embed_pretrained> = option of pretrained embeddings (None, Glove)
        <path_pretrained>  = path to pretrained embeddings directory with txt files
        

    Options:
        -l, --len   normalize final vectors to unit length

    """)

    path_corpus = args['<path_corpus>']
    path_output = args['<path_output>']
    window_size = int(args['<window_size>'])

    dim = int(args['<dim>'])
    k = int(args['<k>'])
    if args['<s>']=='None':
        s = None
    else:
        s = float(args['<s>'])
    min_count = int(args['<min_count>'])
    itera = int(args['<itera>'])

    embed_pretrained = args['<embed_pretrained>']
    path_pretrained = args['<path_pretrained>']

    is_len = args['--len']

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    logging.info(__file__.upper())
    start_time = time.time()

    # Acceptable pretrained embeddings dimensions
    pretrained_dim=[50,100,200,300]

    # Initialize model
    model = gensim.models.Word2Vec(sg=1, # skipgram
                                    hs=0, # negative sampling
                                   negative=k, # number of negative samples
                                   sample=s, # threshold for subsampling, if None, no subsampling is performed
                                   size=dim, window=window_size, min_count=min_count, iter=itera, workers=40)

    # Initialize vocabulary
    vocab_sentences = PathLineSentences(path_corpus)
    logging.getLogger('gensim').setLevel(logging.ERROR)
    model.build_vocab(vocab_sentences)
    total_examples=model.corpus_count
    # retrieve sentences
    sentences = PathLineSentences(path_corpus)

    if not embed_pretrained:
        model.train(sentences, total_examples=total_examples, epochs=model.epochs)

    elif embed_pretrained=='Glove':

        # check if exists pretrained embedding with given dimensions
        if dim in pretrained_dim:
            embeddings_to_load = path_pretrained+"/glove.6B."+str(dim)+"d.txt"
            model_wv = KeyedVectors.load_word2vec_format(embeddings_to_load, binary=False)
            model.build_vocab([list(model_wv.vocab.keys())], update=True)
            model.intersect_word2vec_format(embeddings_to_load, binary=False, lockf=1.0)
            model.train(sentences, total_examples=total_examples, epochs=model.epochs)
    else:
        pass


    if is_len:
        # L2-normalize vectors
        model.init_sims(replace=True)

    # Save the vectors and the model
    model.wv.save_word2vec_format(path_output)
    model.save(path_output + '.model')

    logging.info("--- %s seconds ---" % (time.time() - start_time))
    print("")



if __name__ == '__main__':
    main()
