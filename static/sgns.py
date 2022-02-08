import logging
import sys
sys.path.append('./modules/')
import time
import argparse
import re

from docopt import docopt
import gensim
from gensim.models.word2vec import PathLineSentences
from gensim.models import KeyedVectors
# from gensim.scripts.glove2word2vec import glove2word2vec
# from gensim.test.utils import get_tmpfile

# Acceptable pretrained embeddings dimensions
pretrained_dim = [50, 100, 200, 300]
# Acceptable pretrained embeddings
list_of_pretrained = ['glove']


def get_file_prev_version(path_corpus):
    """"
    example values:
    file_path           --> output/1st-try/sgns_win10_dim50_k5_s0.001_mc3_mc3_i5_nonincr_glove/discovery/t1.0
    file_name           --> mat32t32
    previous_version    -->  31
    previous_file_name  --> ma32t31
    file_full_path      --> output/1st-try/sgns_win10_dim50_k5_s0.001_mc3_mc3_i5_nonincr_glove/discovery/t1.0/mat32t31
    """
    file_path = path_corpus.rsplit('/', 1)[0]
    file_name = path_corpus.rsplit('/', 1)[1]
    previous_version = int([float(n) for n in re.findall(r'-?\d+\.?\d*', file_name)][-1] - 1)
    previous_file_name = file_name[::-1].replace(str(previous_version+1)[::-1], str(previous_version)[::-1], 1)[::-1]
    file_full_path = file_path + "/" + previous_file_name

    return previous_version, file_full_path


def initialize_word2vec_model(path_corpus, algorithm, hs, neg_samples, sampl_threshold, dim,
                              window_size, min_count, iterations, workers):
    # Initialize model
    model = gensim.models.Word2Vec(sg=algorithm,  # skipgram
                                   hs=hs,  # negative sampling
                                   negative=neg_samples,  # number of negative samples
                                   sample=sampl_threshold,
                                   # threshold for subsampling, if None, no subsampling is performed
                                   size=dim, window=window_size, min_count=min_count, iter=iterations, workers=workers)

    # Initialize vocabulary
    vocab_sentences = PathLineSentences(path_corpus)
    logging.getLogger('gensim').setLevel(logging.ERROR)
    # build vocabulary
    model.build_vocab(vocab_sentences)
    total_examples=model.corpus_count
    # retrieve sentences
    sentences = PathLineSentences(path_corpus)

    return model, total_examples, sentences


def train_word2vec_model(pretrained_matrix, embeddings_to_load, apply_incremental, dim, word2vec_model, total_examples, sentences):

    # check if pretrained embeddings exist
    if not pretrained_matrix:
        word2vec_model.train(sentences, total_examples=total_examples, epochs=word2vec_model.epochs)
    elif (pretrained_matrix in list_of_pretrained) or apply_incremental:
        # check if exists pretrained embedding with given dimensions
        if dim in pretrained_dim:
            # initialize embeddings
            model_wv = KeyedVectors.load_word2vec_format(embeddings_to_load, binary=False)
            word2vec_model.build_vocab([list(model_wv.vocab.keys())], update=True)
            word2vec_model.intersect_word2vec_format(embeddings_to_load, binary=False, lockf=1.0)
            word2vec_model.train(sentences, total_examples=total_examples, epochs=word2vec_model.epochs)



    return word2vec_model


def retrieve_embeddings_to_load(pretrained_matrix, pretrained_matrix_path, dim, apply_incremental, file_full_path):
    embeddings_to_load =''
    if apply_incremental:
        embeddings_to_load = file_full_path
    elif pretrained_matrix == 'glove':
        embeddings_to_load = pretrained_matrix_path + "/glove.6B." + str(dim) + "d.txt"

    return embeddings_to_load


def main():
    """
    Make embedding vector space with Negative Sampling from corpus.
    """

    # Get the arguments
    parser = argparse.ArgumentParser(description='Make embedding vector space with Skip-Gram with Negative Sampling from corpus.')

    parser.add_argument('path_corpus', help='path to corpus directory with zipped files')
    parser.add_argument('path_output', help='output path for vectors')
    parser.add_argument('window_size', type=int, help='the linear distance of context words to consider in each direction')
    parser.add_argument('dim', type=int, help='dimensionality of embeddings')
    parser.add_argument('k', type=int, help='number of negative samples parameter (equivalent to shifting parameter for PPMI)')
    parser.add_argument('s', help='threshold for subsampling')
    parser.add_argument('min_count', type=int, help='Number of occurrences for a word to be included in the vocabulary')
    parser.add_argument('itera', type=int, help='number of iterations')
    parser.add_argument('mapping', help='indicates type of embeddings mapping')
    parser.add_argument('w2vec_algorithm', help='word2vec algorithm - sgns/cbow')
    parser.add_argument('--pretrained', default='glove', help='option of pretrained embeddings (None, Glove)')
    parser.add_argument('--path_pretrained', help='path to pretrained embeddings directory with txt files')
    parser.add_argument('--len', action='store_true', help='normalize final vectors to unit length')

    args = parser.parse_args()

    path_corpus = args.path_corpus
    path_output = args.path_output
    window_size = args.window_size
    dim = args.dim
    k = args.k
    min_count = args.min_count
    itera = args.itera
    mapping = args.mapping
    w2vec_algorithm = args.w2vec_algorithm

    if args.s=='None':
        s = None
    else:
        s = float(args.s)

    if args.pretrained is not None:
        pretrained = args.pretrained
    if args.path_pretrained is not None:
        path_pretrained = args.path_pretrained
    if args.len is not None:
        is_len = bool(args.len)

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    logging.info(__file__.upper())
    start_time = time.time()

    if w2vec_algorithm == 'sgns':
        sg=1
    elif w2vec_algorithm == 'cbow':
        sg=0
    else:
        print('ERROR: Definition of word2vec algorithm is missing', file=sys.stderr)
        sys.exit(-1)

    # initialize word2vec model
    model,total_examples,sentences = initialize_word2vec_model(path_corpus=path_corpus,algorithm=sg,
                                                               hs=0,neg_samples=k,sampl_threshold=s,dim=dim,
                                                               window_size=window_size,min_count=min_count,
                                                               iterations=itera,workers=40)

    # check incremental training conditions
    if mapping == 'incremental':
        # get previous matrix info
        previous_version, file_full_path = get_file_prev_version(path_output)

        if previous_version > 0:
            # incremental training of next corpus
            # load previous model
            embeddings_to_load = retrieve_embeddings_to_load(
                pretrained, path_pretrained, dim, True, file_full_path)
            model = train_word2vec_model(pretrained, embeddings_to_load, True, dim, model, total_examples, sentences)
        else:
            # initial training of corpus1 without pretrained embeddings
            embeddings_to_load = retrieve_embeddings_to_load(
                pretrained, path_pretrained, dim, False, file_full_path)
            model = train_word2vec_model(pretrained, embeddings_to_load, False, dim, model, total_examples, sentences)
    else:
        # define embeddings
        embeddings_to_load = retrieve_embeddings_to_load(pretrained, path_pretrained, dim, False, None)
        model = train_word2vec_model(pretrained, embeddings_to_load, False,dim, model, total_examples, sentences)


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
