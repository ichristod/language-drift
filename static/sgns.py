import time
import argparse
import re
import gensim
import logging

from gensim.models.word2vec import PathLineSentences
from gensim.models import KeyedVectors
from twec import TWEC
from training_utils import get_file_prev_version,get_full_corpus_model,retrieve_embeddings_to_load
import sys

# Acceptable pretrained embeddings dimensions
pretrained_dim = [50, 100, 200, 300]
# Acceptable pretrained embeddings
list_of_pretrained = ['glove', 'dewiki', 'latconll17', 'sweconll17','spaconll17']


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
    total_examples = model.corpus_count
    # retrieve sentences
    sentences = PathLineSentences(path_corpus)

    return model, total_examples, sentences


def train_word2vec_model(pretrained_matrix, embeddings_to_load, apply_incremental, apply_twec, dim, word2vec_model, total_examples,
                         sentences):
    # check if pretrained embeddings exist
    if not pretrained_matrix:
        word2vec_model.train(sentences, total_examples=total_examples, epochs=word2vec_model.epochs)
    elif (pretrained_matrix in list_of_pretrained) or apply_incremental or apply_twec:
        # check if exists pretrained embedding with given dimensions
        if dim in pretrained_dim:
            # initialize embeddings
            model_wv = KeyedVectors.load_word2vec_format(embeddings_to_load, binary=False)
            word2vec_model.build_vocab([list(model_wv.vocab.keys())], update=True)
            word2vec_model.intersect_word2vec_format(embeddings_to_load, binary=False, lockf=1.0)
            word2vec_model.train(sentences, total_examples=total_examples, epochs=word2vec_model.epochs)

    return word2vec_model


def main():
    """
    Make embedding vector space with Negative Sampling from corpus.
    """

    # Get the arguments
    parser = argparse.ArgumentParser(
        description='Make embedding vector space with Skip-Gram with Negative Sampling from corpus.')

    parser.add_argument('path_corpus', help='path to corpus directory with zipped files')
    parser.add_argument('path_output', help='output path for vectors')
    parser.add_argument('window_size', type=int,
                        help='the linear distance of context words to consider in each direction')
    parser.add_argument('dim', type=int, help='dimensionality of embeddings')
    parser.add_argument('k', type=int,
                        help='number of negative samples parameter (equivalent to shifting parameter for PPMI)')
    parser.add_argument('s', help='threshold for subsampling')
    parser.add_argument('min_count', type=int, help='Number of occurrences for a word to be included in the vocabulary')
    parser.add_argument('itera', type=int, help='number of iterations')
    parser.add_argument('mapping', help='indicates type of embeddings mapping')
    parser.add_argument('w2vec_method', help='word2vec algorithm - sgns/cbow')
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
    w2vec_method = args.w2vec_method

    if args.s == 'None':
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

    if w2vec_method == 'sgns':
        sg = 1
    elif w2vec_method == 'cbow':
        sg = 0
    else:
        print('ERROR: Definition of word2vec algorithm is missing', file=sys.stderr)
        sys.exit(-1)

    # initialize word2vec model
    model, total_examples, sentences = initialize_word2vec_model(path_corpus=path_corpus, algorithm=sg,
                                                                 hs=0, neg_samples=k, sampl_threshold=s, dim=dim,
                                                                 window_size=window_size, min_count=min_count,
                                                                 iterations=itera, workers=40)

    # check incremental training conditions
    if mapping == 'incremental':
        # get previous matrix info
        previous_version, file_full_path = get_file_prev_version(path_output)

        if previous_version > 0:
            # incremental training of next corpus
            # load previous model
            embeddings_to_load = retrieve_embeddings_to_load(
                pretrained, path_pretrained, dim, True, file_full_path)
            print("Incremental: second corpus")
            model = train_word2vec_model(pretrained, embeddings_to_load, True, False,
                                         dim, model, total_examples, sentences)
        else:
            # initial training of corpus1 without pretrained embeddings

            embeddings_to_load = retrieve_embeddings_to_load(
                pretrained, path_pretrained, dim, False, file_full_path)
            print("Incremental: train first corpus")
            model = train_word2vec_model(pretrained, embeddings_to_load, False, False,
                                         dim, model, total_examples, sentences)
    elif mapping == 'twec':
        file_full_path = get_full_corpus_model(path_output)
        aligner = TWEC(size=dim, siter=model.epochs, diter=model.epochs, workers=model.workers,
                               sg=model.sg, ns=model.negative, min_count=model.vocabulary.min_count, window=model.window)

        if file_full_path:
            # load full corpus model if any
            print("TWEC: train slices")
            aligner.train_slice(path_corpus, path_output, save=True)

        else:
            # initialize full corpus model
            print("TWEC: train full corpus")
            # train the compass: the text should be the concatenation of the text from the slices
            aligner.train_compass(path_corpus, path_output, overwrite=False)
    else:
        # define embeddings
        print("Baseline training")
        embeddings_to_load = retrieve_embeddings_to_load(pretrained, path_pretrained, dim, False, None)
        model = train_word2vec_model(pretrained, embeddings_to_load, False, False,
                                     dim, model, total_examples, sentences)

    if is_len and mapping != 'twec':
        # L2-normalize vectors
        model.init_sims(replace=True)

    if mapping !='twec':
        # Save the vectors and the model
        model.wv.save_word2vec_format(path_output)
        model.save(path_output + '.model')

    logging.info("--- %s seconds ---" % (time.time() - start_time))
    print("")


if __name__ == '__main__':
    main()
