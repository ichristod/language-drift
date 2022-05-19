import numpy as np
import os
import torch
from training import train
from prepare import prepare, N_TOPICS
from training_utils import get_file_name
import argparse

def parse_args():
    desc = "Pytorch implementation of lda2vec"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--n_epochs', type=int, default=5, help='The number of training n_epochs')
    parser.add_argument('--batch_size', type=int, default=1024 * 4, help='The size of batch size')

    parser.add_argument('--lambda_const', type=float, default=100.0, help='Strength of dirichlet prior.')
    parser.add_argument('--num_sampled', type=int, default=15, help='Number of negative words to sample.')
    parser.add_argument('--topics_weight_decay', type=float, default=1e-2, help='L2 regularization for topic vectors.')
    parser.add_argument('--topics_lr', type=float, default=1e-3, help='Learning rate for topic vectors.')
    parser.add_argument('--doc_weights_lr', type=float, default=1e-3, help='Learning rate for document weights.')
    parser.add_argument('--word_vecs_lr', type=float, default=1e-3, help='Learning rate for word vectors.')

    parser.add_argument('--save_every', type=int, default=10, help='Save the model from time to time.')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Clip gradients by absolute value.')

    parser.add_argument('--device', type=str, default='cpu', help='Set gpu mode; [cpu, cuda:0, cuda:1, ...]')
    parser.add_argument('--num_workers', type=int, default='4', help='DataLoader num_workers')
    parser.add_argument('--dataset', help='dataset to create embeddings from')
    parser.add_argument('--dimension', type=int, default='100', help='embeddings dimension')
    parser.add_argument('--path_to_save', help='path to save trained model')

    return parser.parse_args()

def load(path_to_save,dataset_filename):
    unigram_distribution = np.load('./' +str(path_to_save)+'/npy/'+str(dataset_filename)+'_unigram_distribution.npy', allow_pickle=True)
    decoder = np.load('./' +str(path_to_save)+'/npy/'+str(dataset_filename)+'_decoder.npy', allow_pickle=True)
    data = np.load('./' +str(path_to_save)+'/npy/'+str(dataset_filename)+'_data.npy', allow_pickle=True)
    doc_weights_init = np.load('./' +str(path_to_save)+'/npy/'+str(dataset_filename)+'_doc_weights_init.npy', allow_pickle=True)
    return unigram_distribution, decoder, data, doc_weights_init


def main():
    args = parse_args()
    path_to_save = args.path_to_save
    os.makedirs(str(path_to_save)+'/npy', exist_ok=True)

    # retrieve 'corpus1' or 'corpus2'
    dataset_filename = str(args.dataset).rsplit('/', 2)[1]
    print("dataset_filename: ",dataset_filename)
    print("path_to_save: ",path_to_save)

    try:
        unigram_distribution, decoder, data, doc_weights_init = load(path_to_save,dataset_filename)
    except:
        print(f"Required preprocess not done! Wait till preprocess done! ")
        prepare(args.dataset,path_to_save,dataset_filename)
        unigram_distribution, decoder, data, doc_weights_init = load(path_to_save,dataset_filename)
        print("Preprocess done!")
        print("")



    decoder = decoder.item()
    word_vectors = np.random.normal(0, 0.01, (len(decoder), args.dimension))
    word_vectors = torch.FloatTensor(word_vectors).to(args.device)

    train(
        args, data, unigram_distribution, word_vectors,
        dataset_filename, args.path_to_save,
        doc_weights_init=doc_weights_init,
        n_topics=N_TOPICS
    )



if __name__ == '__main__':
    main()