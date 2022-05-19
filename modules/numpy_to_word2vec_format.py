import argparse
import json
import numpy as np
import torch
import fileinput
import sys
import re

def main():
    """
    Create word2vec format files from nparray
    """

    # Get the arguments
    parser = argparse.ArgumentParser(
        description='Create json files of [[idx_1,"sentence_1"], ["idx_2,sentence_2"]]')

    parser.add_argument('dataset_path', help='path to corpus1 directory with zipped files')
    parser.add_argument('trained_dir', help='path to corpus1 directory with zipped files')

    args = parser.parse_args()
    dataset_path = args.dataset_path
    trained_dir = args.trained_dir

    print("dataset_path: ",dataset_path)
    print("trained_dir: ",trained_dir)

    with open('./'+dataset_path, 'r') as fp:
        texts = json.load(fp)

    # data/en_4.0.0/corpus1/lemma_docids.json -> corpus1
    dataset_filename = str(args.dataset_path).rsplit('/', 2)[1]

    decoder = np.load('./'+trained_dir+'/npy/'+dataset_filename+'_decoder.npy',allow_pickle=True).item()

    # corpus1 -> 1
    matrix_name = 'mat'+str(int([float(n) for n in re.findall(r'-?\d+\.?\d*', dataset_filename)][-1] ))
    state = torch.load('./'+trained_dir+'/'+matrix_name+'.pt',
                       map_location=lambda storage, loc: storage)
    word_vectors = state['neg.embedding.weight'].cpu().clone().numpy()
    rev_decoder = {v: k for k, v in decoder.items()}
    vocab = [i for i, j in sorted(rev_decoder.items(), key=lambda x: x[1])]

    rev_decoder_list = list(rev_decoder)

    for i in range(len(vocab)):
        word = rev_decoder_list[i]
        vector = word_vectors[:i + 1]

    c = np.savetxt(trained_dir+'/'+matrix_name, vector, delimiter=' ', fmt='%.8f')

    for idx, line in enumerate(fileinput.input([trained_dir+'/'+matrix_name], inplace=True)):
        sys.stdout.write(rev_decoder_list[idx] + ' {l}'.format(l=line))

    f = open(trained_dir+'/'+matrix_name, 'r+')
    lines = f.readlines()  # read old content
    f.seek(0)  # go back to the beginning of the file
    f.write(str(len(vocab)) + " " + str(100) + "\n")  # write new content at the beginning
    for line in lines:  # write old content after new
        f.write(line)
    f.close()



if __name__ == '__main__':
    main()