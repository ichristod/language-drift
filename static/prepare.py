import numpy as np
import json
import preprocess as pp
from tqdm import tqdm
from gensim import corpora, models
from training_utils import get_file_name

N_TOPICS = 20


def get_train_data(encoded_docs):
    data = []
    # new ids are created here
    for index, (_, doc) in tqdm(enumerate(encoded_docs)):
        windows = pp.get_windows(doc)
        # index represents id of a document,
        # windows is a list of (word, window around this word),
        # where word is in the document
        data += [[index, w[0]] + w[1] for w in windows]
    data = np.array(data, dtype='int32')
    #print("get_train_data(encoded_docs): ",data[:3])
    return data


def prepare(dataset,path_to_save,dataset_filename):
    with open(dataset, 'r') as fp:
        texts = json.load(fp)
        print("texts[:2]: ",texts[:2])

    #print("dataset: ", dataset)
    encoded_docs, decoder, word_counts = pp.preprocess(texts)
    word_counts = np.array(word_counts)
    unigram_distribution = word_counts / sum(word_counts)
    data = get_train_data(encoded_docs)

    np.save(str(path_to_save)+'/npy/'+dataset_filename+'_unigram_distribution', unigram_distribution)
    np.save(str(path_to_save)+'/npy/'+dataset_filename+'_data', data)
    np.save(str(path_to_save)+'/npy/'+dataset_filename+'_decoder', decoder)
    print(f"unigram_distribution, data, and decoder saved!")

    # get LDA
    print("preprocess LDA starts...")
    htexts = [[decoder[j] for j in doc] for i, doc in encoded_docs]
    dictionary = corpora.Dictionary(htexts)
    corpus = [dictionary.doc2bow(text) for text in htexts]

    lda = models.LdaModel(corpus, alpha='auto', id2word=dictionary, num_topics=N_TOPICS, passes=20)
    corpus_lda = lda[corpus]
    doc_weights_init = np.zeros((len(corpus_lda), N_TOPICS))
    for i in tqdm(range(len(corpus_lda))):
        topics = corpus_lda[i]
        for j, prob in topics:
            doc_weights_init[i, j] = prob
    np.save(str(path_to_save)+'/npy/'+dataset_filename+'_doc_weights_init', doc_weights_init)
    print("preprocess LDA done! doc_weights_init saved!")


if __name__ == '__main__':
    prepare()