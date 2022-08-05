import gzip
import json
import os
import torch
import numpy as np
import fileinput
import sys

count=0
docs=[]

with gzip.open('data/de_4.0.0/corpus1/lemma.txt.gz','rt',encoding="utf-8") as corpus1_docids:

    for sentence in corpus1_docids:
        docs.append(sentence.split("\n")[0])

    print("docs: ", docs[:1])
    list_sentences =[]
    # ["sentence_1","sentence_2"] -> [["sentence_1"],["sentence_2"]]
    for idx,element in enumerate(docs):
        docs[idx] = [value for value in str(element).split(" ") if len(value)>2 ]
    print("docs[:2]): ",docs[:1])
    list_results = [[element] for element in docs if len(str(element).split(" ")) >11]

    # 2222641
    # list_results:  [['d Schwierigkeit d Einheit und inner Kontinuität d inmateriell teilen nicht vorstellbar Geist mit seine Individuation und Austeilung an d Vielheit d Seele zu vereinigen kehren in griechisch Philosophie noch oft wieder']]
    print("list_results: ", len(list_results))
    list_results = list_results[:int(len(list_results)/30)]
    print("list_results: ", list_results[:1])
    print("max_sentence_OLD: ",max([len(str(sublist[-1]).split(" ")) for sublist in list_results]))
    print("min_sentence_OLD: ",min([len(str(sublist[-1]).split(" ")) for sublist in list_results]))

    print("max_sentence: ",max([len(str(sublist[-1]).split(" ")) for sublist in list_results if len(str(sublist[-1]).split(" ")) > 10]))
    print("min_sentence: ",min([len(str(sublist[-1]).split(" ")) for sublist in list_results if len(str(sublist[-1]).split(" ")) > 10]))

    print(len(list_results))
    for counter,value in enumerate(list_results):
        if (len(str(value[-1]).split(" ")) > 10):
            list_results[counter].insert(0,counter)
    print(len(list_results[:4]))

    print("max_sentence_NEW: ",max([len(str(sublist[-1]).split(" ")) for sublist in list_results]))
    print("min_sentence_NEW: ",min([len(str(sublist[-1]).split(" ")) for sublist in list_results]))


"""
import numpy as np
file_path='data/${dataset_id}/corpus1/lemma_docids.json'
path_to_save='./output/en_4.0.0/lda2vec_topic20_dim150_epochs5/trained_models'
file_name = file_path.rsplit('/', 2)[1]


unigram_distribution = np.load('./' + str(path_to_save) + '/npy/' + str(file_path.rsplit('/', 2)[1]) + '_unigram_distribution.npy', allow_pickle=True)


# load '../checkpoint/5_epoch_model_state.pt'
with open('./data/en_4.0.0/corpus1/lemma_docids.json', 'r') as fp:
    texts = json.load(fp)

decoder = np.load('./output/en_4.0.0/lda2vec_topic20_dim150_epochs5/trained_models/npy/corpus1_decoder.npy', allow_pickle=True).item()

state = torch.load('./output/en_4.0.0/lda2vec_topic20_dim150_epochs5/trained_models/mat1.pt', map_location=lambda storage, loc: storage)
word_vectors = state['neg.embedding.weight'].cpu().clone().numpy()
rev_decoder = {v: k for k,v in decoder.items()}
vocab = [i for i,j in sorted(rev_decoder.items(), key=lambda x:x[1])]

rev_decoder_list = list(rev_decoder)

for i in range(len(vocab)):
    word = rev_decoder_list[i]
    vector = word_vectors[:i+1]

def get_vector(token,word_vectors):
    index = vocab.index(token)
    return word_vectors[index, :].copy()

c = np.savetxt('testttttt',vector,delimiter =' ',fmt='%.8f')

for idx,line in enumerate(fileinput.input(['testttttt'], inplace=True)):
    sys.stdout.write(rev_decoder_list[idx]+' {l}'.format(l=line))

f = open('testttttt','r+')
lines = f.readlines() # read old content
f.seek(0) # go back to the beginning of the file
f.write(str(len(vocab))+" "+str(150)+"\n") # write new content at the beginning
for line in lines: # write old content after new
    f.write(line)
f.close()

a = open("testttttt", 'r')  # open file in read mode

print("the file contains:")
print(a.read())
#print(list(rev_decoder)[0])
#print(list(decoder)[0])
print(len(vocab))
import re
file_name='corpus1'
print(int([float(n) for n in re.findall(r'-?\d+\.?\d*', file_name)][-1] ))

"""