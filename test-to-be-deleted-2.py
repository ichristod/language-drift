import csv
import logging
import time
from scipy.stats import sem

from docopt import docopt
import numpy as np

topn_src_similar=['asd','wer','zxc']
label='asd'
predicted = True if label in topn_src_similar else False
print(int(predicted))

"""
path_distances="output/lat_4.0.0/lda2vec_topic20_dim100_epochs5_procrustes/distances/distances_intersection.tsv"

# Load data
distances = {}
with open(path_distances, 'r', encoding='utf-8') as f:
    reader = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE, strict=True)
    for row in reader:
        try:
            distances[row[0]] = float(row[1])
        except ValueError:
            pass

# Compute mean, std and threshold
list_distances = np.array(list(distances.values()))

mean = np.mean(list_distances, axis=0)
std = np.std(list_distances, axis=0)
stde = sem(list_distances, axis=0)
print("list.size: ",list_distances.size,"\n","mean: ", mean,"\n","std: ",std,"\n","stde: ", stde)

# Compute mean, std and threshold
list_distances = np.array(list(distances.values()))
upper_quantile = np.quantile(list_distances, 0.2)
list_distances = list_distances[list_distances < upper_quantile]
mean = np.mean(list_distances, axis=0)
std = np.std(list_distances, axis=0)
stde = sem(list_distances, axis=0)
print("list.size: ",list_distances.size,"\n","quantile:",upper_quantile,"\n","mean: ", mean,"\n","std: ",std,"\n","stde: ", stde)


from gensim.models.word2vec import Word2Vec, LineSentence, PathLineSentences
import os
import pandas as pd


df = pd.DataFrame('x', index=range(3), columns=list('abcde'))
print(df)
df.rename({'a': 'X', 'b': 'Y'}, axis=1)
print(df)
df.rename({'a': 'X', 'b': 'Y'}, axis=1, inplace=True)
print(df)

slice_text='output/en_4.0.0/cbow_win10_dim100_k5_s0.001_mc3_mc3_i5_incremental/results/t1.0/pickled_classification_res.pkl'
sentences = LineSentence(slice_text)

model_name = os.path.splitext(os.path.basename(slice_text))[0]
slice_text = slice_text.rsplit('/', 1)[0]
compass_exists = os.path.isfile(slice_text+"/mat1.model")

print("compass_exists: ",compass_exists)
print("slice_text: ",slice_text+"/pickled_classification"+"."+shortcode)


"""
matrix = np.array([[1, 2, 3], [4, 5, 6],[7,8,9]], dtype='float64')
norms = np.sqrt(np.sum(matrix ** 2, axis=1))
print(norms)
# QA: explain the below expression
norms[norms == 0] = 1
print(norms)
matrix /= norms[:, np.newaxis]

print("norms: ",norms)
print("matrix: ",matrix)