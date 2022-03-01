import pandas as pd
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import os

import logging
import time

import csv
import logging
import time
import pandas as pd

from docopt import docopt
from sklearn import metrics

def main():
    """Compute precission, recall, f_1 and f_0.5.
    """

    # Get the argument
    args = docopt("""Create classification report

    Usage:
        class_metrics.py <path_truth> <path_file>  <path_output> <mapping> <w2vec_algorithm> \
                         <pretrained> <window_size> <dim> <t> <data_set_id>

        <path_pickle>           = path to pickle files containing results
        <dataset_id>            = dataset id - $language_$version
        <language>              = language
        <pretrained>            = pretrained

    """)

    path_pickle = args['<path_pickle>']
    dataset_id = args['<dataset_id']
    language = args['<language>']

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    logging.info(__file__.upper())
    start_time = time.time()

    my_path = os.path.abspath(__file__)  # Figures out the absolute path for you in case your working directory moves around.

    text_files = glob.glob(path_pickle + "/**/*.pkl", recursive=True)

    df = pd.DataFrame()
    for file in text_files:
        df = pd.concat([df, pd.read_pickle(file)], ignore_index=True, axis=0)

    # Initial vs Pretrained (across - mapping)
    my_file = 'InitialvsPretrained_across.png'
    sns.set()
    pre = sns.relplot(x='w2vec_algorithm', y='f1', hue='mapping',
                      col="pretrained", col_order=['None', 'glove'],
                      hue_order=['alignment', 'incremental'], palette='BuGn', s=500,
                      data=df.loc[(df['t'] == '1.0') & (df['language'] == language)])
    plt.xlabel('', size=14, family='monospace')
    plt.ylabel('F1-Score', size=14, family='monospace')
    plt.xticks(rotation=45)
    pre.fig.subplots_adjust(top=0.9)
    pre.set_xlabels('', fontsize=14, family='monospace')  # not set_label
    pre.set_ylabels('F1-Score', fontsize=14, family='monospace')
    pre.fig.suptitle('Initial vs Pretrained')
    plt.savefig(os.path.join(my_path,dataset_id,my_file))

    # Incremental vs Alignment (across - pretrained,algorithm)
    my_file = '../AlignmentvsIncremental_across_semeval.png'
    sns.set()
    sns.boxplot(x='mapping', y='f1', saturation=1, palette='BuGn',
                data=df.loc[(df['t'] == '1.0') & (df['language'] == language)])
    # specify axis labels
    plt.xlabel('', size=14, family='monospace')
    plt.ylabel('F1-Score', size=14, family='monospace')
    plt.title('Alignment vs Incremental (across)')
    plt.savefig(os.path.join(my_path,dataset_id,my_file))

    sns.set()
    pree = sns.catplot(x='t', y='f1', hue='pretrained', order=['1.0', '1.5', '2.0', '2.5', '3.0'], col="mapping",
                       hue_order=['None', 'glove'],
                       col_order=['alignment', 'incremental'],
                       palette='BuGn', kind="box", saturation=1,
                       data=df.loc[(df['t'] == '1.0') & (df['language'] == language)])
    pree.fig.subplots_adjust(top=0.9)
    pree.set_xlabels('threshold multiplier', fontsize=14, family='monospace')  # not set_label
    pree.set_ylabels('F1-Score', fontsize=14, family='monospace')
    pree.fig.suptitle('F1 score per threshold')

    sns.set()
    sns.catplot(x='window_size', y='f1', hue='w2vec_algorithm', palette='BuGn',
                col="mapping", col_order=['alignment', 'incremental'], kind="box", saturation=1,
                data=df.loc[(df['t'] == '1.0') & (df['language'] == language)])

    sns.set()
    sns.catplot(x='window_size', y='f1', hue='pretrained', col="mapping",
                col_order=['alignment', 'incremental'], palette='BuGn', kind="box", saturation=1,
                data=df.loc[(df['t'] == '1.0') & (df['language'] == language)])

    logging.info("--- %s seconds ---" % (time.time() - start_time))
    print("")


if __name__ == '__main__':
    main()



