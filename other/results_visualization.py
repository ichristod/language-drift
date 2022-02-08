import glob
import logging
import time
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from docopt import docopt


def main():
    """Compute precission, recall, f_1 and f_0.5.
    """

    # Get the argument
    args = docopt("""Create results visualizations

    Usage:
        results_visualization.py <path> <path_output> 

        <path>              = root path of experiments' results

    """)

    path = args['<path>']
    w2vec_algorithm  = args['<w2vec_algorithm>']
    mapping  = args['<mapping>']

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    logging.info(__file__.upper())
    start_time = time.time()

    text_files = glob.glob(path + "/**/*.pkl", recursive=True)
    df = pd.DataFrame()

    for file in text_files:
        df = pd.concat([df, pd.read_pickle(file)], ignore_index=True, axis=0)

    # Initialization & OP Alignment
    # for not pretrained and all thresholds
    sns.set()
    sns.boxplot(x='w2vec_algorithm', y='f1', order=['cbow', 'sgns'], width=0.5, saturation=1,
                color=sns.color_palette('BuGn')[1],
                data=df.loc[(df['pretrained'] == 'None') & (df['mapping'] == 'alignment')])
    plt.xlabel('', size=14, family='monospace')
    plt.ylabel('F1-Score', size=14, family='monospace')
    plt.xticks(rotation=45)
    plt.title('Initialization & OP Alignment', size=14, family='monospace')
    plt.savefig('Init&Alignment_init.png')

    # Alignment vs Incremental
    # for not pretrained and all thresholds
    sns.set()
    sns.boxplot(x='w2vec_algorithm', y='f1', hue='mapping', width=0.6, saturation=1,
                hue_order=['alignment', 'incremental'], order=['cbow', 'sgns'], palette='BuGn',
                data=df.loc[df['pretrained'] == 'None'])
    plt.xlabel('', size=14, family='monospace')
    plt.ylabel('F1-Score', size=14, family='monospace')
    plt.xticks(rotation=45)
    plt.title('Alignment vs Incremental (init)', size=14, family='monospace')
    plt.savefig('AlignmentvsIncremental_init.png')

    # Alignment vs Incremental (Pretrained Embeddings)
    sns.set()
    pre = sns.catplot(x='w2vec_algorithm', y='f1', hue='mapping',
                      col="pretrained", col_order=['None', 'glove'], saturation=1, kind='box',
                      hue_order=['alignment', 'incremental'], order=['cbow', 'sgns'], palette='BuGn',
                      data=df)
    plt.xlabel('', size=14, family='monospace')
    plt.ylabel('F1-Score', size=14, family='monospace')
    plt.xticks(rotation=45)
    pre.fig.subplots_adjust(top=0.9)
    pre.set_xlabels('', fontsize=14, family='monospace')  # not set_label
    pre.set_ylabels('F1-Score', fontsize=14, family='monospace')
    pre.fig.suptitle('Initial vs Pretrained')
    plt.savefig('InitialvsPretrained.png')


    logging.info("--- %s seconds ---" % (time.time() - start_time))




if __name__ == '__main__':
    main()
