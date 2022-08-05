import pandas as pd
import glob
import matplotlib.pyplot as plt
import seaborn as sns

language = 'en'
embed='None'
plot=True


df = pd.DataFrame()

if plot:
    path = './output/**/**/results'
    text_files = set(glob.glob(path + "/**/*.pkl", recursive=True))
    for file in text_files:
        df = pd.concat([df, pd.read_pickle(file)], ignore_index=True, axis=0)

    #df.to_csv('language_drift_results_4.0.1', index=False)

    sns.set()
    pre = sns.relplot(x='w2vec_algorithm', y='f1_ln', hue='mapping',
                      col="pretrained", col_order=['None',embed],
                      hue_order=['procrustes'], palette='BuGn', s=500,
                      data=df.loc[(df['language'] == language)])
    plt.xlabel('', size=14, family='monospace')
    plt.ylabel('F1-Score', size=14, family='monospace')
    plt.xticks(rotation=45)
    pre.fig.subplots_adjust(top=0.9)
    pre.set_xlabels('', fontsize=14, family='monospace')  # not set_label
    pre.set_ylabels('F1-Score', fontsize=14, family='monospace')
    pre.fig.suptitle('Initial vs Pretrained')
    plt.show()


    sns.set()
    sns.boxplot(x='mapping', y='f1_ln', saturation=1, palette='BuGn', data=df.loc[ (df['language'] == language)])
    # specify axis labels
    plt.xlabel('', size=14, family='monospace')
    plt.ylabel('F1-Score', size=14, family='monospace')
    plt.title('Procrustes vs Incremental (across)')
    plt.show()

    sns.set()
    pree = sns.catplot(x='t', y='f1_ln', hue='pretrained', order=['1.0'], col="mapping",
                       hue_order=['None', embed],
                       col_order=['procrustes','twec','incremental'],
                       palette='BuGn', kind="box", saturation=1, data=df.loc[(df['language'] == language)])
    pree.fig.subplots_adjust(top=0.9)
    pree.set_xlabels('threshold multiplier', fontsize=14, family='monospace')  # not set_label
    pree.set_ylabels('F1-Score', fontsize=14, family='monospace')
    pree.fig.suptitle('F1 score per threshold')
    plt.show()

    sns.set()
    sns.catplot(x='window_size', y='f1_ln', hue='w2vec_algorithm', palette='BuGn',
                col="mapping", col_order=['procrustes','twec','incremental'], kind="box", saturation=1, data=df.loc[(df['language'] == language)])
    plt.show()

    sns.set()
    sns.catplot(x='window_size', y='f1_ln', hue='pretrained', col="mapping", col_order=['procrustes','twec','incremental'],
                palette='BuGn', kind="box", saturation=1, data=df.loc[(df['language'] == language)])
    plt.show()
else:
    path = './output/lat_4.0.0/**/distances'
    text_files = set(glob.glob(path + "/distances_intersection.tsv", recursive=True))
    for file in text_files:
        df = pd.read_csv(file, sep='\t')
        df.hist()

