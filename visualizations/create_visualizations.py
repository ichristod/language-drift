import pandas as pd
import glob
import matplotlib.pyplot as plt
import seaborn as sns

language = 'en'
embed='glove'
path = './results/**/**/classification'


text_files = set(glob.glob(path + "/**/*.pkl", recursive=True))

df = pd.DataFrame()

for file in text_files:
 df = pd.concat([df, pd.read_pickle(file)], ignore_index=True, axis=0)

df.to_csv('language_drift_results', index=False)

sns.set()
pre = sns.relplot(x='w2vec_algorithm', y='f1', hue='mapping',
                  col="pretrained", col_order=['None', embed],
                  hue_order=['alignment', 'incremental'], palette='BuGn', s=500,
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
sns.boxplot(x='mapping', y='f1', saturation=1, palette='BuGn', data=df.loc[ (df['language'] == language)])
# specify axis labels
plt.xlabel('', size=14, family='monospace')
plt.ylabel('F1-Score', size=14, family='monospace')
plt.title('Alignment vs Incremental (across)')
plt.show()

sns.set()
pree = sns.catplot(x='t', y='f1', hue='pretrained', order=['1.0', '1.5', '2.0'], col="mapping",
                   hue_order=['None', embed],
                   col_order=['alignment', 'incremental'],
                   palette='BuGn', kind="box", saturation=1, data=df.loc[(df['language'] == language)])
pree.fig.subplots_adjust(top=0.9)
pree.set_xlabels('threshold multiplier', fontsize=14, family='monospace')  # not set_label
pree.set_ylabels('F1-Score', fontsize=14, family='monospace')
pree.fig.suptitle('F1 score per threshold')
plt.show()

sns.set()
sns.catplot(x='window_size', y='f1', hue='w2vec_algorithm', palette='BuGn',
            col="mapping", col_order=['alignment', 'incremental'], kind="box", saturation=1, data=df.loc[(df['language'] == language)])
plt.show()

sns.set()
sns.catplot(x='window_size', y='f1', hue='pretrained', col="mapping", col_order=['alignment', 'incremental'],
            palette='BuGn', kind="box", saturation=1, data=df.loc[(df['language'] == language)])
plt.show()
