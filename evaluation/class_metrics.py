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
                         <pretrained> <window_size> <dim> <t> <data_set_id> <language>

        <path_truth>            = path to binary gold data (tab-separated)
        <path_file>             = path to file containing words and binary values (tab-separated)
        <path_output>           = csv file to save performance results
        <mapping>               = indicates type of embeddings mapping
        <w2vec_algorithm>       = word2vec algorithm - sgns/cbow
        <pretrained>            = option of pretrained embeddings (none, glove)
        <window_size>           = the linear distance of context words to consider in each direction
        <dim>                   = dimensionality of embeddings
        <t>                     = threshold = mean + t * standard error"
        <data_set_id>           = data set identifier
        <language>              = dataset language

    """)

    path_truth = args['<path_truth>']
    path_file = args['<path_file>']
    path_output = args['<path_output>']
    mapping = args['<mapping>']
    w2vec_algorithm = args['<w2vec_algorithm>']
    pretrained = args['<pretrained>']
    window_size = args['<window_size>']
    dim = args['<dim>']
    t = args['<t>']
    data_set_id = args['<data_set_id>']
    language = args['<language>']

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    logging.info(__file__.upper())

    start_time = time.time()

    # Load gold data
    truth = []
    with open(path_truth, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            truth.append(int(row[1]))

    # Load predictions
    predictions = []
    with open(path_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        reader.__next__()
        for row in reader:
            predictions.append(int(row[1]))

    # Compute metrics
    precision = metrics.precision_score(truth, predictions, zero_division=0)
    recall = metrics.recall_score(truth, predictions, zero_division=0)
    accuracy = metrics.accuracy_score(truth, predictions)
    f1 = metrics.f1_score(truth, predictions, zero_division=0)

    precision = round(precision, 3)
    recall = round(recall, 3)
    f1 = round(f1, 3)

    df = pd.DataFrame({'data_set_id':data_set_id,'w2vec_algorithm': w2vec_algorithm, 'pretrained': pretrained,
                       'mapping': mapping, 'dim': dim, 'window_size': window_size,
                       't': t, 'f1': f1, 'accuracy':accuracy, 'recall': recall,
                       'precision': precision, 't': t, 'language': language},index=[0])
    df.to_pickle(path_output)


    logging.info("--- %s seconds ---" % (time.time() - start_time))    
    print("")


if __name__ == '__main__':
    main()
