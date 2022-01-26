import csv
import logging
import sys
import time
import pandas as pd

from docopt import docopt
import numpy as np 
from sklearn import metrics

def main():
    """Compute precission, recall, f_1 and f_0.5.
    """

    # Get the argument 
    args = docopt("""Create classification report

    Usage:
        class_metrics.py <path_truth> <path_file>  <path_output> 

        <path_truth>    = path to binary gold data (tab-separated)
        <path_file>     = path to file containing words and binary values (tab-separated)
        <path_output>   = csv file to save performance results

    """)

    path_truth = args['<path_truth>']
    path_file = args['<path_file>']
    path_output = args['<path_output>']

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
        for row in reader:
            predictions.append(int(row[1]))

    clsf_report = pd.DataFrame(metrics.classification_report(y_true=truth, y_pred=predictions,
                                                             output_dict=True)).transpose()
    clsf_report.to_csv(path_output, index=True)

    logging.info("--- %s seconds ---" % (time.time() - start_time))    
    print("")


if __name__ == '__main__':
    main()
