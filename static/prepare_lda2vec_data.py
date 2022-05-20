import argparse
import gzip
import json
import os.path

from training_utils import get_file_path

def main():
    """
    Create datasets for the LDA2VEC implementation
    """

    # Get the arguments
    parser = argparse.ArgumentParser(
        description='Create json files of [[idx_1,"sentence_1"], ["idx_2,sentence_2"]]')

    parser.add_argument('path_corpus1', help='path to corpus1 directory with zipped files')
    parser.add_argument('path_corpus2', help='path to corpus2 directory with zipped files')

    args = parser.parse_args()
    path_corpus1 = args.path_corpus1
    path_corpus2 = args.path_corpus2

    path_list=[path_corpus1,path_corpus2]
    print("path_list: ",path_list)
    docs=[]

    for path in path_list:
        with gzip.open(path,'rt',encoding="utf-8") as corpus_docids:

            # "sentence_1" \n,"sentence_2" \n -> ["sentence_1","sentence_2"]
            for sentence in corpus_docids:
                docs.append(sentence.split("\n")[0])

            # remove one letter words
            for idx, element in enumerate(docs):
                docs[idx]  = " ".join([value for value in str(element).split(" ") if len(value) > 2])

            # ["sentence_1","sentence_2"] -> [["sentence_1"],["sentence_2"]]
            list_results = [[element] for element in docs if len(str(element).split(" ")) > 11]
            #list_results = list_results[:int(len(list_results) / 1000)]

            print("Number of sentences in ",path," is: ",len(list_results))
            #print("list_results ",list_results[:2])

            for counter,value in enumerate(list_results):
                # [["sentence_1"],["sentence_2"]] -> [[idx_1,"sentence_1"],[idx_2,"sentence_2"]]
                list_results[counter].insert(0,counter)
            print(list_results[:2])
            print("max_sentence: ", max([len(sublist[-1]) for sublist in list_results]))
            print("min_sentence: ", min([len(sublist[-1]) for sublist in list_results]))
        with open(os.path.join(get_file_path(path),'lemma_docids.json'), 'w') as f:
            json.dump(list_results, f)


if __name__ == '__main__':
    main()