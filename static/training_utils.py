import re

def get_file_path(file_path):
    file_path = file_path.rsplit('/', 1)[0]
    return file_path


def get_file_name(file_path):
    file_name = file_path.rsplit('/', 1)[1]
    return file_name


def get_full_corpus_model (path_output):
    file_full_path = None
    # convention: same name with relevant variable in reproduce_experiment script
    full_corpus_model = 'corpus_concat'
    file_path = get_file_path(path_output)
    file_name = get_file_name(path_output)

    if file_name != full_corpus_model:
        file_full_path = file_path + "/" + full_corpus_model

    return file_full_path


def get_file_prev_version(path_output):

    file_path = get_file_path(path_output)
    file_name = get_file_name(path_output)

    previous_version = int([float(n) for n in re.findall(r'-?\d+\.?\d*', file_name)][-1] - 1)
    previous_file_name = file_name[::-1].replace(str(previous_version + 1)[::-1], str(previous_version)[::-1], 1)[::-1]
    file_full_path = file_path + "/" + previous_file_name

    return previous_version, file_full_path

def retrieve_embeddings_to_load(pretrained_matrix, pretrained_matrix_path, dim, apply_full_path, file_full_path):
    embeddings_to_load = ''

    if apply_full_path:
        # beforehand knowledge of the model
        embeddings_to_load = file_full_path
    elif pretrained_matrix == 'glove':
        embeddings_to_load = pretrained_matrix_path + "/glove.6B." + str(dim) + "d.txt"
    elif pretrained_matrix == 'dewiki':
        embeddings_to_load = pretrained_matrix_path + "/data.txt"
    elif pretrained_matrix == 'latconll17':
        embeddings_to_load = pretrained_matrix_path + "/model.txt"
    elif pretrained_matrix == 'sweconll17':
        embeddings_to_load = pretrained_matrix_path + "/model.txt"

    return embeddings_to_load