import numpy as np


def make_label_mapping_dict(data):
    labels = list(set(data['cluster_label'].tolist()))
    celltype_to_int = {name:i for i, name in enumerate(labels)}
    return celltype_to_int


def split_X_y(data, label_name='cluster_label'):
    columns = list(data.columns)
    # print(columns)
    columns.remove('cluster_label')
    print(columns)
    X = data[columns]

    celltype_to_int = make_label_mapping_dict(data)
    y = np.asarray([celltype_to_int[type] for type in data['cluster_label'].tolist()])
    return X, y