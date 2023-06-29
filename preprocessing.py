import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def make_label_mapping_dict(data):
    labels = list(set(data['cluster_label'].tolist()))
    celltype_to_int = {name: i for i, name in enumerate(labels)}
    return celltype_to_int


def split_X_y(data, label_name='cluster_label'):
    columns = list(data.columns)
    # print(columns)
    columns.remove(label_name)
    X = data[columns].copy()

    celltype_to_int = make_label_mapping_dict(data)
    y = np.asarray([celltype_to_int[type] for type in data['cluster_label'].tolist()])
    return X, y



def create_train_test_splits(X, y):
    splits = []
    for i in range(5):
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=(i+5)*5, test_size=0.3, stratify=y, shuffle=True)
        splits.append((X_train, X_test, y_train, y_test))

    return splits


def choose_split(splits, split_index):

    X, X_test, y, y_test = splits[split_index]
    # reset indices
    X = X.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)

    y = y.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    y = y.to_numpy()  
    y_test = y_test.to_numpy()  

    print(X.shape)
    print(X_test.shape)
    print(y.shape)
    print(y_test.shape)

    return X, X_test, y, y_test




def ncv_results_to_df(results):
    data = []
    for classifier, df in results.items():
        for metric in df.columns:
            for outer_loop, value in enumerate(df[metric], start=1):
                data.append([classifier, metric, outer_loop, value])
    results_df = pd.DataFrame(data, columns=['Classifier', 'Metric', 'Outer Loop', 'Value'])
    return results_df