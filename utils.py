import anndata
import os
from random import shuffle
import matplotlib
import numpy as np
import scanpy as sc
import sklearn as sk
import wget
from scipy import sparse
from sklearn import preprocessing


using = "hpoly"

def get_key(data_name="pbmc"):
    stim_key , ctrl_key, cell_type_key = None, None,None
    if data_name == "pbmc":
        stim_key = "stimulated"
        ctrl_key = "control"
        cell_type_key = "cell_type"
    elif data_name == "hpoly":
        stim_key = "Hpoly.Day10"
        ctrl_key = "Control"
        cell_type_key = "cell_label"
    elif data_name == "salmonella":
        stim_key = "Salmonella"
        ctrl_key = "Control"
        cell_type_key = "cell_label"
    elif data_name == "species":
        stim_key = "LPS6"
        ctrl_key = "unst"
        cell_type_key = "species"
    return stim_key , ctrl_key, cell_type_key

# 对每个单元类型进行重采样，以使每个单元类型具有相同的数量，以解决类不平衡问题
def balancer(adata, cell_type_key="cell_type", condition_key="condition"):
    class_names = np.unique(adata.obs[cell_type_key])
    class_pop = {}
    for cls in class_names:
        class_pop[cls] = adata.copy()[adata.obs[cell_type_key] == cls].shape[0]
    max_number = np.max(list(class_pop.values()))

    all_data_x = []
    all_data_label = []
    all_data_condition = []
    for cls in class_names:
        temp = adata.copy()[adata.obs[cell_type_key] == cls]
        index = np.random.choice(range(len(temp)), max_number)
        if sparse.issparse(temp.X):
            temp_x = temp.X.A[index]
        else:
            temp_x = temp.X[index]
        all_data_x.append(temp_x)
        temp_ct = np.repeat(cls, max_number)
        all_data_label.append(temp_ct)
        temp_cc = np.repeat(np.unique(temp.obs[condition_key]), max_number)
        all_data_condition.append(temp_cc)
    balanced_data = anndata.AnnData(np.concatenate(all_data_x))
    balanced_data.obs[cell_type_key] = np.concatenate(all_data_label)
    balanced_data.obs[condition_key] = np.concatenate(all_data_label)
    class_names = np.unique(balanced_data.obs[cell_type_key])
    class_pop = {}
    for cls in class_names:
        class_pop[cls] = len(balanced_data[balanced_data.obs[cell_type_key] == cls])
    return balanced_data


def extractor(data, cell_type, conditions, cell_type_key="cell_type", condition_key="condition"):
    cell_with_both_condition = data[data.obs[cell_type_key] == cell_type]
    condtion_1 = data[(data.obs[cell_type_key] == cell_type) & (data.obs[condition_key] == conditions["ctrl"])]

    condtion_2 = data[(data.obs[cell_type_key] == cell_type) & (data.obs[condition_key] == conditions["stim"])]
    training = data[~((data.obs[cell_type_key] == cell_type) & (data.obs[condition_key] == conditions["stim"]))]
    return [training, condtion_1, condtion_2, cell_with_both_condition]


def shuffle_data(adata, labels=None):
    ind_list = [i for i in range(adata.shape[0])]
    shuffle(ind_list)
    if sparse.issparse(adata.X):
        x = adata.X.A[ind_list, :]
    else:
        x = adata.X[ind_list, :]
    if labels is not None:
        labels = labels[ind_list]
        adata = anndata.AnnData(x, obs={"labels": list(labels)})
        return adata, labels
    else:
        return anndata.AnnData(x, obs=adata.obs)


def label_encoder(adata):
    le = preprocessing.LabelEncoder()
    labels = le.fit_transform(adata.obs["condition"].tolist())
    return labels.reshape(-1, 1), le