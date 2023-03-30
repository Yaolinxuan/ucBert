import torch
import json
import shutil
from collections import OrderedDict
from dataclasses import dataclass
from glob import glob
from typing import List
from training.training_args import FLAGS
import pandas as pd
import numpy as np
from chemberta.utils.molnet_dataloader import get_dataset_info, load_molnet_dataset
import os

@dataclass
class FinetuneDatasets:
    train_dataset: str
    valid_dataset: torch.utils.data.Dataset
    valid_dataset_unlabeled: torch.utils.data.Dataset
    test_dataset: torch.utils.data.Dataset
    num_labels: int
    norm_mean: List[float]
    norm_std: List[float]


class FinetuneDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, dataset_type, include_labels=True):

        self.encodings = tokenizer(df["smiles"].tolist(), truncation=True, padding=True)
        # self.labels = df["crystalSystem"].tolist()
        self.labels = df[FLAGS.labelsName].tolist()
        self.include_labels = include_labels
        
        # more column for classification and regression
        # print("this is ", dataset_type)
        if dataset_type == 'classification': # a,b,c,alpha,beta,gama
            FLAGS.structure_column = 'a,b,c,alpha,beta,gamma'
            self.feature_list = FLAGS.structure_column.split(',')
            raw_feature = []
            for fea in self.feature_list:
                raw_feature.append(df[fea].tolist())
            self.feature = np.array(raw_feature)
        elif dataset_type == 'regression': # a,b,c,alpha,beta,gama
            # FLAGS.structure_column = 'a,b,c,alpha,beta,gamma,band_gap,energy_per_atom,e_total,is_stable,formation_energy_per_atom,energy_above_hull,volume,density,is_magnetic'
            FLAGS.structure_column = 'a,b,c,alpha,beta,gamma'
            self.feature_list = FLAGS.structure_column.split(',')
            raw_feature = []
            for fea in self.feature_list[:-1]:
                raw_feature.append(df[fea].tolist())
            is_magnetic = [1 if it=='TRUE' else 0 for it in df[self.feature_list[-1]].tolist()]
            raw_feature.append(is_magnetic)
            self.feature = np.array(raw_feature)
        else:
            print("please assign 'classification' or 'regression'")
            exit(0)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.include_labels and self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx])
        item['feature'] = torch.tensor(self.feature[:,idx], dtype=torch.float)
        # print(item)
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])


def get_finetune_datasets(dataset_name, tokenizer, is_molnet, dataset_type):
    """加载数据集

    Args:
        dataset_name (_type_): _description_
        tokenizer (_type_): _description_
        is_molnet (bool): _description_
        dataset_type (_type_): _description_

    Returns:
        _type_: _description_
    """    
    if is_molnet:
        tasks, (train_df, valid_df, test_df), _ = load_molnet_dataset(
            dataset_name, split=FLAGS.split, df_format="chemprop"
        )
        assert len(tasks) == 1
    else:
        train_df = pd.read_csv(os.path.join(dataset_name, "train.csv"))
        valid_df = pd.read_csv(os.path.join(dataset_name, "valid.csv"))
        test_df = pd.read_csv(os.path.join(dataset_name, "test.csv"))
    print("load data from {}".format(dataset_name))
    
    train_dataset = FinetuneDataset(train_df, tokenizer, dataset_type)
    valid_dataset = FinetuneDataset(valid_df, tokenizer, dataset_type)
    valid_dataset_unlabeled = FinetuneDataset(valid_df, tokenizer, dataset_type, include_labels=False)
    test_dataset = FinetuneDataset(test_df, tokenizer, dataset_type, include_labels=False)

    num_labels = len(np.unique(train_dataset.labels))
    print("num labels",num_labels)
    norm_mean = [np.mean(np.array(train_dataset.labels), axis=0)]
    norm_std = [np.std(np.array(train_dataset.labels), axis=0)]

    return FinetuneDatasets(
        train_dataset,
        valid_dataset,
        valid_dataset_unlabeled,
        test_dataset,
        num_labels,
        norm_mean,
        norm_std,
    )