import os

def get_dataset_info(FLAGS):
    dataset_name_or_path = FLAGS.datasets[0]
    return dataset_name_or_path,os.path.splitext(os.path.basename(dataset_name_or_path))[0], FLAGS.dataset_types[0]

def __dataset_check(dataset_folder):
    """检查数据集

    Args:
        dataset_folder (_type_): 数据集的文件夹
    """    
    assert os.path.exists(os.path.join(dataset_folder, "train.csv"))
    assert os.path.exists(os.path.join(dataset_folder, "valid.csv"))
    assert os.path.exists(os.path.join(dataset_folder, "test.csv"))
def dataset_check(datasets):
    print("Assuming each dataset is a folder containing CSVs...")

    for dataset_folder in datasets:
        __dataset_check(dataset_folder)




