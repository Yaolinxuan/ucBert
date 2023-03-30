import os
import torch
import sys
sys.path.append(".")
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from absl import app
from scipy.special import softmax
from scipy.stats import pearsonr
from sklearn.metrics import (
    average_precision_score,
    matthews_corrcoef,
    mean_squared_error,
    roc_auc_score,
)

from transformers import RobertaTokenizerFast, Trainer, TrainingArguments,EvalPrediction
from transformers.trainer_callback import EarlyStoppingCallback
from training.config import train_params
from training.dataset_loader import get_finetune_datasets
from training.utils import get_dataset_info,dataset_check
from training.training_args import FLAGS
from training.model_utils import load_model_config,load_model
from training.compute_metric import compute_classification_metrics,compute_regression_metrics
import tensorboard


def main(argv):
    torch.manual_seed(FLAGS.seed)
    dataset_check(FLAGS.datasets)
    
    dataset_name_or_path,dataset_name,dataset_type = get_dataset_info(FLAGS)
    
    run_dir = os.path.join(FLAGS.output_dir, FLAGS.model_type, dataset_type)
    
    if os.path.exists(run_dir) and not FLAGS.overwrite_output_dir:
        print(f"Run dir already exists for dataset: {dataset_name}")
    else:
        print(f"Finetuning on {dataset_name}")
        finetune_chemberta(
            dataset_name_or_path, dataset_type, run_dir, is_molnet,FLAGS.model_type
        )


def finetune_chemberta(dataset_name, dataset_type, run_dir, is_molnet,model_type):
    """模型微调

    Args:
        dataset_name (_type_): 数据集
        dataset_type (_type_): _description_
        run_dir (_type_): _description_
        is_molnet (bool): _description_
        model_type (_type_): 模型的类型 baseline,DualGate...
    """    
    tokenizer = RobertaTokenizerFast.from_pretrained(
        FLAGS.tokenizer_path, max_len=FLAGS.max_tokenizer_len, use_auth_token=False
    )

    finetune_datasets = get_finetune_datasets(
        dataset_name, tokenizer, is_molnet, dataset_type)
    config = load_model_config(FLAGS)

    
    # Add new param
    config.is_dual = FLAGS.is_dual
    config.alpha = FLAGS.alpha
    config.beta = FLAGS.beta
    if dataset_type == "classification":
        config.num_labels = finetune_datasets.num_labels
        metric_for_best_model = "weighted avg_f1-score"

    elif dataset_type == "regression":
        config.num_labels = 1
        config.norm_mean = finetune_datasets.norm_mean
        config.norm_std = finetune_datasets.norm_std
        metric_for_best_model = "rmse"
    
    config.structure_column = FLAGS.structure_column

    compute_metrics_map = {
        "classification":compute_classification_metrics,
        "regression":compute_regression_metrics
    }

    model = load_model(FLAGS,config,dataset_type =  dataset_type,model_type = model_type)
    
    training_args = TrainingArguments(
        evaluation_strategy="steps",
        output_dir=run_dir,
        overwrite_output_dir=FLAGS.overwrite_output_dir,
        per_device_eval_batch_size=FLAGS.per_device_eval_batch_size,
        logging_steps=FLAGS.logging_steps,
        load_best_model_at_end=True,
        report_to=["tensorboard"],
        eval_steps=FLAGS.eval_steps,
        save_total_limit=5,
        metric_for_best_model = metric_for_best_model,
        seed = random_seed,
        **train_params
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=finetune_datasets.train_dataset,
        eval_dataset=finetune_datasets.valid_dataset,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=FLAGS.early_stopping_patience)
        ],
        compute_metrics=compute_metrics_map[dataset_type]
    )

    trainer.train()
    # predictions = trainer.predict(finetune_datasets.test_dataset)
    # labels = finetune_datasets.test_dataset.labels

    # print(compute_metrics_map[dataset_type](EvalPrediction(predictions=predictions,label_ids=labels)))
    




if __name__ == "__main__":
    from training.training_args import FLAGS
    is_molnet = False
    random_seed = 1
    app.run(main)
