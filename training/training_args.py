from absl import flags
FLAGS = flags.FLAGS

# Settings
flags.DEFINE_string(name="output_dir", default="./finetune_model", help="")
flags.DEFINE_boolean(name="overwrite_output_dir", default=True, help="")
flags.DEFINE_string(name="run_name", default="default_run", help="")
flags.DEFINE_integer(name="seed", default=0, help="Global random seed.")
flags.DEFINE_string(
    name="labelsName", default="measured log solubility in mols per litre", help="")

# Model params
flags.DEFINE_string(
    name="pretrained_model_name_or_path",
    default="./DeepChem/ChemBERTa-10M-MTR",
    help="Arg to HuggingFace model.from_pretrained(). Can be either a path to a local model or a model ID on HuggingFace Model Hub. If not given, trains a fresh model from scratch (non-pretrained).",
)
flags.DEFINE_boolean(
    name="freeze_base_model",
    default=False,
    help="If True, freezes the parameters of the base model during training. Only the classification/regression head parameters will be trained. (Only used when `pretrained_model_name_or_path` is given.)",
)
flags.DEFINE_boolean(
    name="is_molnet",
    default=False,
    help="If true, assumes all dataset are MolNet datasets.",
)

# RobertaConfig params (only for non-pretrained models)
flags.DEFINE_integer(name="vocab_size", default=600, help="")
flags.DEFINE_integer(name="max_position_embeddings", default=515, help="")
flags.DEFINE_integer(name="num_attention_heads", default=6, help="")
flags.DEFINE_integer(name="num_hidden_layers", default=6, help="")
flags.DEFINE_integer(name="type_vocab_size", default=1, help="")

# Train params
flags.DEFINE_integer(name="logging_steps", default=50, help="")
flags.DEFINE_integer(name="early_stopping_patience", default=3, help="")
flags.DEFINE_integer(name="num_train_epochs_max", default=10, help="")
flags.DEFINE_integer(name="per_device_train_batch_size", default=64, help="")
flags.DEFINE_integer(name="per_device_eval_batch_size", default=64, help="")
flags.DEFINE_integer(name="eval_steps", default=100, help="")


# model type
flags.DEFINE_string(name="model_type", default="baseline", help="")



# dual attention part
flags.DEFINE_boolean(
    name="is_dual",
    default=True,
    help="If true, using dual gate attention.",
)
flags.DEFINE_boolean(
    name="alpha",
    default=True,
    help="If true, using alpha gate.",
)
flags.DEFINE_boolean(
    name="beta",
    default=True,
    help="If true, using beta gate.",
)
flags.DEFINE_string(
    name="structure_column", default="scaffold", help="using structure info."
)

flags.DEFINE_integer(
    name="n_trials",
    default=5,
    help="Number of different hyperparameter combinations to try. Each combination will result in a different finetuned model.",
)
flags.DEFINE_integer(
    name="n_seeds",
    default=5,
    help="Number of unique random seeds to try. This only applies to the final best model selected after hyperparameter tuning.",
)

# Dataset params
flags.DEFINE_list(
    name="datasets",
    default="./myData/finetuneData",
    help="Comma-separated list of MoleculeNet dataset names.",
)
flags.DEFINE_string(
    name="split", default="scaffold", help="DeepChem data loader split_type."
)
flags.DEFINE_list(
    name="dataset_types",
    default="classification",
    help="List of dataset types (ex: classification,regression). Include 1 per dataset, not necessary for MoleculeNet datasets.",
)

# Tokenizer params
flags.DEFINE_string(
    name="tokenizer_path",
    default="seyonec/SMILES_tokenized_PubChem_shard00_160k",
    help="",
)
flags.DEFINE_integer(name="max_tokenizer_len", default=512, help="")
