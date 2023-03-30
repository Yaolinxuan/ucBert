from collections import OrderedDict
from transformers import RobertaConfig
import torch
import os
from training.models_with_cif_features import CifRobertaForSequenceClassification,CifRobertaForRegression
from training.baseline_models import RobertaForRegression,RobertaForSequenceClassification

def load_model_config(FLAGS):
        
    if FLAGS.pretrained_model_name_or_path:
        config = RobertaConfig.from_pretrained(
            FLAGS.pretrained_model_name_or_path, use_auth_token=False
        )
    else:
        config = RobertaConfig(
            vocab_size=FLAGS.vocab_size,
            max_position_embeddings=FLAGS.max_position_embeddings,
            num_attention_heads=FLAGS.num_attention_heads,
            num_hidden_layers=FLAGS.num_hidden_layers,
            type_vocab_size=FLAGS.type_vocab_size,
            is_gpu=torch.cuda.is_available(),
        )
    return config
            

def prune_state_dict(model_dir):
    """Remove problematic keys from state dictionary"""
    if not (model_dir and os.path.exists(os.path.join(model_dir, "pytorch_model.bin"))):
        return None

    state_dict_path = os.path.join(model_dir, "pytorch_model.bin")
    assert os.path.exists(
        state_dict_path
    ), f"No `pytorch_model.bin` file found in {model_dir}"
    loaded_state_dict = torch.load(state_dict_path)
    state_keys = loaded_state_dict.keys()
    keys_to_remove = [
        k for k in state_keys if k.startswith("regression") or k.startswith("norm")
    ]

    new_state_dict = OrderedDict({**loaded_state_dict})
    for k in keys_to_remove:
        del new_state_dict[k]
    return new_state_dict


def load_model(FLAGS,config,dataset_type = "classification",model_type = "baseline"):
    if model_type == "baseline":
        if dataset_type == "classification":
            model_class = RobertaForSequenceClassification
        elif dataset_type == "regression":
            model_class = RobertaForRegression
    elif model_type  == "DualGate":
        if dataset_type == "classification":
            model_class = CifRobertaForSequenceClassification
        elif dataset_type == "regression":
            model_class = CifRobertaForRegression
    else:
        raise Exception('Model type is unknown.')
    
    state_dict = prune_state_dict(FLAGS.pretrained_model_name_or_path)
    if FLAGS.pretrained_model_name_or_path:
        model = model_class.from_pretrained(
            FLAGS.pretrained_model_name_or_path,
            config=config,
            state_dict=state_dict,
            use_auth_token=True,
        )
        if FLAGS.freeze_base_model:
            for name, param in model.base_model.named_parameters():
                param.requires_grad = False
    else:
        model = model_class(config=config)

    return model