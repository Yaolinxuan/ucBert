python training/finetune.py  \
    --datasets=./myData/classification \
    --pretrained_model_name_or_path=./DeepChem/ChemBERTa-10M-MTR \
    --dataset_types=classification \
    --is_molnet=False \
    --output_dir=./finetune_model \
    --labelsName=crystalSystem \
    --model_type=baseline