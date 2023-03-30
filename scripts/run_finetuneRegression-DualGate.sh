python training/finetune.py \
    --datasets=./myData/regression \
    --pretrained_model_name_or_path=./DeepChem/ChemBERTa-10M-MTR \
    --dataset_types=regression \
    --is_molnet=False \
    --output_dir=./finetune_model \
    --labelsName=band_gap \
    --model_type=DualGate