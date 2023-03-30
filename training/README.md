
model_type 有两个取值：baseline,DualGate

* baseline 对应 training/baseline_models.py 中的模型
* DualGate 对应 training/models_with_cif_features.py 中的模型



```sh

python training/finetune.py \
    --datasets=./myData/regression \
    --pretrained_model_name_or_path=./DeepChem/ChemBERTa-10M-MTR \
    --dataset_types=regression \
    --is_molnet=False \
    --output_dir=./finetune_model \
    --labelsName=band_gap \
    --model_type=baseline

```


模型的选择主要是通过 training/model_utils.py 中的load_model 函数来实现

# Todo



* 特征的选择: 'a,b,c,alpha,beta,gamma,band_gap,energy_per_atom,e_total,is_stable,formation_energy_per_atom,energy_above_hull,volume,density,is_magnetic'，特别是回归任务中特征的选择

* 回归问题中的y 一般数值变化范围较小，是否需要转为log(y)

* 调参（学习率,批次大小）

* 数据集不平衡，（主要是分类问题中）

* 结果可视化（tensorboard已经画了一些，主要是训练过程中loss和metrics 的变化）

* 可解释分析