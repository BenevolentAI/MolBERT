# MolBERT
This repository contains the implementation of the MolBERT, a state-of-the-art representation learning method based on the modern language model BERT. 

The details are described in *["Molecular representation learning with language models and domain-relevant auxiliary tasks"](https://arxiv.org/abs/2011.13230)*, presented at the Machine Learning for Molecules Workshop @ NeurIPS 2020. 

Work done by Benedek Fabian, Thomas Edlich, Héléna Gaspar, Marwin Segler, Joshua Meyers, Marco Fiscato, Mohamed Ahmed

## Installation
Create your conda environment first:
```shell script
conda create -y -q -n molbert -c rdkit rdkit=2019.03.1.0 python=3.7.3
```

Then install the package by running the following commands from the cloned directory:
```shell script
conda activate molbert
pip install -e . 
```

## Run tests
To verify your installation, execute the tests:
```shell script
python -m pytest . -p no:warnings
```

## Load pretrained model
You can download the pretrained model [here](https://ndownloader.figshare.com/files/25611290)

After downloading the weights, you can follow `scripts/featurize.py` to load the model and use it as a featurizer (you just need to replace the path in the script).

## Train model from scratch:
You can use the guacamol dataset (links at the [bottom](#data))
```shell script
molbert_smiles \
    --train_file data/guacamol_baselines/guacamol_v1_train.smiles \
    --valid_file data/guacamol_baselines/guacamol_v1_valid.smiles \
    --max_seq_length 128 \
    --batch_size 16 \
    --masked_lm 1 \
    --num_physchem_properties 200 \
    --is_same_smiles 0 \
    --permute 1 \
    --max_epochs 20 \
    --num_workers 8 \
    --val_check_interval 1
```

Add the `--tiny` flag to train a smaller model on a CPU, or the `--fast_dev_run` flag for testing purposes. For full list of options see `molbert/apps/args.py` and `molbert/apps/smiles.py`.

## Finetune
After you have trained a model, and you would like to finetune on a certain training set, you can use the `FinetuneSmilesMolbertApp` class to further specialize your model to your task.

For classification you can set can set the mode to `classification` and the `output_size` to 2.
```shell script
molbert_finetune \
    --train_file path/to/train.csv \
    --valid_file path/to/valid.csv \
    --test_file path/to/test.csv \
    --mode classification \
    --output_size 2 \
    --pretrained_model_path path/to/lightning_logs/version_0/checkpoints/last.ckpt \
    --label_column my_label_column
```
For regression set the mode to `regression` and the `output_size` to 1.
```shell script
molbert_finetune \
    --train_file path/to/train.csv \
    --valid_file path/to/valid.csv \
    --test_file path/to/test.csv \
    --mode regression \
    --output_size 1 \
    --pretrained_model_path path/to/lightning_logs/version_0/checkpoints/last.ckpt \
    --label_column pIC50
```

To reproduce the finetuning experiments we direct you to use `scripts/run_qsar_test_molbert.py` and `scripts/run_finetuning.py`. 
Both scripts rely on the [Chembench](https://github.com/shenwanxiang/ChemBench) and optionally the [CDDD](https://github.com/jrwnter/cddd) repositories. 
Please follow the installation instructions described in their READMEs.

## Data
#### Guacamol datasets
You can download pre-built datasets [here](https://figshare.com/projects/GuacaMol/56639):

md5 `05ad85d871958a05c02ab51a4fde8530` [training](https://ndownloader.figshare.com/files/13612760)  
md5 `e53db4bff7dc4784123ae6df72e3b1f0` [validation](https://ndownloader.figshare.com/files/13612766)  
md5 `677b757ccec4809febd83850b43e1616` [test](https://ndownloader.figshare.com/files/13612757)  
md5 `7d45bc95c33c10cb96ef5e78c38ac0b6` [all](https://ndownloader.figshare.com/files/13612745)  
