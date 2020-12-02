import os
import tempfile

import pytest
import torch
import yaml

from molbert.apps.finetune import FinetuneSmilesMolbertApp
from molbert.apps.smiles import SmilesMolbertApp
from molbert.models.finetune import FinetuneSmilesMolbertModel
from molbert.models.smiles import SmilesMolbertModel
from molbert.utils.featurizer.molfeaturizer import SmilesIndexFeaturizer


@pytest.fixture()
def data_path():
    return os.path.join(os.path.dirname(__file__), 'test_data.smi')


@pytest.fixture()
def finetune_data_path():
    return os.path.join(os.path.dirname(__file__), 'test_data_regression.csv')


@pytest.fixture()
def smiles_args(data_path):
    raw_args_str = f'--train_file {data_path} ' f'--valid_file {data_path} ' f'--test_file {data_path} ' f'--tiny'
    raw_args = raw_args_str.split(' ')
    app = SmilesMolbertApp()

    return app.parse_args(raw_args)


@pytest.fixture()
def featurizer():
    return SmilesIndexFeaturizer.bert_smiles_index_featurizer(64)


@pytest.fixture()
def smiles_model(smiles_args):
    model = SmilesMolbertModel(smiles_args)
    return model


@pytest.fixture()
def finetune_args(smiles_model, finetune_data_path):
    # smiles model is needed to create checkpoint
    tmp_dir = tempfile.mkdtemp()
    os.makedirs(os.path.join(tmp_dir, 'version_0'))

    pretrained_path = os.path.join(tmp_dir, 'version_0/checkpoint.ckpt')
    torch.save(smiles_model, pretrained_path)

    hparams_path = os.path.join(tmp_dir, 'hparams.yaml')
    dummy_hparams = {"tiny": True, "masked_lm": 1, "is_same_smiles": 0, "num_physchem_properties": 0}
    with open(hparams_path, 'w') as outfile:
        yaml.dump(dummy_hparams, outfile)

    raw_args_str = (
        f'--train_file {finetune_data_path} '
        f'--valid_file {finetune_data_path} '
        f'--test_file {finetune_data_path} '
        f'--pretrained_model_path {pretrained_path} '
        f'--label_column measured_log_solubility_in_mols_per_litre '
        f'--output_size 1 '
        f'--mode regression '
        f'--tiny'
    )
    raw_args = raw_args_str.split(' ')
    app = FinetuneSmilesMolbertApp()

    return app.parse_args(raw_args)


@pytest.fixture()
def finetune_model(finetune_args):
    model = FinetuneSmilesMolbertModel(finetune_args)
    return model


@pytest.fixture()
def dummy_model_inputs():
    inputs = dict(
        input_ids=torch.zeros(64, dtype=torch.long).unsqueeze(0),
        token_type_ids=torch.zeros(64, dtype=torch.long).unsqueeze(0),
        attention_mask=torch.zeros(64, dtype=torch.long).unsqueeze(0),
    )
    return inputs


@pytest.fixture()
def dummy_labels():
    labels = dict(
        lm_label_ids=torch.tensor([1] * 64, dtype=torch.long).unsqueeze(0),
        unmasked_lm_label_ids=torch.tensor([1] * 64, dtype=torch.long).unsqueeze(0),
        is_same=torch.tensor(0, dtype=torch.long).unsqueeze(0),
        finetune=torch.tensor(0, dtype=torch.long).unsqueeze(0),
    )

    return labels
