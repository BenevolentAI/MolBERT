from molbert.datasets.finetune import BertFinetuneSmilesDataset
from molbert.models.finetune import FinetuneSmilesMolbertModel
from molbert.tests.utils import (  # noqa: F401
    finetune_args,
    finetune_model,
    dummy_model_inputs,
    dummy_labels,
    smiles_model,
    smiles_args,
    data_path,
    finetune_data_path,
)


def test_finetune_model_tasks(finetune_args, finetune_model, dummy_model_inputs, dummy_labels):  # noqa: F811
    model = FinetuneSmilesMolbertModel(finetune_args)

    config = model.get_config()
    tasks = model.get_tasks(config)

    assert len(tasks) == 1

    # test that forward returns outputs for all tasks
    output = model(dummy_model_inputs)

    assert 'finetune' in output.keys()


def test_load_datasets(finetune_args):  # noqa: F811
    model = FinetuneSmilesMolbertModel(finetune_args)

    datasets = model.load_datasets()

    for key in ['train', 'valid', 'test']:
        assert key in datasets.keys()
        assert isinstance(datasets[key], BertFinetuneSmilesDataset)
