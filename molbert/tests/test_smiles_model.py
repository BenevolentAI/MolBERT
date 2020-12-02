from molbert.datasets.smiles import BertSmilesDataset
from molbert.models.smiles import SmilesMolbertModel
from molbert.tasks.tasks import IsSameTask, MaskedLMTask, PhyschemTask
from molbert.tests.utils import data_path, smiles_args, smiles_model, dummy_model_inputs, dummy_labels  # noqa: F401


def test_smiles_model_tasks(smiles_args, smiles_model, dummy_model_inputs, dummy_labels):  # noqa: F811
    for masked_lm in [True, False]:
        for is_same in [True, False]:
            for num_physchem in [0, 200]:
                if not (masked_lm or is_same or num_physchem):
                    continue

                smiles_args.masked_lm = masked_lm
                smiles_args.is_same_smiles = is_same
                smiles_args.num_physchem_properties = num_physchem
                smiles_model = SmilesMolbertModel(smiles_args)

                config = smiles_model.get_config()
                tasks = smiles_model.get_tasks(config)

                num_expected_tasks = int(masked_lm) + int(is_same) + (num_physchem > 0)
                assert len(tasks) == num_expected_tasks

                # test that forward returns outputs for all tasks
                output = smiles_model(dummy_model_inputs)

                if masked_lm:
                    masked_lm_task = list(filter(lambda t: t.name == 'masked_lm', tasks))
                    assert len(masked_lm_task) == 1
                    masked_lm_task = masked_lm_task[0]
                    assert isinstance(masked_lm_task, MaskedLMTask)
                    assert 'masked_lm' in output.keys()

                if is_same:
                    is_same_task = list(filter(lambda t: t.name == 'is_same', tasks))
                    assert len(is_same_task) == 1
                    is_same_task = is_same_task[0]
                    assert isinstance(is_same_task, IsSameTask)
                    assert 'is_same' in output.keys()

                if num_physchem > 0:
                    physchem_task = list(filter(lambda t: t.name == 'physchem_props', tasks))
                    assert len(physchem_task) == 1
                    physchem_task = physchem_task[0]
                    assert isinstance(physchem_task, PhyschemTask)
                    assert 'physchem_props' in output.keys()


def test_smiles_load_datasets(smiles_args):  # noqa: F811
    model = SmilesMolbertModel(smiles_args)

    datasets = model.load_datasets()

    for key in ['train', 'valid', 'test']:
        assert key in datasets.keys()
        assert isinstance(datasets[key], BertSmilesDataset)
