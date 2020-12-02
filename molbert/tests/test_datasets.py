from molbert.datasets.finetune import BertFinetuneSmilesDataset
from molbert.datasets.smiles import BertSmilesDataset
from molbert.utils.lm_utils import get_seq_lengths
from molbert.tests.utils import finetune_data_path, data_path, featurizer  # noqa: F401


def test_smiles_dataset(data_path, featurizer):  # noqa: F811
    max_len = 64
    single_seq_len, total_seq_len = get_seq_lengths(max_len, False)
    dataset = BertSmilesDataset(
        input_path=data_path,
        featurizer=featurizer,
        single_seq_len=single_seq_len,
        total_seq_len=total_seq_len,
        num_physchem=0,
        is_same=False,
    )

    (inputs, labels), _ = dataset[0]
    assert inputs['input_ids'].shape == (max_len,)
    assert inputs['token_type_ids'].shape == (max_len,)
    assert inputs['attention_mask'].shape == (max_len,)
    assert labels['lm_label_ids'].shape == (max_len,)
    assert labels['unmasked_lm_label_ids'].shape == (max_len,)


def test_smiles_dataset_is_same(data_path, featurizer):  # noqa: F811
    max_len = 64
    single_seq_len, total_seq_len = get_seq_lengths(max_len, True)
    dataset = BertSmilesDataset(
        input_path=data_path,
        featurizer=featurizer,
        single_seq_len=single_seq_len,
        total_seq_len=total_seq_len,
        num_physchem=0,
        is_same=True,
    )

    (inputs, labels), _ = dataset[0]
    assert inputs['input_ids'].shape == (total_seq_len,)
    assert inputs['token_type_ids'].shape == (total_seq_len,)
    assert inputs['attention_mask'].shape == (total_seq_len,)
    assert labels['lm_label_ids'].shape == (total_seq_len,)
    assert labels['unmasked_lm_label_ids'].shape == (total_seq_len,)


def test_smiles_dataset_physchem(data_path, featurizer):  # noqa: F811
    max_len = 64
    num_physchem = 200
    single_seq_len, total_seq_len = get_seq_lengths(max_len, False)
    dataset = BertSmilesDataset(
        input_path=data_path,
        featurizer=featurizer,
        single_seq_len=single_seq_len,
        total_seq_len=total_seq_len,
        num_physchem=num_physchem,
        is_same=False,
    )

    (inputs, labels), _ = dataset[0]
    assert inputs['input_ids'].shape == (max_len,)
    assert inputs['token_type_ids'].shape == (max_len,)
    assert inputs['attention_mask'].shape == (max_len,)
    assert labels['lm_label_ids'].shape == (max_len,)
    assert labels['unmasked_lm_label_ids'].shape == (max_len,)
    assert labels['physchem_props'].shape == (num_physchem,)


def test_finetune_dataset(finetune_data_path, featurizer):  # noqa: F811
    max_len = 64
    single_seq_len, total_seq_len = get_seq_lengths(max_len, False)
    dataset = BertFinetuneSmilesDataset(
        input_path=finetune_data_path,
        featurizer=featurizer,
        single_seq_len=single_seq_len,
        total_seq_len=total_seq_len,
        label_column='measured_log_solubility_in_mols_per_litre',
    )

    (inputs, labels), _ = dataset[0]
    assert inputs['input_ids'].shape == (max_len,)
    assert inputs['token_type_ids'].shape == (max_len,)
    assert inputs['attention_mask'].shape == (max_len,)
    assert labels['lm_label_ids'].shape == (max_len,)
    assert labels['unmasked_lm_label_ids'].shape == (max_len,)
    assert labels['finetune'].shape == (1,)
