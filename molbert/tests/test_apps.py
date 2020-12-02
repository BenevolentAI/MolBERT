import glob
import tempfile


from molbert.apps.finetune import FinetuneSmilesMolbertApp
from molbert.apps.smiles import SmilesMolbertApp
from molbert.tests.utils import data_path, finetune_data_path, smiles_model, smiles_args  # noqa: F401


def test_smiles_and_finetune_model(data_path, finetune_data_path, smiles_model):  # noqa: F811
    output_dir = tempfile.mkdtemp()
    raw_args_str = (
        f"--max_seq_length 512 "
        f"--batch_size 32 "
        f"--max_epochs 1 "
        f"--num_workers 0 "
        f"--fast_dev_run 0 "
        f"--train_file {data_path} "
        f"--valid_file {data_path} "
        f"--test_file {data_path} "
        f"--masked_lm 1 "
        f"--is_same_smiles 1 "
        f"--num_physchem_properties 200 "
        f"--gpus 0 "
        f"--learning_rate 0.0001 "
        f"--learning_rate_scheduler linear_with_warmup "
        f"--default_root_dir {output_dir} "
        f"--tiny"
    )
    raw_args = raw_args_str.split(' ')
    SmilesMolbertApp().run(raw_args)

    ckpt = glob.glob(f'{output_dir}/**/*.ckpt', recursive=True)[0]

    raw_args_str = (
        f"--max_seq_length 512 "
        f"--batch_size 16 "
        f"--max_epochs 3 "
        f"--num_workers 0 "
        f"--fast_dev_run 0 "
        f"--train_file {finetune_data_path} "
        f"--valid_file {finetune_data_path} "
        f"--test_file {finetune_data_path} "
        f"--mode regression "
        f"--output_size 1 "
        f"--pretrained_model_path {ckpt} "
        f"--label_column measured_log_solubility_in_mols_per_litre "
        f"--freeze_level 1 "
        f"--gpus 0 "
        f"--learning_rate 0.0001 "
        f"--learning_rate_scheduler linear_with_warmup "
        f"--tiny"
    )
    raw_args = raw_args_str.split(' ')

    FinetuneSmilesMolbertApp().run(raw_args)
