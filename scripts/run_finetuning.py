import os
from datetime import datetime

import click
import tempfile

from molbert.apps.finetune import FinetuneSmilesMolbertApp
from molbert.utils.chembench_utils import get_data, get_summary_df


def finetune(
    dataset,
    train_path,
    valid_path,
    test_path,
    mode,
    label_column,
    pretrained_model_path,
    max_epochs,
    freeze_level,
    learning_rate,
    num_workers,
    batch_size,
):
    """
    This function runs finetuning for given arguments.

    Args:
        dataset: Name of the MoleculeNet dataset, e.g. BBBP
        train_path: file to the csv file containing the training data
        valid_path: file to the csv file containing the validation data
        test_path: file to the csv file containing the test data
        mode: either regression or classification
        label_column: name of the column in the csv files containing the labels
        pretrained_model_path: path to a pretrained molbert model
        max_epochs: how many epochs to run at most
        freeze_level: determines what parts of the model will be frozen. More details are given in molbert/apps/finetune.py
        learning_rate: what learning rate to use
        num_workers: how many workers to use
        batch_size: what batch size to use for training
    """

    default_path = os.path.join('./logs/', datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ'))
    output_dir = os.path.join(default_path, dataset)
    raw_args_str = (
        f"--max_seq_length 512 "
        f"--batch_size {batch_size} "
        f"--max_epochs {max_epochs} "
        f"--num_workers {num_workers} "
        f"--fast_dev_run 0 "
        f"--train_file {train_path} "
        f"--valid_file {valid_path} "
        f"--test_file {test_path} "
        f"--mode {mode} "
        f"--output_size {1 if mode == 'regression' else 2} "
        f"--pretrained_model_path {pretrained_model_path} "
        f"--label_column {label_column} "
        f"--freeze_level {freeze_level} "
        f"--gpus 1 "
        f"--learning_rate {learning_rate} "
        f"--learning_rate_scheduler linear_with_warmup "
        f"--default_root_dir {output_dir}"
    )

    raw_args = raw_args_str.split(" ")

    lightning_trainer = FinetuneSmilesMolbertApp().run(raw_args)
    return lightning_trainer


def cv(dataset, summary_df, pretrained_model_path, freeze_level, learning_rate, num_workers, batch_size):
    """
    This function runs cross-validation for finetuning MolBERT. The splits are obtained from ChemBench.

    Args:
        dataset: Name of the MoleculeNet dataset
        summary_df: summary dataframe loaded from chembench
        pretrained_model_path: path to a pretrained MolBERT model
        freeze_level: determines which parts of the model will be frozen.
        learning_rate: what learning rate to use
        num_workers: how many processes to use for data loading
        batch_size: what batch size to use
    """
    df, indices = get_data(dataset)
    print('dataset loaded', df.shape)

    for i, (train_idx, valid_idx, test_idx) in enumerate(indices):
        train_df = df.iloc[train_idx]
        valid_df = df.iloc[valid_idx]
        test_df = df.iloc[test_idx]

        with tempfile.TemporaryDirectory() as tmpdir:
            train_path = os.path.join(tmpdir, f"{dataset}_train.csv")
            valid_path = os.path.join(tmpdir, f"{dataset}_valid.csv")
            test_path = os.path.join(tmpdir, f"{dataset}_test.csv")

            train_df.to_csv(train_path)
            valid_df.to_csv(valid_path)
            test_df.to_csv(test_path)

            mode = summary_df[summary_df['task_name'] == dataset].iloc[0]['task_type'].strip()
            print('mode =', mode)

            trainer = finetune(
                dataset=dataset,
                train_path=train_path,
                valid_path=valid_path,
                test_path=test_path,
                mode=mode,
                label_column=df.columns[-1],
                pretrained_model_path=pretrained_model_path,
                max_epochs=50,
                freeze_level=freeze_level,
                learning_rate=learning_rate,
                num_workers=num_workers,
                batch_size=batch_size,
            )
            print(f'fold {i}: saving model to: ', trainer.ckpt_path)

            trainer.test()


@click.command()
@click.option('--pretrained_model_path', type=str, required=True)
@click.option('--freeze_level', type=int, required=True)
@click.option('--learning_rate', type=float, required=True)
@click.option('--num_workers', type=int, default=1)
@click.option('--batch_size', type=int, default=16)
def main(pretrained_model_path, freeze_level, learning_rate, num_workers, batch_size):
    summary_df = get_summary_df()

    for dataset in summary_df['task_name'].unique():
        print(f'Running experiment for {dataset}')
        cv(dataset, summary_df, pretrained_model_path, freeze_level, learning_rate, num_workers, batch_size)


if __name__ == "__main__":
    main()
