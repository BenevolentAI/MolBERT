""""
Adapted from: https://github.com/jrwnter/cddd/blob/master/cddd/evaluation.py

Module to to test the performance of the translation model to extract
    meaningfull features for a QSAR modelling. TWO QSAR datasets were extracted
    from literature:
    Ames mutagenicity: K. Hansen, S. Mika, T. Schroeter, A. Sutter, A. Ter Laak,
    T. Steger-Hartmann, N. Heinrich and K.-R. MuÌ´Lller, J. Chem.
    Inf. Model., 2009, 49, 2077–2081.
    Lipophilicity: Z. Wu, B. Ramsundar, E. N. Feinberg, J. Gomes, C. Geniesse,
    A. S. Pappu, K. Leswing and V. Pande, Chemical Science, 2018,
    9, 513–530.
 """

import os
import json
from datetime import datetime

import click
import numpy as np
import pandas as pd
from sklearn.svm import SVC, SVR
from sklearn import metrics
from cddd.inference import InferenceModel
from molbert.utils.featurizer.molbert_featurizer import MolBertFeaturizer
from molbert.utils.featurizer.molfeaturizer import MorganFPFeaturizer, PhysChemFeaturizer
from molbert.utils.chembench_utils import get_data, get_summary_df

_default_molbert_model_dir = '/path/to/checkpoint.ckpt'
CDDD_MODEL_DIR = '/path/to/directory/of/cddd/default_model'


def batchify(iterable, batch_size):
    for ndx in range(0, len(iterable), batch_size):
        batch = iterable[ndx: min(ndx + batch_size, len(iterable))]
        yield batch


def cv(dataset, summary_df, cddd_model_dir, molbert_model_dir):
    df, indices = get_data(dataset)
    cddd = InferenceModel(cddd_model_dir)  # type: ignore
    molbert = MolBertFeaturizer(molbert_model_dir, embedding_type='average-1-cat-pooled', max_seq_len=200, device='cpu')  # type: ignore
    ecfp = MorganFPFeaturizer(fp_size=2048, radius=2, use_counts=True, use_features=False)
    rdkit_norm = PhysChemFeaturizer(normalise=True)

    cddd_fn = lambda smiles: cddd.seq_to_emb(smiles)
    molbert_fn = lambda smiles: molbert.transform(smiles)[0]
    ecfp_fn = lambda smiles: ecfp.transform(smiles)[0]
    rdkit_norm_fn = lambda smiles: rdkit_norm.transform(smiles)[0]

    for i, (train_idx, valid_idx, test_idx) in enumerate(indices):
        train_df = df.iloc[train_idx]
        valid_df = df.iloc[valid_idx]

        # combine train and valid set as SVMs don't use a validation set, but NNs do.
        # this way they use the same amount of data.
        train_df = pd.concat([train_df, valid_df])
        test_df = df.iloc[test_idx]

        fn_combos = [('cddd', cddd_fn), ('molbert', molbert_fn), ('ECFP4', ecfp_fn), ('rdkit_norm', rdkit_norm_fn)]

        for feat_name, feat_fn in fn_combos:
            train_features = np.vstack([feat_fn(batch) for batch in batchify(train_df['SMILES'], 256)])
            train_labels = train_df[df.columns[-1]]

            test_features = np.vstack([feat_fn(batch) for batch in batchify(test_df['SMILES'], 256)])
            test_labels = test_df[df.columns[-1]]

            mode = summary_df[summary_df['task_name'] == dataset].iloc[0]['task_type'].strip()

            np.random.seed(i)
            if mode == 'regression':
                model = SVR(C=5.0)
            elif mode == 'classification':
                model = SVC(5.0, probability=True)
            else:
                raise ValueError(f'Mode has to be either classification or regression but was {mode}.')

            model.fit(train_features, train_labels)

            predictions = model.predict(test_features)

            if mode == 'classification':
                # predict probabilities (needed for some metrics) and get probs of positive class ([:, 1])
                prob_predictions = model.predict_proba(test_features)[:, 1]
                metrics_dict = {
                    'AUROC': lambda: metrics.roc_auc_score(test_labels, prob_predictions),
                    'AveragePrecision': lambda: metrics.average_precision_score(test_labels, prob_predictions),
                    'Accuracy': lambda: metrics.accuracy_score(test_labels, predictions),
                }
            else:
                metrics_dict = {
                    'MAE': lambda: metrics.mean_absolute_error(test_labels, predictions),
                    'RMSE': lambda: np.sqrt(metrics.mean_squared_error(test_labels, predictions)),
                    'MSE': lambda: metrics.mean_squared_error(test_labels, predictions),
                    'R2': lambda: metrics.r2_score(test_labels, predictions),
                }

            metric_values = {}
            for name, callable_metric in metrics_dict.items():
                try:
                    metric_values[name] = callable_metric()
                except Exception as e:
                    print(f'unable to calculate {name} metric')
                    print(e)
                    metric_values[name] = np.nan

            default_path = os.path.join('./logs/', datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ'))
            output_dir = os.path.join(default_path, dataset, str(i))
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, f'{feat_name}_metrics.json'), 'w+') as fp:
                json.dump(metric_values, fp)


@click.command()
@click.option('--molbert_model_dir', type=str, default=_default_molbert_model_dir)
@click.option('--cddd_model_dir', type=str, default=CDDD_MODEL_DIR)
def main(molbert_model_dir, cddd_model_dir):
    summary_df = get_summary_df()

    for dataset in summary_df['task_name'].unique():
        print(f'Running experiment for {dataset}')
        cv(dataset, summary_df, cddd_model_dir, molbert_model_dir)


if __name__ == "__main__":
    main()
