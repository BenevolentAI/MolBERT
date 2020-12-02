import os
import pickle
import chembench
from chembench import load_data


def get_data(dataset):
    """ Check if exists, download if not, save splits return paths to separated splits """
    df, indices = load_data(dataset)
    df = df.rename(columns={'smiles': 'SMILES'})
    df.columns = [col.replace(' ', '_') for col in df.columns]
    return df, indices


def get_summary_df():
    chembench_path = os.path.dirname(chembench.__file__)
    with open(os.path.join(chembench_path, 'notebook/summary.pkl'), 'rb') as f:
        summary_df = pickle.load(f)

    # filter such that dataframe only contains datasets with a single task
    summary_df = summary_df[summary_df['n_task'] == 1]

    # filter out PDB tasks
    summary_df = summary_df[~summary_df['task_name'].str.contains('PDB')]

    return summary_df
