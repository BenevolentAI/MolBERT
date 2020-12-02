import logging
from typing import List, Optional, Tuple, Union

import pandas as pd
import torch
from molbert.datasets.base import BaseBertDataset

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class BertFinetuneSmilesDataset(BaseBertDataset):
    def __init__(
        self,
        input_path,
        featurizer,
        single_seq_len,
        total_seq_len,
        label_column,
        permute=False,
        *args,
        **kwargs,
    ):
        super().__init__(input_path, featurizer, single_seq_len, total_seq_len, *args, **kwargs)
        self.permute = permute
        data = pd.read_csv(input_path)
        self.labels = data[label_column]

    @staticmethod
    def load_sequences(sequence_file):
        data = pd.read_csv(sequence_file)
        return data['SMILES']

    def get_related_seq(self, index: int, max_retries: int = 10) -> Tuple[Optional[List[str]], bool]:
        # unused for finetuning
        raise NotImplementedError

    def get_unrelated_seq(self, avoid_idx, max_retries: int = 10) -> Tuple[Optional[List[str]], bool]:
        # unused for finetuning
        raise NotImplementedError

    def get_sequence(self, index: int, *args, **kwargs) -> Tuple[Optional[List[str]], bool]:
        """
        Returns a permuted SMILES given a position index
        """
        sequence = self.sequences[index]
        if sequence is None:
            return None, False

        if self.permute:
            permutation = self.featurizer.permute_smiles(smiles_str=sequence)
        else:
            permutation = self.featurizer.standardise(sequence)

        if permutation is None or not self.featurizer.is_legal(permutation):
            return None, False

        return permutation, True

    def prepare_sample(
        self,
        cur_id: int,
        t1: Optional[Union[List[str], str]],
        t2: Optional[Union[List[str], str]] = None,
        is_same: Optional[bool] = None,
    ):

        assert t1 is not None
        assert t2 is None
        assert not is_same

        encoded_tokens_a = list(self.featurizer.encode(t1))

        inputs, labels = super().prepare_sample(cur_id, encoded_tokens_a, None, is_same)

        labels['finetune'] = torch.tensor(self.labels[cur_id], dtype=torch.float).unsqueeze(0)

        return inputs, labels

    def get_invalid_sample(self):

        inputs, labels = super().get_invalid_sample()
        labels['finetune'] = torch.tensor(0, dtype=torch.float).unsqueeze(0)

        return inputs, labels
