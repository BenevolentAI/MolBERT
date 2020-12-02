import logging
import random
from typing import List, Optional, Tuple, Union

import torch
from molbert.datasets.base import BaseBertDataset
from molbert.utils.featurizer.molfeaturizer import PhysChemFeaturizer

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class BertSmilesDataset(BaseBertDataset):
    def __init__(
        self,
        input_path,
        featurizer,
        single_seq_len,
        total_seq_len,
        num_physchem=0,
        permute=False,
        named_descriptor_set='all',
        *args,
        **kwargs,
    ):

        super().__init__(input_path, featurizer, single_seq_len, total_seq_len, *args, **kwargs)
        self.num_physchem = num_physchem
        self.permute = permute
        self.physchem_featurizer: PhysChemFeaturizer

        if self.num_physchem > 0:
            descriptor_list = PhysChemFeaturizer.get_descriptor_subset(named_descriptor_set, self.num_physchem)
            self.physchem_featurizer = PhysChemFeaturizer(descriptors=descriptor_list, normalise=True)

    def calculate_physchem_props(self, smiles: str):
        physchem, valid = self.physchem_featurizer.transform_single(smiles)
        assert bool(valid), f'Cannot compute the physchem props for {smiles}'

        return physchem[: self.num_physchem]

    def get_sequence(
        self, index: int, permute_sample: bool = False, *args, **kwargs
    ) -> Tuple[Optional[List[str]], bool]:
        """
        Returns a permuted SMILES given a position index
        """

        sequence = self.sequences[index]
        if sequence is None:
            return None, False

        if self.permute or permute_sample:
            permutation = self.featurizer.permute_smiles(smiles_str=sequence)
        else:
            permutation = self.featurizer.standardise(sequence)

        if permutation is None or not self.featurizer.is_legal(permutation):
            return None, False

        return permutation, True

    def get_related_seq(self, index: int, max_retries: int = 10) -> Tuple[Optional[List[str]], bool]:
        """
        Returns a permuted SMILES given a position index
        """
        # related seq is always permuted
        seq, valid = self.get_sequence(index, permute_sample=True)

        return seq, valid

    def get_unrelated_seq(self, avoid_idx: int, max_retries: int = 10) -> Tuple[Optional[List[str]], bool]:
        """
        Tries to returns a permuted SMILES different than the one in input
        """
        for _ in range(max_retries):

            index = random.randint(0, len(self) - 1)

            if index == avoid_idx:
                continue

            seq, valid = self.get_sequence(index, permute_sample=True)

            if valid:
                return seq, True

        return None, False

    def prepare_sample(
        self,
        cur_id: int,
        t1: Optional[Union[List[str], str]],
        t2: Optional[Union[List[str], str]] = None,
        is_same: Optional[bool] = None,
    ):

        assert t1 is not None

        encoded_tokens_a = list(self.featurizer.encode(t1))
        encoded_tokens_b = list(self.featurizer.encode(t2)) if t2 else None

        inputs, labels = super().prepare_sample(cur_id, encoded_tokens_a, encoded_tokens_b, is_same)

        if self.num_physchem > 0:
            # get the original sequence of the sample to support cache indexing
            sequence = str(self.sequences[cur_id])
            physchem_properties = self.calculate_physchem_props(sequence)

            labels['physchem_props'] = torch.tensor(physchem_properties, dtype=torch.float)

        return inputs, labels

    def get_invalid_sample(self):

        inputs, labels = super().get_invalid_sample()

        if self.num_physchem > 0:
            labels['physchem_props'] = torch.zeros(self.num_physchem, dtype=torch.float)

        return inputs, labels
