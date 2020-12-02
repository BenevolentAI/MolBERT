import logging
import random
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union

import torch
from molbert.utils.lm_utils import InputExample, convert_example_to_features, unmask_lm_labels
from torch.utils.data import Dataset

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class BaseBertDataset(Dataset, ABC):
    def __init__(
        self, input_path, featurizer, single_seq_len, total_seq_len, is_same: bool = False, inference_mode: bool = False
    ):

        self.sequence_file = input_path
        self.single_seq_len = featurizer.max_length
        self.featurizer = featurizer

        self.sequences = self.load_sequences(self.sequence_file)

        self.sample_counter = 0
        self.is_same = is_same
        self.inference_mode = inference_mode
        self.single_seq_len = single_seq_len
        self.total_seq_len = total_seq_len

    @staticmethod
    def load_sequences(sequence_file):
        with open(sequence_file, 'rt') as f:
            return [line.strip() for line in f.readlines()]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        cur_id = index
        self.sample_counter += 1
        t1, t2, is_same, valid = self.get_sample(index)

        if not valid:
            return self.get_invalid_sample(), False  # not valid

        return self.prepare_sample(cur_id, t1, t2, is_same), True  # valid

    def get_invalid_sample(self):
        inputs = dict(
            input_ids=torch.zeros(self.total_seq_len, dtype=torch.long),
            token_type_ids=torch.zeros(self.total_seq_len, dtype=torch.long),
            attention_mask=torch.zeros(self.total_seq_len, dtype=torch.long),
        )

        labels = dict(
            lm_label_ids=torch.tensor([-1] * self.total_seq_len, dtype=torch.long),
            unmasked_lm_label_ids=torch.tensor([-1] * self.total_seq_len, dtype=torch.long),
        )

        if self.is_same:
            labels['is_same'] = torch.tensor(0, dtype=torch.long)

        return inputs, labels

    def get_sample(self, index: int) -> Tuple[Optional[List[str]], Optional[List[str]], bool, bool]:
        """
        Get one sample from the data consisting of one or two sequences.

        If `is_same` task is selected two sequences are returned:
            - with prob. 50% these are two related sequences.
            - with 50% the second sequence will be a random one.
        Args:
            index: The index of the input sequence

        Returns:
            seq1: first sequences
            seq2: second sequence
            label: relative to the `is_same` task
            valid: overall validity of this sample
        """
        is_same_label = False

        # retrieve the first sequence
        seq1, valid1 = self.get_sequence(index)

        # if the requested sequence is invalid -> bail
        if not valid1 or (seq1 and len(seq1) > self.single_seq_len):
            return None, None, is_same_label, False

        # just `maskedLM`, no `is_same`
        if not self.is_same:
            return seq1, None, is_same_label, True

        # get second sample
        if random.random() > 0.5:
            # get a related sequence
            seq2, valid2 = self.get_related_seq(index)
            is_same_label = True
        else:
            # get a un-related second sequence
            seq2, valid2 = self.get_unrelated_seq(avoid_idx=index)

        # if the requested mol is invalid OR too long -> bail
        if not valid2 or (seq2 and len(seq2) > self.single_seq_len):
            return None, None, is_same_label, False

        return seq1, seq2, is_same_label, True

    def prepare_sample(
        self,
        cur_id: int,
        t1: Optional[Union[List[str], str]],
        t2: Optional[Union[List[str], str]] = None,
        is_same: Optional[bool] = None,
    ):

        # combine to one sample
        cur_example = InputExample(guid=cur_id, tokens_a=t1, tokens_b=t2, is_next=is_same)

        # transform sample to original_features
        cur_features = convert_example_to_features(
            cur_example, self.total_seq_len, self.featurizer, self.inference_mode
        )

        # get the unmasked label id's - useful for calculating accuracy
        unmasked_lm_label_ids = unmask_lm_labels(cur_features.input_ids, cur_features.lm_label_ids)

        # create final tensor
        inputs = dict(
            input_ids=torch.tensor(cur_features.input_ids, dtype=torch.long),
            token_type_ids=torch.tensor(cur_features.segment_ids, dtype=torch.long),
            attention_mask=torch.tensor(cur_features.input_mask, dtype=torch.long),
        )

        labels = dict(
            lm_label_ids=torch.tensor(cur_features.lm_label_ids, dtype=torch.long),
            unmasked_lm_label_ids=torch.tensor(unmasked_lm_label_ids, dtype=torch.long),
        )

        if self.is_same:
            labels['is_same'] = torch.tensor(cur_features.is_next, dtype=torch.long)

        return inputs, labels

    @abstractmethod
    def get_sequence(self, index, *args, **kwargs) -> Tuple[Optional[List[str]], bool]:
        raise NotImplementedError

    @abstractmethod
    def get_related_seq(self, index: int, max_retries: int = 10) -> Tuple[Optional[List[str]], bool]:
        raise NotImplementedError

    @abstractmethod
    def get_unrelated_seq(self, avoid_idx, max_retries: int = 10) -> Tuple[Optional[List[str]], bool]:
        raise NotImplementedError
