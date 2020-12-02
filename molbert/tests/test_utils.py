import random

import numpy as np

from molbert.utils.lm_utils import (
    InputExample,
    _truncate_seq_pair,
    convert_example_to_features,
    get_seq_lengths,
    random_word,
    unmask_lm_labels,
)
from molbert.utils.featurizer.molfeaturizer import SmilesIndexFeaturizer

TOKENIZER = SmilesIndexFeaturizer.bert_smiles_index_featurizer(10)


def test_get_seq_lenghts_with_issame():
    seqlen = 10
    single_seq_len, total_seq_len = get_seq_lengths(seqlen, is_same=True)

    assert single_seq_len == seqlen - 2
    assert total_seq_len == 2 * seqlen


def test_get_seq_lenghts_without_issame():
    seqlen = 10
    single_seq_len, total_seq_len = get_seq_lengths(seqlen, is_same=False)

    assert single_seq_len == seqlen - 2
    assert total_seq_len == seqlen


def test_get_unmasked_labels():
    random.seed(1)
    tokens_a = list('C1CCCCC1')
    tokens_b = None
    example = InputExample(guid=1, tokens_a=tokens_a, tokens_b=tokens_b, is_next=False)

    # transform sample to original_features
    features = convert_example_to_features(example, 10, TOKENIZER)

    # get the unmasked label id's - useful for calculating accuracy
    unmasked_lm_label_ids = unmask_lm_labels(features.input_ids, features.lm_label_ids)

    # for all input tokens
    for i in range(len(features.input_ids)):
        # if token is masked:
        if features.lm_label_ids[i] == -1:
            # then the unmasked token is equal to the input token
            assert unmasked_lm_label_ids[i] == features.input_ids[i]
        else:
            # else the unmasked label is equal to the lm_label_id
            assert unmasked_lm_label_ids[i] == features.lm_label_ids[i]


def test_random_word():
    smiles = list('C1CCCCC1')
    random.seed(1)
    expected_output_labels = np.array([TOKENIZER.token_to_idx[t] for t in smiles])

    masked_tokens, output_labels = random_word(smiles, TOKENIZER)

    assert np.array_equal(masked_tokens, np.array(['F', '1', 'C', 'C', 'C', 'C', '[MASK]', '[MASK]']))
    mask = np.array([True, False, False, False, False, False, True, True])
    expected_output_labels[~mask] = -1

    assert np.array_equal(output_labels, expected_output_labels)


def test_convert_example_to_features():
    example = InputExample(guid=1, tokens_a=list('C1CCCCC1'), tokens_b=None)

    convert_example_to_features(example, TOKENIZER.max_length, TOKENIZER)


def test_truncate_seq_pair_concatenation_is_shorter_than_max_length():
    # given two sequences and a max_length where the two sequences together are shorter than the max length
    tokens_a = list(range(10))
    tokens_b = list(range(5))
    max_length = 20

    # when truncation is called
    _truncate_seq_pair(tokens_a, tokens_b, max_length)

    # then the sequences haven't changed
    assert tokens_a == list(range(10))
    assert tokens_b == list(range(5))


def test_truncate_seq_pair_concatenation_is_longer_than_max_length():
    # given two sequences and a max_length where the two sequences together are longer than the max length
    tokens_a = list(range(10))
    tokens_b = list(range(5))
    max_length = 10

    # when truncation is called
    _truncate_seq_pair(tokens_a, tokens_b, max_length)

    # then the longer sequences has been truncated
    assert tokens_a == list(range(5))
    assert tokens_b == list(range(5))
