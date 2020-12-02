import logging
import random

import numpy as np
from transformers import BertConfig

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class InputFeatures(object):
    """A single set of original_features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, is_next, lm_label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.is_next = is_next
        self.lm_label_ids = lm_label_ids


class InputExample(object):
    """A single training/test example for the language model."""

    def __init__(self, guid, tokens_a, tokens_b=None, is_next=None, lm_labels=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            tokens_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            tokens_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.tokens_a = tokens_a
        self.tokens_b = tokens_b
        self.is_next = is_next  # nextSentence
        self.lm_labels = lm_labels  # masked words for language model


def unmask_lm_labels(input_ids, masked_lm_labels):
    """
    Get unmasked LM labels
    """
    input_ids = np.asarray(input_ids)
    masked_lm_labels = np.asarray(masked_lm_labels)

    inp_shape = input_ids.shape
    unmasked_lm_labels = np.copy(input_ids.flatten())
    masked_token_indices = np.where(masked_lm_labels.flatten() != -1)[0]
    masked_tokens = masked_lm_labels.flatten()[masked_token_indices]
    unmasked_lm_labels[masked_token_indices] = masked_tokens
    unmasked_lm_labels = unmasked_lm_labels.reshape(inp_shape)
    return unmasked_lm_labels


def get_seq_lengths(single_seq_len, is_same):

    if is_same:
        # when there are 2 tokens, max_seq_length is double and account for BERT adding [CLS], [SEP], [SEP]
        total_seq_len = single_seq_len * 2
    else:
        # Account for BERT adding [CLS], [SEP]
        total_seq_len = single_seq_len

    single_seq_len -= 2
    return single_seq_len, total_seq_len


def random_word(tokens, tokenizer, inference_mode: bool = False):
    """
    Masking some random tokens for Language Model task with probabilities as in the original BERT paper.

    Args:
        tokens: list of str, tokenized sentence.
        tokenizer: Tokenizer, object used for tokenization (we need it's vocab here)
        inference_mode: if True, don't do any input modifications. Used at inference time.

    Returns
        tokens: masked tokens
        output_label: labels for LM prediction
    """
    output_label = []

    for i in range(len(tokens)):
        prob = random.random()
        # mask token with 15% probability
        if prob < 0.15 and not inference_mode:
            prob /= 0.15

            # 80% randomly change token to mask token
            if prob < 0.8:
                token = '[MASK]'
            # 10% randomly change token to random token
            elif prob < 0.9:
                token = random.choice(list(tokenizer.token_to_idx.items()))[0]
                while (token in tokenizer.symbols) or (token == tokens[i]):
                    token = random.choice(list(tokenizer.token_to_idx.items()))[0]
            # -> rest 10% randomly keep current token
            else:
                token = tokens[i]

            # set the replace token and append token to output (we will predict these later)
            try:
                output_label.append(tokenizer.token_to_idx[tokens[i]])
                tokens[i] = token
            except KeyError:
                # For unknown words (should not occur with BPE vocab)
                output_label.append(tokenizer.token_to_idx['[UNK]'])
                logger.warning('Cannot find token "{}" in token_to_idx. Using [UNK] instead'.format(tokens[i]))
        else:
            # no masking token (will be ignored by loss function later)
            output_label.append(-1)

    return tokens, output_label


def convert_example_to_features(example, max_seq_length, tokenizer, inference_mode: bool = False):
    """
    Convert a raw sample (pair of sentences as tokenized strings) into a proper training sample with
    IDs, LM labels, input_mask, CLS and SEP tokens etc.

    Args:
        example: InputExample, containing sentence input as strings and is_next label
        max_seq_length: maximum length of sequence.
        tokenizer: Tokenizer
        inference_mode: if True, don't do any input modifications. Used at inference time.

    Returns:
        features: InputFeatures, containing all inputs and labels of one sample as IDs (as used for model training)
    """
    tokens_a = example.tokens_a
    tokens_b = example.tokens_b

    if tokens_b is None:
        tokens_b = []

    tokens_a, t1_label = random_word(tokens_a, tokenizer, inference_mode)
    tokens_b, t2_label = random_word(tokens_b, tokenizer, inference_mode)

    # concatenate lm labels and account for CLS, SEP, SEP
    lm_label_ids = [-1] + t1_label + [-1] + (t2_label + [-1] if len(t2_label) > 0 else [])

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0   0   0   0  0     0 0
    #
    # Where 'type_ids' are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambiguously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as as the 'sentence vector'. Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens = []
    segment_ids = []
    tokens.append('[CLS]')
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append('[SEP]')
    segment_ids.append(0)

    if len(tokens_b) > 0:
        for token in tokens_b:
            tokens.append(token)
            segment_ids.append(1)
        tokens.append('[SEP]')
        segment_ids.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        lm_label_ids.append(-1)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(lm_label_ids) == max_seq_length

    # if example.guid < 5:
    #     logger.info('*** Example ***')
    #     logger.info('guid: %s' % example.guid)
    #     logger.info('tokens: %s' % ' '.join([str(x) for x in tokens]))
    #     logger.info('input_ids: %s' % ' '.join([str(x) for x in input_ids]))
    #     logger.info('input_mask: %s' % ' '.join([str(x) for x in input_mask]))
    #     logger.info('segment_ids: %s' % ' '.join([str(x) for x in segment_ids]))
    #     logger.info('LM label: %s ' % lm_label_ids)
    #     logger.info('Is next sentence label: %s ' % example.is_next)

    features = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        lm_label_ids=lm_label_ids,
        is_next=example.is_next,
    )
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


class BertConfigExtras(BertConfig):
    """
    Same as BertConfig, BUT
    adds any kwarg as a member field
    """

    def __init__(
        self,
        vocab_size_or_config_json_file,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        **kwargs,
    ):
        super(BertConfigExtras, self).__init__(
            vocab_size_or_config_json_file,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            max_position_embeddings=max_position_embeddings,
            type_vocab_size=type_vocab_size,
            initializer_range=initializer_range,
        )

        for k, v in kwargs.items():
            setattr(self, k, v)
