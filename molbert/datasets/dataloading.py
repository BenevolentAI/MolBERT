import logging
from typing import Callable

import torch
from torch.utils.data import DataLoader


class MolbertDataLoader(DataLoader):
    """
    A custom data loader that does some molbert specific things.
    1) it skips invalid batches and replaces them with oversampled valid batches such that always n_batches are
       created.
    2) it does the valid filtering and trimming in the workers
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # See dataloader.pyi for examplanation of type: ignore
        self.collate_fn = self.wrapped_collate_fn(self.collate_fn)  # type: ignore

    @staticmethod
    def trim_batch(batch_inputs, batch_labels):
        _, cols = torch.where(batch_inputs['attention_mask'] == 1)
        max_idx: int = int(cols.max().item() + 1)

        for k in ['input_ids', 'token_type_ids', 'attention_mask']:
            batch_inputs[k] = batch_inputs[k][:, :max_idx].contiguous()

        for k in ['lm_label_ids', 'unmasked_lm_label_ids']:
            batch_labels[k] = batch_labels[k][:, :max_idx].contiguous()

        return batch_inputs, batch_labels

    def wrapped_collate_fn(self, collate_fn) -> Callable:
        def collate(*args, **kwargs):
            batch = collate_fn(*args, **kwargs)

            # valids here is a sequence of valid flags with the same length as the batch
            (batch_inputs, batch_labels), valids = batch

            if not valids.all():
                # filter invalid
                batch_inputs = {k: v[valids] for k, v in batch_inputs.items()}
                batch_labels = {k: v[valids] for k, v in batch_labels.items()}
                # keep trues only to make sure the format is the same
                valids = valids[valids]

            # whole batch is invalid?
            if len(valids) == 0:
                return (None, None), valids

            # trim out excessive padding
            batch_inputs, batch_labels = self.trim_batch(batch_inputs, batch_labels)

            return (batch_inputs, batch_labels), valids

        return collate

    def __iter__(self):
        num_batches_so_far = 0
        num_total_batches = len(self)
        num_accessed_batches = 0

        while num_batches_so_far < num_total_batches:
            for (batch_inputs, batch_labels), valids in super().__iter__():
                num_accessed_batches += 1
                if len(valids) == 0:
                    logging.info('EMPTY BATCH ENCOUNTERED. Skipping...')
                    continue
                num_batches_so_far += 1
                yield (batch_inputs, batch_labels), valids

        logging.info(
            f'Epoch finished. Accessed {num_accessed_batches} batches in order to train on '
            f'{num_total_batches} batches.'
        )
