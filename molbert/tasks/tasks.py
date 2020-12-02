from abc import ABC, abstractmethod

import torch
from torch import nn
from torch.nn.modules.loss import CrossEntropyLoss, MSELoss
from transformers.modeling_bert import BertLMPredictionHead

from molbert.tasks.heads import PhysChemHead, FinetuneHead, IsSameHead


class BaseTask(nn.Module, ABC):
    def __init__(self, name):
        super().__init__()
        self.name = name

    @abstractmethod
    def forward(self, sequence_output, pooled_output):
        raise NotImplementedError

    @abstractmethod
    def compute_loss(self, batch_labels, batch_predictions) -> torch.Tensor:
        raise NotImplementedError


class FinetuneTask(BaseTask):
    def __init__(self, name, config):
        super().__init__(name)
        self.loss: nn.Module
        self.mode = config.mode
        if config.mode == 'regression':
            self.loss = MSELoss()
        elif config.mode == 'classification':
            self.loss = CrossEntropyLoss()
        else:
            raise ValueError(f'config.mode must be in [regression, classification] but was {config.mode}')

        self.head = FinetuneHead(config)

    def forward(
        self,
        sequence_output,
        pooled_output,
    ):
        return self.head(pooled_output)

    def compute_loss(self, batch_labels, batch_predictions) -> torch.Tensor:
        predictions = batch_predictions[self.name]
        labels = batch_labels[self.name]

        if self.mode == 'classification':
            labels = labels.long().squeeze(1)

        return self.loss(predictions, labels)


class PhyschemTask(BaseTask):
    def __init__(self, name, config):
        super().__init__(name)
        self.loss = MSELoss()

        self.physchem_head = PhysChemHead(config)

    def forward(self, sequence_output, pooled_output):
        return self.physchem_head(pooled_output)

    def compute_loss(self, batch_labels, batch_predictions) -> torch.Tensor:
        return self.loss(batch_predictions[self.name], batch_labels[self.name])


class MaskedLMTask(BaseTask):
    def __init__(self, name, config):
        super().__init__(name)
        self.loss = CrossEntropyLoss(ignore_index=-1)
        self.vocab_size = config.vocab_size
        self.masked_lm_head = BertLMPredictionHead(config)

    def forward(self, sequence_output, pooled_output):
        return self.masked_lm_head(sequence_output)

    def compute_loss(self, batch_labels, batch_predictions) -> torch.Tensor:
        return self.loss(
            batch_predictions['masked_lm'].view(-1, self.vocab_size), batch_labels['lm_label_ids'].view(-1)
        )


class IsSameTask(BaseTask):
    def __init__(self, name, config):
        super().__init__(name)
        self.loss = CrossEntropyLoss(ignore_index=-1)
        self.is_same_head = IsSameHead(config)

    def forward(self, sequence_output, pooled_output):
        return self.is_same_head(pooled_output)

    def compute_loss(self, batch_labels, batch_predictions) -> torch.Tensor:
        return self.loss(batch_predictions[self.name].view(-1, 2), batch_labels[self.name].view(-1))
