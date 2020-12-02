import json
import logging
import os
from typing import List, Dict

import numpy as np
import torch
from pytorch_lightning.metrics import RMSE, MSE, AUROC, AveragePrecision, Accuracy, MAE
from sklearn.metrics import r2_score
from torch import nn
from torch.utils.data import DataLoader

from molbert.datasets.dataloading import MolbertDataLoader
from molbert.datasets.finetune import BertFinetuneSmilesDataset
from molbert.models.base import MolbertModel, MolbertBatchType
from molbert.tasks.tasks import BaseTask, FinetuneTask
from molbert.utils.featurizer.molfeaturizer import SmilesIndexFeaturizer
from molbert.utils.lm_utils import BertConfigExtras

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class FinetuneSmilesMolbertModel(MolbertModel):
    def get_config(self):
        if not hasattr(self.hparams, 'vocab_size') or not self.hparams.vocab_size:
            self.hparams.vocab_size = 42

        if self.hparams.tiny:
            config = BertConfigExtras(
                vocab_size_or_config_json_file=self.hparams.vocab_size,
                hidden_size=16,
                num_hidden_layers=2,
                num_attention_heads=2,
                intermediate_size=32,
                max_position_embeddings=self.hparams.max_position_embeddings,
                mode=self.hparams.mode,
                output_size=self.hparams.output_size,
                label_column=self.hparams.label_column,
            )
        else:
            config = BertConfigExtras(
                vocab_size_or_config_json_file=self.hparams.vocab_size,
                hidden_size=768,
                num_hidden_layers=12,
                num_attention_heads=12,
                intermediate_size=3072,
                max_position_embeddings=self.hparams.max_position_embeddings,
                mode=self.hparams.mode,
                output_size=self.hparams.output_size,
                label_column=self.hparams.label_column,
            )

        return config

    def get_tasks(self, config):
        """ Task list should be converted to nn.ModuleList before, not done here to hide params from torch """
        tasks: List[BaseTask] = [FinetuneTask(name='finetune', config=config)]

        return tasks

    def load_datasets(self):
        featurizer = SmilesIndexFeaturizer.bert_smiles_index_featurizer(self.hparams.max_seq_length)

        train_dataset = BertFinetuneSmilesDataset(
            input_path=self.hparams.train_file,
            featurizer=featurizer,
            single_seq_len=self.hparams.max_seq_length,
            total_seq_len=self.hparams.max_seq_length,
            label_column=self.hparams.label_column,
            is_same=False,
        )

        validation_dataset = BertFinetuneSmilesDataset(
            input_path=self.hparams.valid_file,
            featurizer=featurizer,
            single_seq_len=self.hparams.max_seq_length,
            total_seq_len=self.hparams.max_seq_length,
            label_column=self.hparams.label_column,
            is_same=False,
        )

        test_dataset = BertFinetuneSmilesDataset(
            input_path=self.hparams.test_file,
            featurizer=featurizer,
            single_seq_len=self.hparams.max_seq_length,
            total_seq_len=self.hparams.max_seq_length,
            label_column=self.hparams.label_column,
            is_same=False,
            inference_mode=True,
        )

        return {'train': train_dataset, 'valid': validation_dataset, 'test': test_dataset}

    def evaluate_metrics(self, batch_labels, batch_predictions) -> Dict[str, torch.Tensor]:

        if self.hparams.mode == 'classification':
            # transformers convention is to output classification as two neurons.
            # In order to convert this to a class label we take the argmax.
            probs = nn.Softmax(dim=1)(batch_predictions)
            preds = torch.argmax(probs, dim=1).squeeze()
            probs_of_positive_class = probs[:, 1]
            batch_labels = batch_labels.squeeze()
        else:
            preds = batch_predictions

        if self.hparams.mode == 'classification':
            metrics = {
                'AUROC': lambda: AUROC()(probs_of_positive_class, batch_labels),
                'AveragePrecision': lambda: AveragePrecision()(probs_of_positive_class, batch_labels),
                'Accuracy': lambda: Accuracy()(preds, batch_labels),
            }
        else:
            metrics = {
                'MAE': lambda: MAE()(preds, batch_labels),
                'RMSE': lambda: RMSE()(preds, batch_labels),
                'MSE': lambda: MSE()(preds, batch_labels),
                # sklearn metrics work the other way round metric_fn(y_true, y_pred)
                'R2': lambda: r2_score(batch_labels.cpu(), preds.cpu()),
            }

        out = {}
        for name, callable_metric in metrics.items():
            try:
                out[name] = callable_metric().item()
            except Exception as e:
                logger.info(f'unable to calculate {name} metric')
                logger.info(e)
                out[name] = np.nan

        return out

    def test_step(self, batch: MolbertBatchType, batch_idx: int) -> Dict[str, Dict[str, torch.Tensor]]:  # type: ignore
        """
        For a certain batch, performs a forward step and evaluates the losses
        Args:
            batch: Contains three components:
                - input dictionary for the batch with keys 'input_ids', 'token_type_ids' and 'attention_mask';
                - label dictionary of the expected outputs such as 'lm_label_ids', 'unmasked_lm_label_ids' and
                additional ones, depending on the tasks;
                - and an array of masks (should be all true) with the length of the true batch size
        """
        (batch_inputs, batch_labels), _ = batch
        y_hat = self.forward(batch_inputs)

        return dict(predictions=y_hat, labels=batch_labels)

    def test_epoch_end(
        self, outputs: List[Dict[str, Dict[str, torch.Tensor]]]
    ) -> Dict[str, Dict[str, torch.Tensor]]:  # type: ignore
        all_predictions = torch.cat([out['predictions']['finetune'] for out in outputs])
        all_predictions_dict = dict(finetune=all_predictions)
        all_labels = torch.cat([out['labels']['finetune'] for out in outputs])
        all_labels_dict = dict(finetune=all_labels)

        losses = self.evaluate_losses(all_labels_dict, all_predictions_dict)
        loss = torch.sum(torch.stack(list(losses.values())))

        # add metrics to the test set evaluation
        metrics = self.evaluate_metrics(all_labels_dict['finetune'], all_predictions_dict['finetune'])

        tensorboard_logs = {'test_loss': loss, **losses}
        metrics_path = os.path.join(os.path.dirname(self.trainer.ckpt_path), 'metrics.json')
        logger.info('writing test set metrics to', metrics_path)
        logger.info(metrics)
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        return {'loss': loss, 'metrics': metrics, 'test_loss': loss, 'log': tensorboard_logs}  # type: ignore

    def test_dataloader(self) -> DataLoader:
        """ load the test set in one large batch """
        dataset = self.datasets['test']
        return MolbertDataLoader(dataset, batch_size=1024, num_workers=self.hparams.num_workers)
