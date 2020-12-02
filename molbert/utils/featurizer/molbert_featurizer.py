import logging
import os
from argparse import Namespace
from typing import Tuple, Sequence, Any, Dict, Union, Optional

import numpy as np
import torch
import yaml

from molbert.models.smiles import SmilesMolbertModel
from molbert.utils.featurizer.molfeaturizer import SmilesIndexFeaturizer

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class MolBertFeaturizer:
    """
    This featurizer takes a molbert model and transforms the input data and
    returns the representation in the last layer (pooled output and sequence_output).
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str = None,
        embedding_type: str = 'pooled',
        max_seq_len: Optional[int] = None,
        permute: bool = False,
    ) -> None:
        """
        Args:
            checkpoint_path: path or S3 location of trained model checkpoint
            device: device for torch
            embedding_type: method to reduce MolBERT encoding to an output set of features. Default: 'pooled'
                Other options are embeddings summed or concat across layers, and then averaged
                Raw sequence and pooled output is also available (set to 'dict')
                average-sum-[2|4], average-cat-[2,4], average-[1|2|3|4], average-1-cat-pooled, pooled, dict
            max_seq_len: used by the tokenizer, SMILES longer than this will fail to featurize
                MolBERT was trained with SuperPositionalEncodings (TransformerXL) to decoupled from the training setup
                By default the training config is used (128). If you have long SMILES to featurize, increase this value
        """
        super().__init__()
        self.checkpoint_path = checkpoint_path
        self.model_dir = os.path.dirname(os.path.dirname(checkpoint_path))
        self.hparams_path = os.path.join(self.model_dir, 'hparams.yaml')
        self.device = device or 'cuda' if torch.cuda.is_available() else 'cpu'
        self.embedding_type = embedding_type
        self.output_all = False if self.embedding_type in ['pooled'] else True
        self.max_seq_len = max_seq_len
        self.permute = permute

        # load config
        with open(self.hparams_path) as yaml_file:
            config_dict = yaml.load(yaml_file, Loader=yaml.FullLoader)

        logger.debug('loaded model trained with hparams:')
        logger.debug(config_dict)

        # load smiles index featurizer
        self.featurizer = self.load_featurizer(config_dict)

        # load model
        self.config = Namespace(**config_dict)
        self.model = SmilesMolbertModel(self.config)
        self.model.load_from_checkpoint(self.checkpoint_path, hparam_overrides=self.model.__dict__)

        # HACK: manually load model weights since they don't seem to load from checkpoint (PL v.0.8.5)
        checkpoint = torch.load(self.checkpoint_path, map_location=lambda storage, loc: storage)
        self.model.load_state_dict(checkpoint['state_dict'])

        self.model.eval()
        self.model.freeze()

        self.model = self.model.to(self.device)

        if self.output_all:
            self.model.model.config.output_hidden_states = True

    def __getstate__(self):
        self.__dict__.update({'model': self.model.to('cpu')})
        self.__dict__.update({'device': 'cpu'})
        return self.__dict__

    @property
    def output_size(self) -> int:
        return self.model.config.hidden_size

    def transform_single(self, smiles: str) -> Tuple[np.ndarray, bool]:
        features, valid = self.transform([smiles])
        return features, valid[0]

    def transform(self, molecules: Sequence[Any]) -> Tuple[Union[Dict, np.ndarray], np.ndarray]:
        input_ids, valid = self.featurizer.transform(molecules)

        input_ids = self.trim_batch(input_ids, valid)

        token_type_ids = np.zeros_like(input_ids, dtype=np.long)
        attention_mask = np.zeros_like(input_ids, dtype=np.long)

        attention_mask[input_ids != 0] = 1

        input_ids = torch.tensor(input_ids, dtype=torch.long, device=self.device)
        token_type_ids = torch.tensor(token_type_ids, dtype=torch.long, device=self.device)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long, device=self.device)

        with torch.no_grad():
            outputs = self.model.model.bert(
                input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask
            )

        if self.output_all:
            sequence_output, pooled_output, hidden = outputs
        else:
            sequence_output, pooled_output = outputs

        # set invalid outputs to 0s
        valid_tensor = torch.tensor(
            valid, dtype=sequence_output.dtype, device=sequence_output.device, requires_grad=False
        )

        pooled_output = pooled_output * valid_tensor[:, None]

        # concatenate and sum last 4 layers
        if self.embedding_type == 'average-sum-4':
            sequence_out = torch.sum(torch.stack(hidden[-4:]), dim=0)  # B x L x H
        # concatenate and sum last 2 layers
        elif self.embedding_type == 'average-sum-2':
            sequence_out = torch.sum(torch.stack(hidden[-2:]), dim=0)  # B x L x H
        # concatenate last four hidden layer
        elif self.embedding_type == 'average-cat-4':
            sequence_out = torch.cat(hidden[-4:], dim=-1)  # B x L x 4*H
        # concatenate last two hidden layer
        elif self.embedding_type == 'average-cat-2':
            sequence_out = torch.cat(hidden[-2:], dim=-1)  # B x L x 2*H
        # only last layer - same as default sequence output
        elif self.embedding_type == 'average-1':
            sequence_out = hidden[-1]  # B x L x H
        # only penultimate layer
        elif self.embedding_type == 'average-2':
            sequence_out = hidden[-2]  # B x L x H
        # only 3rd to last layer
        elif self.embedding_type == 'average-3':
            sequence_out = hidden[-3]  # B x L x H
        # only 4th to last layer
        elif self.embedding_type == 'average-4':
            sequence_out = hidden[-4]  # B x L x H
        # defaults to last hidden layer
        else:
            sequence_out = sequence_output  # B x L x H

        sequence_out = sequence_out * valid_tensor[:, None, None]

        sequence_out = sequence_out.detach().cpu().numpy()
        pooled_output = pooled_output.detach().cpu().numpy()

        if self.embedding_type == 'pooled':
            out = pooled_output
        elif self.embedding_type == 'average-1-cat-pooled':
            sequence_out = np.mean(sequence_out, axis=1)
            out = np.concatenate([sequence_out, pooled_output], axis=-1)
        elif self.embedding_type.startswith('average'):
            out = np.mean(sequence_out, axis=1)
        else:
            out = dict(sequence_output=sequence_out, pooled_output=pooled_output)

        return out, valid

    def load_featurizer(self, config_dict):
        # load smiles index featurizer
        if self.max_seq_len is None:
            max_seq_len = config_dict.get('max_seq_length')
            logger.debug('getting smiles index featurizer of length: ', max_seq_len)
        else:
            max_seq_len = self.max_seq_len
        return SmilesIndexFeaturizer.bert_smiles_index_featurizer(max_seq_len, permute=self.permute)

    @staticmethod
    def trim_batch(input_ids, valid):

        # trim input horizontally if there is at least 1 valid data point
        if any(valid):
            _, cols = np.where(input_ids[valid] != 0)
        # else trim input down to 1 column (avoids empty batch error)
        else:
            cols = np.array([0])

        max_idx: int = int(cols.max().item() + 1)

        input_ids = input_ids[:, :max_idx]

        return input_ids
