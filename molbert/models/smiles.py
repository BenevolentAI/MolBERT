import logging
from typing import List

from molbert.datasets.smiles import BertSmilesDataset
from molbert.models.base import MolbertModel
from molbert.tasks.tasks import MaskedLMTask, IsSameTask, PhyschemTask, BaseTask
from molbert.utils.lm_utils import get_seq_lengths, BertConfigExtras
from molbert.utils.featurizer.molfeaturizer import PhysChemFeaturizer, SmilesIndexFeaturizer

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class SmilesMolbertModel(MolbertModel):
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
                num_physchem_properties=self.hparams.num_physchem_properties,
                named_descriptor_set=self.hparams.named_descriptor_set,
                is_same_smiles=self.hparams.is_same_smiles,
            )
        else:
            config = BertConfigExtras(
                vocab_size_or_config_json_file=self.hparams.vocab_size,
                hidden_size=768,
                num_hidden_layers=12,
                num_attention_heads=12,
                intermediate_size=3072,
                max_position_embeddings=self.hparams.max_position_embeddings,
                num_physchem_properties=self.hparams.num_physchem_properties,
                named_descriptor_set=self.hparams.named_descriptor_set,
                is_same_smiles=self.hparams.is_same_smiles,
            )

        return config

    def get_tasks(self, config):
        """ Task list should be converted to nn.ModuleList before, not done here to hide params from torch """
        tasks: List[BaseTask] = []
        if self.hparams.masked_lm:
            tasks.append(MaskedLMTask(name='masked_lm', config=config))

        if self.hparams.is_same_smiles:
            tasks.append(IsSameTask(name='is_same', config=config))

        if self.hparams.num_physchem_properties > 0:
            config = self.check_physchem_output_size(config)
            tasks.append(PhyschemTask(name='physchem_props', config=config))

        return tasks

    def check_physchem_output_size(self, config):
        num_physchems_for_subset = PhysChemFeaturizer(named_descriptor_set=config.named_descriptor_set).output_size

        if config.num_physchem_properties > num_physchems_for_subset:
            logging.info(f'Setting num_physchem_properties to {num_physchems_for_subset}.')
            config.num_physchem_properties = num_physchems_for_subset
            self.hparams.num_physchem_properties = num_physchems_for_subset
        return config

    def load_datasets(self):
        single_seq_len, total_seq_len = get_seq_lengths(self.hparams.max_seq_length, self.hparams.is_same_smiles)

        featurizer = SmilesIndexFeaturizer.bert_smiles_index_featurizer(total_seq_len)

        train_dataset, valid_dataset, test_dataset = None, None, None

        train_dataset = BertSmilesDataset(
            input_path=self.hparams.train_file,
            featurizer=featurizer,
            single_seq_len=single_seq_len,
            total_seq_len=total_seq_len,
            is_same=self.hparams.is_same_smiles,
            num_physchem=self.hparams.num_physchem_properties,
            permute=self.hparams.permute,
            named_descriptor_set=self.hparams.named_descriptor_set,
        )

        if self.hparams.valid_file:
            valid_dataset = BertSmilesDataset(
                input_path=self.hparams.valid_file,
                featurizer=featurizer,
                single_seq_len=single_seq_len,
                total_seq_len=total_seq_len,
                is_same=self.hparams.is_same_smiles,
                num_physchem=self.hparams.num_physchem_properties,
                permute=self.hparams.permute,
                named_descriptor_set=self.hparams.named_descriptor_set,
            )

        if self.hparams.test_file:
            test_dataset = BertSmilesDataset(
                input_path=self.hparams.test_file,
                featurizer=featurizer,
                single_seq_len=single_seq_len,
                total_seq_len=total_seq_len,
                is_same=self.hparams.is_same_smiles,
                num_physchem=self.hparams.num_physchem_properties,
                permute=self.hparams.permute,
                named_descriptor_set=self.hparams.named_descriptor_set,
            )

        assert (
            self.hparams.vocab_size == train_dataset.featurizer.vocab_size
        ), f"{self.hparams.vocab_size} should equal {train_dataset.featurizer.vocab_size}"

        return {'train': train_dataset, 'valid': valid_dataset, 'test': test_dataset}
