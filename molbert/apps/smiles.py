from argparse import ArgumentParser

from molbert.apps.base import BaseMolbertApp
from molbert.models.base import MolbertModel
from molbert.models.smiles import SmilesMolbertModel


class SmilesMolbertApp(BaseMolbertApp):
    @staticmethod
    def get_model(args) -> MolbertModel:
        return SmilesMolbertModel(args)

    @staticmethod
    def add_parser_arguments(parser: ArgumentParser) -> ArgumentParser:
        """
        Adds model specific options to the default parser
        """
        parser.add_argument(
            '--num_physchem_properties', type=int, default=0, help='Adds physchem property task (how many to predict)'
        )
        parser.add_argument('--is_same_smiles', type=int, default=0, help='Adds is_same_smiles task')
        parser.add_argument('--permute', type=int, default=0, help='Permute smiles')
        parser.add_argument(
            '--named_descriptor_set', type=str, default='all', help='What set of descriptors to use ("all" or "simple")'
        )
        parser.add_argument('--vocab_size', default=42, type=int, help='Vocabulary size for smiles index featurizer')
        return parser


if __name__ == '__main__':
    SmilesMolbertApp().run()
