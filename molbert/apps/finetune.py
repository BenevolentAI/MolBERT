import os
import pprint
from argparse import ArgumentParser, Namespace

import yaml

from molbert.apps.base import BaseMolbertApp
from molbert.models.base import MolbertModel
from molbert.models.finetune import FinetuneSmilesMolbertModel


class FinetuneSmilesMolbertApp(BaseMolbertApp):
    @staticmethod
    def get_model(args) -> MolbertModel:
        pprint.pprint(args)
        model = FinetuneSmilesMolbertModel(args)
        pprint.pprint(model)

        model = BaseMolbertApp.load_model_weights(model=model, checkpoint_file=args.pretrained_model_path)

        if args.freeze_level != 0:
            print('Freezing base model')
            FinetuneSmilesMolbertApp.freeze_network(model, args.freeze_level)

        return model

    @staticmethod
    def freeze_network(model: MolbertModel, freeze_level: int):
        """
        Freezes specific layers of the model depending on the freeze_level argument:

         0: freeze nothing
        -1: freeze all BERT weights but not the task head
        -2: freeze the pooling layer
        -3: freeze the embedding layer
        -4: freeze the task head but not the base layer
        n>0: freeze the bottom n layers of the base model.
        """

        model_bert = model.model.bert
        model_tasks = model.model.tasks

        model_bert_encoder = model.model.bert.encoder
        model_bert_pooler = model.model.bert.pooler
        model_bert_embeddings = model.model.bert.embeddings

        if freeze_level == 0:
            # freeze nothing
            return

        elif freeze_level > 0:
            # freeze the encoder/transformer
            n_encoder_layers = len(model_bert_encoder.layer)

            # we'll always freeze layers bottom up - starting from layers closest to the embeddings
            frozen_layers = min(freeze_level, n_encoder_layers)
            #
            for i in range(frozen_layers):
                layer = model_bert_encoder.layer[i]
                for param in layer.parameters():
                    param.requires_grad = False

        elif freeze_level == -1:
            # freeze everything bert
            for param in model_bert.parameters():
                param.requires_grad = False

        elif freeze_level == -2:
            # freeze the pooling layer
            for param in model_bert_pooler.parameters():
                param.requires_grad = False

        elif freeze_level == -3:
            # freeze the embedding layer
            for param in model_bert_embeddings.parameters():
                param.requires_grad = False

        elif freeze_level == -4:
            # freeze the task head
            for param in model_tasks.parameters():
                param.requires_grad = False

    @staticmethod
    def add_parser_arguments(parser: ArgumentParser) -> ArgumentParser:
        """
        Adds model specific options to the default parser
        """
        parser.add_argument(
            '--train_file',
            type=str,
            required=True,
            help='Path to train dataset to use for finetuning. Must be csv file.',
        )
        parser.add_argument(
            '--valid_file',
            type=str,
            required=True,
            help='Path to validation dataset to use for finetuning. Must be csv file.',
        )
        parser.add_argument(
            '--test_file', type=str, required=True, help='Path to test dataset to use for finetuning. Must be csv file.'
        )
        parser.add_argument('--smiles_column', type=str, default='SMILES', help='Column in csv file containing SMILES.')
        parser.add_argument('--label_column', type=str, required=True, help='Column in csv file containing labels.')
        parser.add_argument(
            '--mode',
            type=str,
            required=True,
            help='regression or classification',
            choices=['regression', 'classification'],
        )
        parser.add_argument(
            '--pretrained_model_path', type=str, required=True, help='Path to pretrained Molbert model.'
        )
        parser.add_argument(
            '--output_size',
            type=int,
            required=True,
            help='Number of task output dimensions. 1 for regression, n_classes for classification',
        )
        parser.add_argument(
            '--freeze_level',
            type=int,
            default=0,
            help=""" Freezes specific layers of the model depending on the argument:
                                     0: freeze nothing
                                    -1: freeze ever BERT weight but not the task head
                                    -2: freeze the pooling layer
                                    -3: freeze the embedding layer
                                    -4: freeze the task head but not the base layer
                                   n>0: freeze the bottom n layers of the base model.""",
        )

        return parser

    def parse_args(self, args) -> Namespace:
        """
        Override base to insert default model specific arguments.
        """
        parsed_args = super().parse_args(args)

        model_dir = os.path.dirname(os.path.dirname(parsed_args.pretrained_model_path))
        hparams_path = os.path.join(model_dir, 'hparams.yaml')
        with open(hparams_path, 'r') as yaml_file:
            config_dict = yaml.load(yaml_file, Loader=yaml.FullLoader)

        # update model specific parameters
        parsed_args.tiny = config_dict['tiny']
        parsed_args.masked_lm = config_dict['masked_lm']
        parsed_args.is_same_smiles = config_dict['is_same_smiles']
        parsed_args.num_physchem_properties = config_dict['num_physchem_properties']

        return parsed_args


def main():
    FinetuneSmilesMolbertApp().run()


if __name__ == '__main__':
    main()
