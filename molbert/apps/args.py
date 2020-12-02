import argparse


def get_default_parser():
    parser = argparse.ArgumentParser(conflict_handler='resolve')

    parser.add_argument('--train_file', default=None, type=str, required=True, help='The input train corpus.')
    parser.add_argument('--valid_file', default=None, type=str, required=False, help='The input validation corpus.')
    parser.add_argument('--test_file', default=None, type=str, required=False, help='The input test corpus.')
    parser.add_argument(
        '--default_root_dir',
        default=None,
        type=str,
        help='The output directory where the model checkpoints will be written. ' 'Default: cwd',
    )
    parser.add_argument(
        '--max_seq_length',
        default=128,
        type=int,
        help='The maximum total input sequence length after WordPiece tokenization. \n'
        'Sequences longer than this will be truncated, and sequences shorter \n'
        'than this will be padded.',
    )
    parser.add_argument('--batch_size', default=32, type=int, help='Total batch size for training.')
    parser.add_argument('--learning_rate', default=3e-5, type=float, help='The initial learning rate for Adam.')
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument(
        '--warmup_proportion',
        default=0.1,
        type=float,
        help='Proportion of training to perform linear learning rate warmup for. ' 'E.g., 0.1 = 10%% of training.',
    )
    parser.add_argument(
        "--learning_rate_scheduler",
        default='linear_with_warmup',
        type=str,
        help="Options: linear_with_warmup, cosine_annealing_warm_restarts, reduce_on_plateau, constant",
    )
    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight decay if we apply some.")
    parser.add_argument(
        '--accumulate_grad_batches',
        type=int,
        default=1,
        help='Accumulates grads every k batches or as set up in the dict.',
    )
    parser.add_argument('--gpus', type=int, default=0, help="How many GPUs to train on")
    parser.add_argument(
        '--distributed_backend',
        type=str,
        default=None,
        help="The distributed backend to use (dp, ddp, ddp2, ddp_spawn, ddp_cpu)",
    )
    parser.add_argument(
        '--amp_level',
        type=str,
        default='O2',
        help="For fp16: Apex AMP optimization level selected in ['None', 'O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument('--precision', type=int, default=32, help="Full precision (32), half precision (16).")
    parser.add_argument('--num_nodes', type=int, default=1, help="Number of GPU nodes for distributed training")
    parser.add_argument(
        '--tpu_cores',
        type=str,
        default=None,
        help="How many TPU cores to train on (1 or 8) / Single TPU to train on [1]",
    )
    parser.add_argument('--masked_lm', default=1, type=int, help='Whether to use the masked lm task.')
    parser.add_argument('--tiny', action='store_true', help='Tiny model for debugging')
    parser.add_argument(
        '--val_check_interval',
        default=0.25,
        type=float,
        help='How often within one training epoch to check the validation set',
    )
    parser.add_argument(
        '--limit_val_batches',
        default=1.0,
        type=float,
        help='How much of validation dataset to check (floats = percent, int = num_batches)',
    )
    parser.add_argument(
        '--resume_from_checkpoint',
        type=str,
        default=None,
        help='To resume training from a specific checkpoint pass in the path here. This can be a URL.',
    )
    parser.add_argument('--min_epochs', type=int, default=1, help='Minimum number of epochs')
    parser.add_argument('--max_epochs', type=int, default=20, help='Maximum number of epochs')
    parser.add_argument('--num_workers', default=0, type=int, help='Number of processes to use for Dataloader')
    parser.add_argument('--max_position_embeddings', default=512, type=int, help='Max size of positional embeddings')
    parser.add_argument('--deterministic', type=int, default=0, help='Disable cuDNN optimisations')
    parser.add_argument(
        '--fast_dev_run',
        type=int,
        default=0,
        help='If set to 1: runs 1 batch of train, test and val to find any bugs ' '(ie: a sort of unit test).',
    )
    parser.add_argument(
        '--progress_bar_refresh_rate',
        default=25,
        type=int,
        help='How often to refresh progress bar (in steps). Value ``0`` disables progress bar. '
        'Ignored when a custom callback is passed to :paramref:`~Trainer.callbacks`.',
    )
    parser.add_argument('--seed', default=42, type=int, help='Seed for random initialisation')

    return parser


def parse_args():
    parser = get_default_parser()
    args = parser.parse_args()
    return args
