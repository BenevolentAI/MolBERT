from dataclasses import dataclass
import torch
from molbert.tasks.tasks import PhyschemTask, MaskedLMTask, IsSameTask, FinetuneTask


@dataclass
class FakeConfig:
    mode: str
    hidden_size: int = 16
    output_size: int = 1
    hidden_dropout_prob: float = 0.25
    num_physchem_properties: int = 200
    vocab_size: int = 2
    hidden_act: str = 'relu'
    layer_norm_eps: float = 1e-6


def test_tasks():
    batch_size = 16
    batch_labels = dict(
        lm_label_ids=torch.ones((batch_size, 1), dtype=torch.long),
        is_same=torch.ones((batch_size, 1), dtype=torch.long),
        physchem=5 * torch.ones((batch_size, 200)),
        finetune_reg=7 * torch.ones((batch_size, 1)),
        finetune_cls=torch.ones((batch_size, 1), dtype=torch.long),
    )

    # fake predictions that all have softmax values of [0, 1]
    masked_lm_predictions = torch.zeros((batch_size, 2))
    masked_lm_predictions[:, 0] = 1

    batch_predictions = dict(
        masked_lm=masked_lm_predictions,
        is_same=torch.ones((batch_size, 2)),
        physchem=5 * torch.ones((batch_size, 200)),
        finetune_reg=7 * torch.ones((batch_size, 1)),
        finetune_cls=torch.ones((batch_size, 2)),
    )

    tasks = [
        FinetuneTask('finetune_reg', FakeConfig(mode='regression')),
        FinetuneTask('finetune_cls', FakeConfig(mode='classification', output_size=2)),
        PhyschemTask('physchem', FakeConfig(mode='regression')),
        MaskedLMTask('masked_lm', FakeConfig(mode='regression')),
        IsSameTask('is_same', FakeConfig(mode='regression')),
    ]

    seq_len = 10
    hidden_size = 16
    fake_sequence_output = torch.ones((batch_size, seq_len, hidden_size))
    fake_pooled_output = torch.ones((batch_size, hidden_size))

    for task in tasks:
        output = task(fake_sequence_output, fake_pooled_output)
        # just checks that the forward works without error and returns something
        assert output is not None
        loss = task.compute_loss(batch_labels, batch_predictions)

        # assert that loss is one dimensional
        assert loss.shape == torch.Size([])
