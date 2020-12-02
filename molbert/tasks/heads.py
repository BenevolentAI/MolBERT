from torch import nn


class IsSameHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.is_same_clf = nn.Sequential(
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, 2),
        )

    def forward(self, pooled_output):
        return self.is_same_clf(pooled_output)


class PhysChemHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.physchem_clf = nn.Sequential(
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.num_physchem_properties),
        )

    def forward(self, pooled_output):
        return self.physchem_clf(pooled_output)


class FinetuneHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.finetune_net = nn.Sequential(nn.Linear(config.hidden_size, config.output_size))

    def forward(self, pooled_output):
        return self.finetune_net(pooled_output)
