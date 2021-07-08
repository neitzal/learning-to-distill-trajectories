import numpy as np
import torch
from torch import nn as nn
from torch.utils.data import TensorDataset


def make_synth_dataset(n_examples, input_dim, initial_transform, seed):
    rng = np.random.RandomState(seed)
    inputs = rng.randn(n_examples, input_dim).astype(np.float32)
    inputs = torch.from_numpy(inputs)

    # assert relevant_dim % 2 == 1, 'relevant_dim should be odd (such that labels are balanced)'
    hidden = initial_transform(inputs).detach()
    assert hidden.shape == (n_examples, 2)
    labels = (hidden >= 0).sum(dim=1) == 1  # XOR
    labels = labels.to(torch.float32)

    # Transform hidden data:
    switches = rng.randint(0, 2, (n_examples, hidden.shape[1])).astype(np.float32)
    switches = 2 * switches - 1
    switches = torch.from_numpy(switches)
    trajectory_data = torch.cat((hidden * switches, switches), dim=1)

    return TensorDataset(inputs, trajectory_data, labels)


def make_datasets(input_dim, n_train, n_test_examples,
                  test_set_seed, train_set_seed):
    initial_transform = nn.Linear(input_dim, 2, bias=False)
    train_ds = make_synth_dataset(n_train, input_dim,
                                  initial_transform,
                                  seed=train_set_seed)
    test_ds = make_synth_dataset(n_test_examples, input_dim, initial_transform,
                                 seed=test_set_seed)
    assert not any(t.requires_grad for t in train_ds.tensors)
    return train_ds, test_ds
