import numpy as np
import torch
from torch import nn as nn
from torch.distributions import Categorical
from torch.utils.data import TensorDataset


def make_synth_stochastic_dataset(n_examples, input_dim, initial_transform,
                                  privileged_transform,
                                  n_classes, privileged_dim, seed):
    rng = np.random.RandomState(seed)
    inputs = rng.randn(n_examples, input_dim).astype(np.float32)
    inputs = torch.from_numpy(inputs)

    hidden = initial_transform(inputs).detach()
    assert hidden.shape == (n_examples, n_classes)

    scale_logit_factor = 3.0
    hidden = scale_logit_factor * hidden

    labels = Categorical(logits=hidden).sample()
    labels = labels.to(torch.long)

    privileged_data = privileged_transform(hidden).detach()
    return TensorDataset(inputs, privileged_data, labels)


def make_datasets(input_dim, n_classes,
                  privileged_transform,
                  n_train,
                  n_test_examples,
                  test_set_seed,
                  train_set_seed):
    initial_transform = nn.Linear(input_dim, n_classes, bias=False)
    train_ds = make_synth_stochastic_dataset(n_train, input_dim,
                                             initial_transform, privileged_transform,
                                             n_classes, privileged_dim=n_classes,
                                             seed=train_set_seed)
    test_ds = make_synth_stochastic_dataset(n_test_examples, input_dim,
                                            initial_transform, privileged_transform,
                                            n_classes, privileged_dim=n_classes,
                                            seed=test_set_seed)
    assert not any(t.requires_grad for t in train_ds.tensors)
    return train_ds, test_ds
