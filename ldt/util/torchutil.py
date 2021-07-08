from itertools import zip_longest

import torch
import torch.nn.functional as F


def set_grads(params, grads):
    for p, g in zip_longest(params, grads):
        assert p.shape == g.shape
        p.grad = g


def cycle_dataloader(dl):
    while True:
        for x in dl:
            yield x


def make_entropy_regularized_CE(entropy_reg_coef):
    def regularized_cross_entropy(input, target, *args, **kwargs):
        if entropy_reg_coef > 0.0:
            logp = F.log_softmax(input)
            entropy = (torch.exp(logp) * logp).sum(dim=1).mean()
        else:
            entropy = 0.0
        return (F.cross_entropy(input, target, *args, **kwargs)
                + entropy_reg_coef*entropy)

    return regularized_cross_entropy
