import torch
from torch.distributions import Categorical
from torch.nn import functional as F


def prepare_batch(batch, device):
    inputs, privileged_data, labels = batch
    inputs = inputs.to(device, non_blocking=True)
    privileged_data = privileged_data.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True, dtype=torch.long)
    return inputs, privileged_data, labels


def compute_teaching_loss(teacher,
                          inputs,
                          privileged_data,
                          activations):
    if teacher is None:
        return 0.0, dict()

    target_logits = teacher(privileged_data)

    assert target_logits.shape == activations.shape, (
        f'target_logits.shape={target_logits.shape}, '
        f'activations.shape={activations.shape}')

    target_dist = Categorical(logits=target_logits)
    student_dist = Categorical(logits=activations)

    teaching_loss = torch.distributions.kl.kl_divergence(target_dist,
                                                         student_dist).mean()

    assert teaching_loss.shape == ()

    teaching_metrics = dict()
    return teaching_loss, teaching_metrics


