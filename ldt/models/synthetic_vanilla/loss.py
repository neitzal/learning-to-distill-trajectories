import torch
from torch.nn import functional as F


def prepare_batch(batch, device):
    inputs, privileged_data, labels = batch
    inputs = inputs.to(device, non_blocking=True)
    privileged_data = privileged_data.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True, dtype=torch.float32)
    return inputs, privileged_data, labels


def compute_teaching_loss(teacher,
                          inputs,
                          privileged_data,
                          activations):
    if teacher is None:
        return 0.0, dict()

    target_representations = teacher(privileged_data)

    assert target_representations.shape == activations.shape, (
        f'target_representations.shape={target_representations.shape}, '
        f'activations.shape={activations.shape}')

    teaching_loss = F.mse_loss(activations, target_representations)
    assert teaching_loss.shape == ()

    teaching_metrics = dict()
    return teaching_loss, teaching_metrics
