import types
from typing import Optional, List, Union, Callable, Any, Tuple, Sequence

import higher
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from loguru import logger
from torch import nn, Tensor, optim
from torch.types import Device
from torch.utils.data import DataLoader

from ldt.meta.metrics import BaseMetric
from ldt.util.torchutil import cycle_dataloader, set_grads


def train_meta_step(
        student: nn.Module,
        device: Union[str, Device],
        train_loader: DataLoader,
        prepare_batch: Callable[[Any, Device], Tuple[Tensor, Tensor, Tensor]],
        inner_optimizer: optim.Optimizer,
        n_inner: int,
        criterion: Callable[[Tensor, Tensor], Tensor],
        teacher: nn.Module,
        compute_teaching_loss: Callable[[nn.Module, Tensor, Tensor, Any], Tensor],
        meta_opt: optim.Optimizer,
        meta_params: Sequence[Tensor],
        valid_loader: DataLoader,
        teaching_coef: Optional[float] = None,
        label_coef: float = 1.0,
        clip_meta_grad_value: Optional[float] = None,
        weight_norm_dim: Optional[int] = None,
        track_intermediate_losses: bool = False,
        additional_metrics: Optional[List[BaseMetric]] = None):
    """
    Take one step of updating the meta-parameters by unrolling an inner-loop
    and computing the meta-gradient.
    The actual student-parameters will be unchanged by this — they are only
    updated in the teacher's imagination.

    :param student: An nn.Module which, given input, returns prediction and
                    internal activations
    :param device: cpu or cuda
    :param train_loader: DataLoader which provides training batches
    :param prepare_batch: Method that can transform batches from dataloader
    :param inner_optimizer: Inner-loop optimizer
    :param n_inner: Number of inner steps
    :param criterion: Loss function between prediction and labels
    :param teacher: An nn.Module which, given privileged data, provides targets
                    for the student's internal activations
    :param compute_teaching_loss: Takes teacher's targets and student's
                                  activations and computes the loss tensor
    :param meta_opt: Optimizer for the meta-parameters
    :param meta_params: (Usually) the teacher's trainable weights
    :param valid_loader: DataLoader which provides validation batches
    :param teaching_coef: Weight of the teaching loss in the overall loss
    :param label_coef: Weight of the label loss in the overall loss
    :param clip_meta_grad_value: Gradient clipping for meta-gradients
    :param weight_norm_dim: Dimension of weight norm. (Needed because WN needs
                            to be patched manually. Ignored if no WN is used.
    :param track_intermediate_losses: Add metrics for inner-loop losses
    :param additional_metrics: Custom metrics to evaluate
    :return: Dictionary with meta-training metrics
    """

    if isinstance(meta_params, types.GeneratorType):
        meta_params = list(meta_params)

    assert len(meta_params) > 0

    student.train()
    teacher.train()

    meta_grads, train_metrics = compute_meta_grads(
        student, teacher, meta_params, compute_teaching_loss, inner_optimizer,
        n_inner, criterion, teaching_coef, label_coef,
        train_loader, valid_loader, prepare_batch, device,
        track_intermediate_losses, weight_norm_dim, additional_metrics)

    # Replace nans in meta grads with zeros (unless all of them are nan, in
    #  which case there is nothing to save)
    def replace_nan_with_zero(m):
        if not torch.all(torch.isnan(m)):
            m[torch.isnan(m)] = 0.0

    for m in meta_grads:
        replace_nan_with_zero(m)
        if clip_meta_grad_value is not None:
            m.clamp_(-clip_meta_grad_value, clip_meta_grad_value)

    set_grads(meta_params, meta_grads)

    meta_opt.step()

    logger.info(f'len(meta_grads): {len(meta_grads)}')

    # Compute and track meta-gradient magnitude
    squared_length = 0.0
    for meta_grad in meta_grads:
        squared_length += (meta_grad.detach() ** 2).sum()
    train_metrics['meta/grad_mag'] = torch.sqrt(squared_length).item()

    logger.info(f'meta/grad_mag: {train_metrics["meta/grad_mag"]}')

    return train_metrics


def compute_meta_grads(student, teacher, meta_params, compute_teaching_loss,
                       inner_optimizer, n_inner, criterion,
                       teaching_coef, label_coef,
                       train_loader, valid_loader, prepare_batch,
                       device,
                       track_intermediate_losses, weight_norm_dim,
                       additional_metrics: Optional[List[BaseMetric]]):
    """
    Computes the meta-gradients. See `train_meta_step` for arg description.
    :return: Tuple of meta-gradients and training metrics.
    """
    assert student.training, 'Please call student.train() before computing meta_grads'
    assert isinstance(meta_params, list), 'Please provide meta parameters as a list.'

    train_metrics = {}
    train_loss_sum = 0.0
    train_acc_sum = 0.0
    n_batches = 0

    f_student = higher.monkeypatch(student, device, copy_initial_weights=True)
    diff_inner_optimizer = higher.get_diff_optim(
        inner_optimizer,
        student.parameters(),
        fmodel=f_student,
        device=device,
        track_higher_grads=True
    )

    for batch_idx, batch in enumerate(
            cycle_dataloader(train_loader)):

        if batch_idx >= n_inner:
            break

        inputs, privileged_data, labels = prepare_batch(batch, device)

        # Patching weight norm
        for module in f_student.modules():
            if hasattr(module, 'weight_g'):
                module.weight = torch._weight_norm(module.weight_v,
                                                   module.weight_g,
                                                   weight_norm_dim)

        output, activations = f_student(inputs)

        if not isinstance(output, tuple):
            assert output.shape[0] == labels.shape[0]

        label_loss = criterion(output, labels)

        teaching_loss_output = compute_teaching_loss(
            teacher, inputs, privileged_data, activations)
        assert isinstance(teaching_loss_output, tuple)
        teaching_loss, teaching_metrics = teaching_loss_output

        if teaching_coef is not None:
            weighted_teaching_loss = teaching_coef * teaching_loss
        else:
            weighted_teaching_loss = teaching_loss

        loss = label_coef * label_loss + weighted_teaching_loss

        diff_inner_optimizer.step(loss)

        pred = _get_pred(output)
        if pred is not None:
            acc = (pred == labels.view_as(pred)).to(torch.float32).mean().item()
        else:
            acc = np.nan

        train_loss_sum += loss.item()
        train_acc_sum += acc
        n_batches += 1

        if track_intermediate_losses:
            valid_loss, valid_metrics = get_validation_loss(f_student, device,
                                                            valid_loader,
                                                            criterion,
                                                            prepare_batch,
                                                            additional_metrics)
            train_metrics[f'valid_loss_{batch_idx + 1}'] = valid_loss.item()
            step_valid_metrics = {m + f'_{batch_idx + 1}': v
                                  for m, v in valid_metrics.items()}
            train_metrics.update(step_valid_metrics)
            train_metrics[f'train/loss_{batch_idx}'] = loss.item()
            train_metrics[f'train/label_loss_{batch_idx}'] = label_loss.item()
            for key, value in teaching_metrics.items():
                train_metrics[f'{key}_{batch_idx}'] = float(value)

    if n_batches != n_inner:
        raise ValueError(f'Expected {n_inner} inner steps, but '
                         f'performed {n_batches}. Is the dataloader large '
                         f'enough?')

    if track_intermediate_losses:
        train_metrics[f'train/label_loss_{batch_idx + 1}'] = label_loss.item()

    meta_loss, valid_metrics = get_validation_loss(f_student, device,
                                                   valid_loader, criterion,
                                                   prepare_batch,
                                                   additional_metrics)

    train_metrics.update({
        'meta/training_loss': train_loss_sum / n_batches,
        'meta/last_step_label_loss': label_loss.item(),
        'meta/train_acc': train_acc_sum / n_batches,
        'meta/loss': meta_loss.item(),
    })
    train_metrics.update({'meta/' + m: v for m, v in valid_metrics.items()})

    meta_grads = torch.autograd.grad(meta_loss, meta_params)
    return meta_grads, train_metrics


def train_regular(student, device, train_loader, prepare_batch,
                  real_optimizer,
                  n_train_steps, criterion, teacher, compute_teaching_loss,
                  teaching_coef,
                  label_coef=1.0,
                  additional_metrics: Optional[List[BaseMetric]] = None
                  ):
    """
    Train the student parameters for a given number of steps, using the current
    version of the teacher.

    :param student: An nn.Module which, given input, returns prediction and
                    internal activations
    :param device: cpu or cuda
    :param train_loader: DataLoader which provides training batches
    :param prepare_batch: Method that can transform batches from dataloader
    :param real_optimizer: Student optimizer. In contrast to the inner-loop
                           optimizer, this one takes actual updates.
    :param n_train_steps: Number of training batches on which the student
                          will be trained.
    :param criterion: Loss function between prediction and labels
    :param teacher: An nn.Module which, given privileged data, provides targets
                    for the student's internal activations
    :param compute_teaching_loss: Takes teacher's targets and student's
                                  activations and computes the loss tensor
    :param teaching_coef: Weight of the teaching loss in the overall loss
    :param label_coef: Weight of the label loss in the overall loss
    :param additional_metrics: Custom metrics to evaluate
    :return: Dictionary with training metrics
    """
    student.train()
    metrics = {}
    loss_sum = 0.0
    label_loss_sum = 0.0
    acc_sum = 0.0
    n_batches = 0
    total = 0

    epoch_predictions = []
    epoch_labels = []
    epoch_privileged_data = []

    if additional_metrics is None:
        additional_metrics = []

    if teaching_coef is not None:
        teaching_coef = teaching_coef.detach()

    all_teaching_metrics = []

    for batch_idx, batch in enumerate(train_loader):
        if batch_idx >= n_train_steps:
            break

        inputs, privileged_data, labels = prepare_batch(batch, device)

        output, activations = student(inputs)

        if (criterion is F.cross_entropy
                or criterion is F.nll_loss
                or criterion.__name__ == 'regularized_cross_entropy'):
            assert labels.dtype is torch.long
            assert labels.ndim == 1
            assert output.shape[1] >= labels.max().item() + 1
            batch_size = labels.shape[0]
            assert output.shape[0] == batch_size
        elif criterion is F.binary_cross_entropy_with_logits:
            assert output.shape == labels.shape
            batch_size = labels.shape[0]
        elif criterion is F.mse_loss:
            assert output.shape == labels.shape
            batch_size = labels.shape[0]
        else:
            raise NotImplementedError('Make assertion for every criterion type!')

        label_loss = criterion(output, labels)

        if teacher is not None:
            teaching_loss_output = compute_teaching_loss(
                teacher, inputs, privileged_data, activations
            )
            assert isinstance(teaching_loss_output, tuple)
            teaching_loss, teaching_metrics = teaching_loss_output
            if teaching_metrics:
                all_teaching_metrics.append(teaching_metrics)

            if teaching_coef is None:
                logger.error('teaching_coefs None even though teacher is not None')
                weighted_teaching_loss = teaching_loss
            else:
                weighted_teaching_loss = teaching_coef * teaching_loss
        else:
            weighted_teaching_loss = 0.0

        loss = label_coef * label_loss + weighted_teaching_loss

        real_optimizer.zero_grad()

        loss.backward()
        real_optimizer.step()

        pred = _get_pred(output)
        if pred is not None:
            acc = (pred == labels.view_as(pred)).to(torch.float32).mean().item()
        else:
            acc = np.nan

        loss_sum += loss.item() * batch_size
        label_loss_sum += label_loss.item() * batch_size
        acc_sum += acc * batch_size
        n_batches += 1
        total += batch_size

        if torch.is_tensor(output):
            epoch_predictions.extend(output.detach().cpu().numpy())
        if torch.is_tensor(labels):
            epoch_labels.extend(labels.detach().cpu().numpy())
        if torch.is_tensor(privileged_data):
            epoch_privileged_data.extend(privileged_data.detach().cpu().numpy())

    if len(all_teaching_metrics) > 0:
        mean_teaching_metrics = pd.DataFrame(all_teaching_metrics).mean(axis=0).to_dict()
        for k, v in mean_teaching_metrics.items():
            metrics[f'teaching_metrics/{k}'] = v

    if n_batches != n_train_steps:
        logger.warning(f'Expected {n_train_steps} steps steps, but only '
                       f'performed {n_batches}. Is the dataloader large '
                       f'enough?')

    metrics['train/loss'] = loss_sum / total
    metrics['train/label_loss'] = label_loss_sum / total
    metrics['train/acc'] = acc_sum / total
    if epoch_predictions:
        mean_label_pred = (np.array(epoch_predictions) > 0).mean()
    else:
        mean_label_pred = np.nan
    metrics['train/label_prediction'] = mean_label_pred

    for additional_metric in additional_metrics:
        metrics['train/' + additional_metric.key] = additional_metric(
            epoch_predictions, epoch_labels, epoch_privileged_data)

    return metrics


def _get_pred(output):
    """Process model prediction depending on label type"""
    if isinstance(output, tuple):
        return None
    if output.ndim == 1:
        pred = (output > 0.0).to(torch.float32)
    elif output.ndim == 2:
        pred = torch.argmax(output, dim=1)
    else:
        raise ValueError()
    return pred


def get_validation_loss(model, device, valid_loader, criterion, prepare_batch,
                        additional_metrics):
    """
    Wrapper around _average_epoch_loss for validation
    """
    previous_model_train_state = model.training
    model.eval()
    valid_loss, metrics = _average_epoch_loss(model, valid_loader, criterion,
                                              prepare_batch, device,
                                              additional_metrics)
    model.train(previous_model_train_state)
    return valid_loss, {'valid_' + m: v for m, v in metrics.items()}


def run_test_epoch(model, device, test_loader, criterion, prepare_batch,
                   additional_metrics=None):
    """
    Wrapper around _average_epoch_loss for test
    """
    model.eval()
    with torch.no_grad():
        test_loss, metrics = _average_epoch_loss(model, test_loader, criterion,
                                                 prepare_batch, device,
                                                 additional_metrics)

    test_metrics = {'test/loss': test_loss}
    test_metrics.update({'test/' + m: v for m, v in metrics.items()})
    return test_metrics


def _average_epoch_loss(model, dataloader, criterion, prepare_batch, device,
                        additional_metrics):
    """
    Compute the loss of the model on the dataloader and return metrics.
    """
    if additional_metrics is None:
        additional_metrics = []

    loss_cuml = 0.0
    correct = 0
    total = 0
    epoch_predictions = []
    epoch_labels = []
    epoch_privileged_data = []
    metrics = dict()
    for batch in dataloader:
        inputs, privileged_data, labels = prepare_batch(batch, device)
        output, _ = model(inputs)
        if not isinstance(output, tuple):
            assert output.shape[0] == labels.shape[0]
        batch_loss = criterion(output, labels, reduction='sum')
        if not batch_loss.requires_grad:
            # In tests epoch, convert to float
            batch_loss = batch_loss.item()
        loss_cuml += batch_loss
        pred = _get_pred(output)
        if pred is not None:
            correct += pred.eq(labels.view_as(pred)).sum().item()

        if torch.is_tensor(labels):
            total += labels.shape[0]
        else:
            total += labels[0].shape[0]

        if torch.is_tensor(output):
            epoch_predictions.extend(output.detach().cpu().numpy())
        if torch.is_tensor(labels):
            epoch_labels.extend(labels.detach().cpu().numpy())
        if torch.is_tensor(privileged_data):
            epoch_privileged_data.extend(privileged_data.detach().cpu().numpy())

    loss = loss_cuml / total
    acc = correct / total

    for additional_metric in additional_metrics:
        metrics[additional_metric.key] = additional_metric(
            epoch_predictions, epoch_labels, epoch_privileged_data)

    if epoch_predictions:
        mean_label_pred = (np.array(epoch_predictions) > 0).mean()
    else:
        mean_label_pred = np.nan
    metrics.update({'acc': acc,
                    'correct': correct,
                    'label_prediction': mean_label_pred
                    })
    return loss, metrics
