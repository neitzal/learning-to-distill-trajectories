from copy import deepcopy
from functools import partial

import numpy as np
import torch
from experiment_utils.miscutils.misc import even_integer_partition
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from ldt.meta.training import train_regular, get_validation_loss


def minimal_dataloaders(train_batch_size):
    train_x = torch.tensor([[1.0, 1.0], [-1.0, -1.0]])
    train_x_priv = torch.tensor([[1.0], [-1.0]])
    train_y = torch.tensor([[1.0], [-1.0]])
    train_loader = DataLoader(TensorDataset(train_x, train_x_priv, train_y),
                              batch_size=train_batch_size,
                              shuffle=False)
    valid_x = torch.tensor([[1.0, -1.0], [-1.0, +1.0]])
    valid_y_priv = torch.tensor([[1.0], [-1.0]])
    valid_y = torch.tensor([[1.0], [-1.0]])
    valid_loader = DataLoader(TensorDataset(valid_x, valid_y_priv, valid_y),
                              batch_size=2,
                              shuffle=False)
    return train_loader, valid_loader


class MinimalTeacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.phi = nn.Parameter(data=torch.tensor([1., 1.]), requires_grad=True)

    def forward(self, x):
        return x * self.phi


class MinimalStudent(nn.Module):
    def __init__(self):
        super().__init__()
        self.theta1 = nn.Parameter(data=torch.tensor([1., 1.]), requires_grad=True)
        self.theta2 = nn.Parameter(data=torch.tensor([[1. / 4], [1. / 4]]), requires_grad=True)

    def forward(self, x):
        h = x * self.theta1
        y = torch.matmul(h, self.theta2)
        return y, h


def compute_teaching_loss_minimal(teacher, pre_states, trajectory_data, activations):
    teaching_signal = teacher(trajectory_data)
    teaching_metrics = dict()
    # return torch.mean(torch.abs(activations - teaching_signal)), teaching_metrics
    return F.mse_loss(activations, teaching_signal), teaching_metrics


def prepare_batch_minimal(batch, device):
    return [b.to(device) for b in batch]


def compute_meta_grads_fd(student, teacher,
                          compute_teaching_loss,
                          make_optimizer,
                          n_inner,
                          inner_lr,
                          loss_fn,
                          device,
                          prepare_batch,
                          label_coef,
                          teaching_coef,
                          train_loader,
                          valid_loader,
                          epsilon,
                          component_subsets=None
                          ):
    """
    component_subsets: Dictionary parameter_idx -> set(gradient_components).
                       If specified, only these gradient components will be
                       evaluated.
    """

    meta_gradient_shapes = [p.shape for p in teacher.parameters()]

    meta_grads_fd = []

    compute_perturbed_loss_here = partial(
        compute_perturbed_loss,
        student=student, teacher=teacher,
        compute_teaching_loss=compute_teaching_loss,
        make_optimizer=make_optimizer,
        inner_lr=inner_lr,
        n_inner=n_inner,
        loss_fn=loss_fn,
        teaching_coef=teaching_coef,
        label_coef=label_coef,
        train_loader=train_loader,
        valid_loader=valid_loader,
        device=device,
        prepare_batch=prepare_batch)

    unperturbed_loss = compute_perturbed_loss_here(
        grad_idx=None, perturb_idx=None, epsilon=0)
    for grad_idx, shape in enumerate(meta_gradient_shapes):
        grad_estimate = []
        for perturb_idx in range(np.prod(shape)):

            # Skip computation if index is not requested
            if (component_subsets is not None
                    and perturb_idx not in component_subsets[grad_idx]):
                grad_estimate.append(None)
                continue

            perturbed_loss = compute_perturbed_loss_here(
                grad_idx=grad_idx, perturb_idx=perturb_idx, epsilon=epsilon)
            grad_estimate.append((perturbed_loss - unperturbed_loss) / epsilon)
        meta_grads_fd.append(grad_estimate)

    return meta_grads_fd

    # # Visualization
    # all_grads.append({'epsilon': epsilon,
    #                   'grad': meta_grads_fd[0][1]})
    # df = pd.DataFrame(all_grads)
    # ax = df.plot(x='epsilon', y='grad', logx=True, marker='o', markersize=2, lw=0.5)
    # ax.axhline(0.07, color='gray', ls='-', alpha=0.5)
    # # plt.plot(all_grads, marker='o')
    # plt.show()


def perturb_weights(model, idx, perturb_idx, epsilon):
    flat_params = list(model.parameters())[idx].data.view(-1)
    flat_params[perturb_idx] += epsilon


def compute_perturbed_loss(student, teacher,
                           compute_teaching_loss,
                           make_optimizer,
                           inner_lr,
                           n_inner,
                           loss_fn,
                           teaching_coef,
                           label_coef,
                           train_loader, valid_loader,
                           device,
                           prepare_batch,
                           grad_idx, perturb_idx, epsilon):

    student = deepcopy(student)
    teacher = deepcopy(teacher)
    teaching_coef = deepcopy(teaching_coef)
    train_loader = deepcopy(train_loader)
    valid_loader = deepcopy(valid_loader)
    real_optimizer = make_optimizer(student.parameters())

    # Perturbation
    if grad_idx is not None:
        perturb_weights(teacher, grad_idx, perturb_idx, epsilon)

    metrics = train_regular(
        student, device=device,
        train_loader=train_loader,
        prepare_batch=prepare_batch,
        real_optimizer=real_optimizer,
        n_train_steps=n_inner,
        criterion=loss_fn,
        teacher=teacher,
        compute_teaching_loss=compute_teaching_loss,
        teaching_coef=teaching_coef,
        label_coef=label_coef,
        additional_metrics=None
    )

    loss, _ = get_validation_loss(student, device, valid_loader, loss_fn,
                                  prepare_batch, additional_metrics=None)
    return loss.item()



def filter_fd_series(fd_series, discard_frac=0.5, iterations=5):
    """

    Filter out zeros, then take the median and remove outliers repeatedly.
    Probably not the best method. We should instead find the "mode", but
    with continuous data. I.e. the region with the highest density.
    """
    # Remove zeros if there are not too many
    if np.isclose(fd_series, 0).sum() < 0.8 * discard_frac * len(fd_series):
        fd_series = fd_series[~np.isclose(fd_series, 0)]

    discard_n = int(round(discard_frac * len(fd_series)))
    if discard_n <= iterations:
        raise ValueError('Too few elements!')


    discard_steps = even_integer_partition(discard_n, iterations)
    for discard in discard_steps:
        median = np.median(fd_series)
        closest_idxs = np.argsort(np.abs(fd_series - median))
        fd_series = fd_series[closest_idxs[:-discard]]

    return fd_series
