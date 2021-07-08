import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F
from torch import optim

from ldt.meta.training import train_meta_step, compute_meta_grads
from tests.meta_test_helpers import MinimalTeacher, MinimalStudent, prepare_batch_minimal, \
    minimal_dataloaders, compute_meta_grads_fd, compute_teaching_loss_minimal, filter_fd_series


@pytest.mark.parametrize('train_batch_size,n_inner',
                         [(1, 2)])
@pytest.mark.parametrize('optim_cls,inner_lr',
                         [(optim.SGD, 1e-2)])
@pytest.mark.parametrize('inner_momentum',
                         [0.0, 0.9])
@pytest.mark.parametrize('device',
                         ['cpu',  # 'cuda',
                         ])
@pytest.mark.parametrize('teaching_coef',
                         [1e0, 1e1])
def test_meta_grads_equal_finite_difference(train_batch_size, n_inner,
                                            optim_cls, inner_lr,
                                            inner_momentum,
                                            device,
                                            teaching_coef):
    """Test whether the meta gradients are close to finite difference gradients"""
    train_loader, valid_loader = minimal_dataloaders(train_batch_size)
    teacher = MinimalTeacher()
    teacher.to(device)
    student = MinimalStudent()
    student.to(device)
    inner_lr = torch.tensor(inner_lr)
    prepare_batch = prepare_batch_minimal

    def make_optimizer(params):
        if optim_cls is optim.SGD:
            return optim.SGD(params, lr=inner_lr, momentum=inner_momentum)
        elif optim_cls is optim.Adam:
            return optim.Adam(params, lr=inner_lr, betas=(inner_momentum, 0.999))
        else:
            raise ValueError()

    inner_optimizer = make_optimizer(student.parameters())
    label_coef = 1.0

    loss_fn = F.mse_loss
    teaching_coef = torch.tensor(teaching_coef)
    meta_params = list(teacher.parameters())

    student.train()
    meta_grads_ag, train_metrics = compute_meta_grads(
        student, teacher, meta_params=meta_params,
        compute_teaching_loss=compute_teaching_loss_minimal,
        inner_optimizer=inner_optimizer, n_inner=n_inner,
        criterion=loss_fn,
        teaching_coef=teaching_coef,
        label_coef=label_coef,
        train_loader=train_loader, valid_loader=valid_loader,
        prepare_batch=prepare_batch, device=device,
        track_intermediate_losses=False,
        weight_norm_dim=None,
        additional_metrics=None
    )

    # To get the "proper" finite difference, sweep over many orders of magnitude
    # for epsilon and then aggregate.
    fd_data = []
    for epsilon in 10 ** np.linspace(-7, 1, num=500):
        meta_grads_fd = compute_meta_grads_fd(
            student, teacher,
            compute_teaching_loss=compute_teaching_loss_minimal,
            make_optimizer=make_optimizer,
            n_inner=n_inner,
            inner_lr=inner_lr,
            loss_fn=loss_fn,
            teaching_coef=teaching_coef,
            label_coef=label_coef,
            train_loader=train_loader, valid_loader=valid_loader,
            prepare_batch=prepare_batch, device=device,
            epsilon=epsilon,
        )

        for i_param, param_grad in enumerate(meta_grads_fd):
            for i_elem, grad in enumerate(param_grad):
                fd_data.append({
                    'epsilon': epsilon,
                    'i_param': i_param,
                    'i_elem': i_elem,
                    'grad': grad,
                })
    meta_grads_ag = [g.detach().cpu().numpy().ravel().tolist()
                     for g in meta_grads_ag]

    fd_df = pd.DataFrame(fd_data)

    # To find the best numeric gradient, filter out zeros first.
    for param_elem, sub_df in fd_df.groupby(by=['i_param', 'i_elem']):
        i_param, i_elem = param_elem
        fd_series = sub_df.grad.values
        fd_series = filter_fd_series(fd_series, discard_frac=0.5, iterations=5)
        fd_derivative = np.median(fd_series)

        flat_ag_grads = meta_grads_ag[i_param]
        ag_derivative = flat_ag_grads[i_elem]
        assert ag_derivative == pytest.approx(fd_derivative, rel=1e-3)



def test_minimal_meta():
    """ Tiny sanity check """
    train_loader, valid_loader = minimal_dataloaders(train_batch_size=2)
    teacher = MinimalTeacher()
    student = MinimalStudent()
    inner_lr = torch.tensor(1e-1)
    prepare_batch = prepare_batch_minimal
    inner_optimizer = optim.SGD(student.parameters(), lr=inner_lr,
                                momentum=0.0, weight_decay=0.0)

    loss_fn = F.mse_loss
    teaching_coef = None

    meta_opt = optim.SGD(teacher.parameters(), lr=0.1, momentum=0.0, weight_decay=0.0)

    for _ in range(1):
        train_meta_step(student, device='cpu',
                        train_loader=train_loader,
                        prepare_batch=prepare_batch,
                        inner_optimizer=inner_optimizer,
                        n_inner=1,
                        criterion=loss_fn,
                        teacher=teacher,
                        compute_teaching_loss=compute_teaching_loss_minimal,
                        meta_opt=meta_opt,
                        meta_params=teacher.parameters(),
                        valid_loader=valid_loader,
                        teaching_coef=teaching_coef,
                        label_coef=1.0,
                        clip_meta_grad_value=None,
                        weight_norm_dim=None,
                        track_intermediate_losses=False,
                        additional_metrics=None
                        )
    assert student.theta1.data.tolist() == [1., 1.]
    assert student.theta2.data.tolist() == [[0.25], [0.25]]
    assert teacher.phi[0] > 1.0
    assert teacher.phi[1] < 1.0

