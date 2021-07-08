import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import TensorDataset, DataLoader

from ldt.meta.training import compute_meta_grads
from ldt.models.synthetic_vanilla.loss import compute_teaching_loss, prepare_batch
from ldt.models.synthetic_vanilla.student import StudentModel
from ldt.models.synthetic_vanilla.teacher import Teacher
from tests.meta_test_helpers import compute_meta_grads_fd


def visualize_meta_gradient():
    """
    Make more realistic setup and check if the meta-gradient passes certain
    sanity checks. A complete finite differencing would be too expensive to
    compute, but we can perform it on several selected elements.

     - check if loss is decreasing in direction of gradient, at roughly the rate
       claimed by the gradient
     - check if for random directions orthogonal to the gradient, the loss
       is not changing in the first order
     - finite difference on a small, random subset of weights
        (reuse previous fd code!)
    """
    rng = np.random.default_rng()
    device = 'cpu'
    input_dim = 16
    privileged_dim = 7
    hidden_dim = 32
    teacher_hidden_dim = 64

    batch_size = 16
    n_batches = 5
    inner_lr = 1e-2
    inner_momentum = 0.0
    n_inner = n_batches

    loss_fn = F.mse_loss

    teaching_coef = 1e2

    label_coef = 1.0

    optim_cls = optim.SGD
    student = StudentModel(input_dim , hidden_dim)
    teacher = Teacher(privileged_dim=privileged_dim, hidden_dim=hidden_dim,
                      teacher_hidden_dim=teacher_hidden_dim)

    train_set = get_random_dataset(batch_size, input_dim, n_batches, privileged_dim, rng)
    valid_set = get_random_dataset(batch_size, input_dim, n_batches, privileged_dim, rng)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False)

    def make_optimizer(params):
        if optim_cls is optim.SGD:
            return optim.SGD(params, lr=inner_lr, momentum=inner_momentum)
        elif optim_cls is optim.Adam:
            return optim.Adam(params, lr=inner_lr, betas=(inner_momentum, 0.999))
        else:
            raise ValueError()

    inner_optimizer = make_optimizer(student.parameters())

    meta_params = list(teacher.parameters())

    student.train()
    meta_grads, train_metrics = compute_meta_grads(
        student, teacher, meta_params=meta_params,
        compute_teaching_loss=compute_teaching_loss,
        inner_optimizer=inner_optimizer,
        n_inner=n_inner,
        criterion=loss_fn,
        teaching_coef=teaching_coef,
        label_coef=label_coef,
        train_loader=train_loader,
        valid_loader=valid_loader,
        prepare_batch=prepare_batch,
        device=device,
        track_intermediate_losses=False,
        weight_norm_dim=None,
        additional_metrics=None
    )

    component_subsets = {
        0: {7*64 // 2},
        1: {64 // 2},
        2: {64 * 64 // 2},
        3: {64 // 2},
        4: {64*32 // 2},
        5: {32 // 2},
    }
    #

    fd_data = []
    for epsilon in 10 ** np.linspace(-6, 1, num=20):
        meta_grads_fd = compute_meta_grads_fd(
            student, teacher,
            compute_teaching_loss=compute_teaching_loss,
            make_optimizer=make_optimizer,
            n_inner=n_inner,
            inner_lr=inner_lr,
            loss_fn=loss_fn,
            teaching_coef=teaching_coef,
            label_coef=label_coef,
            train_loader=train_loader, valid_loader=valid_loader,
            prepare_batch=prepare_batch,
            device=device,
            epsilon=epsilon,
            component_subsets=component_subsets
        )
        # assert isinstance(meta_grads_fd, tuple)
        # assert len(meta_grads_fd) == 1
        # meta_grads_fd = meta_grads_fd[0]

        for i_param, param_grad in enumerate(meta_grads_fd):
            for i_elem, grad in enumerate(param_grad):
                if grad is not None:
                    fd_data.append({
                        'epsilon': epsilon,
                        'i_param': i_param,
                        'i_elem': i_elem,
                        'grad': grad,
                    })

    ad_data = []
    for i_param, meta_grad in enumerate(meta_grads):
        for i_elem, grad in enumerate(meta_grad.view(-1)):
            if i_elem in component_subsets[i_param]:
                ad_data.append({'i_param': i_param,
                                'i_elem': i_elem,
                                'grad': grad.item()})
    ad_df = pd.DataFrame(ad_data)



    fd_df = pd.DataFrame(fd_data)
    ncols = len(ad_df)
    fig, axs = plt.subplots(figsize=(2*ncols, 2), ncols=ncols)
    for ax, (_, line) in zip(axs, ad_df.iterrows()):
        sub_fd_df = fd_df[(fd_df.i_param == line.i_param) & (fd_df.i_elem == line.i_elem)]
        ax.plot(sub_fd_df.epsilon, sub_fd_df.grad)
        grad = line['grad']
        ax.axhline(y=grad, ls=':')
        ax.set_ylim(min(0, 2 * grad), max(0, 2 * grad))
        ax.set_xscale('log')
    fig.tight_layout()
    plt.show()




def get_random_dataset(batch_size, input_dim, n_batches, privileged_dim, rng):
    dataset_size = n_batches * batch_size
    x = rng.normal(0, 1, (dataset_size, input_dim))
    x_priv = rng.normal(0, 1, (dataset_size, privileged_dim))
    y = rng.normal(0, 1, (dataset_size,))
    dataset = TensorDataset(torch.tensor(x, dtype=torch.float32, ),
                            torch.tensor(x_priv, dtype=torch.float32),
                            torch.tensor(y, dtype=torch.float32))

    return dataset


if __name__ == '__main__':
    visualize_meta_gradient()
