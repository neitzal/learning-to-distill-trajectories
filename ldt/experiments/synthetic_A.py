import gc
import time

import numpy as np
import torch
import wandb
from loguru import logger
from torch import optim as optim
from torch.nn import functional as F

from ldt.datasets.synthetic_A import make_datasets
from ldt.meta.manager import MetaManager
from ldt.meta.training import train_meta_step, train_regular, run_test_epoch
from ldt.models.synthetic_vanilla.loss import prepare_batch, compute_teaching_loss
from ldt.models.synthetic_vanilla.student import StudentModel
from ldt.models.synthetic_vanilla.teacher import Teacher


def train(params):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(params.seed)
    np.random.seed(params.seed)

    use_cuda = False
    logger.info(f'use_cuda {use_cuda}')
    device = torch.device('cuda' if use_cuda else 'cpu')

    if params.method == 'meta':
        n_inner = params.n_inner
    else:
        n_inner = 0

    if params.method == 'meta':
        validation_split = params.validation_split
    else:
        validation_split = 0.0

    valid_batch_size = params.batch_size

    n_test_examples = 1000
    test_batch_size = n_test_examples

    if params.method == 'meta':
        n_train_train_split = int((1 - validation_split) * params.n_train)
        train_batches_per_iteration = n_train_train_split // params.batch_size
        assert train_batches_per_iteration >= 1
    else:
        train_batches_per_iteration = params.n_train // params.batch_size

    student = StudentModel(input_dim=params.input_dim, hidden_dim=params.student_hidden_dim)
    student.to(device)

    if params.method in ['meta', 'fixed-teacher', 'perfect-teacher']:
        teacher = Teacher(privileged_dim=4,
                          student_hidden_dim=params.student_hidden_dim,
                          teacher_hidden_dim=params.teacher_hidden_dim)
        teacher.to(device)
        if params.method == 'perfect-teacher':
            raise NotImplementedError()

    else:
        teacher = None

    if params.method == 'meta':
        meta_params = list(teacher.parameters())
    else:
        meta_params = None

    teaching_coef = torch.tensor(
        params.teaching_coef,
        requires_grad=False,
        dtype=torch.float32,
        device=device
    )

    if params.method == 'meta':
        meta_optimizer = optim.Adam(meta_params, lr=params.meta_lr,
                                    betas=(params.meta_momentum, 0.999))
    elif params.method in ['fixed-teacher', 'no-teacher', 'perfect-teacher']:
        meta_optimizer = None
    else:
        raise ValueError('Unknown method')

    train_ds_torch, test_ds_torch = make_datasets(
        params.input_dim, params.n_train,
        n_test_examples,
        test_set_seed=1234 + params.seed,
        train_set_seed=params.seed)

    tests_per_epoch = 1

    inner_optimizer = optim.Adam(student.parameters(),
                                 lr=params.inner_lr,
                                 betas=(params.inner_momentum, 0.999),
                                 weight_decay=params.inner_weight_decay
                                 )

    criterion = F.binary_cross_entropy_with_logits

    all_meta_metrics = []
    all_test_metrics = []

    manager = MetaManager(
        train_set=train_ds_torch,
        test_set=test_ds_torch,
        train_batch_size=params.batch_size,
        meta_train_batch_size=params.batch_size,
        valid_batch_size=valid_batch_size,
        test_batch_size=test_batch_size,
        validation_split=validation_split,
        train_steps_per_iteration=train_batches_per_iteration,
        n_epochs=params.n_epochs,
        tests_per_epoch=tests_per_epoch,
        pin_memory=True,
        num_workers=0,
        rng=np.random.RandomState(params.seed),
        drop_last={
            'train_batch': False,
            'train_iteration': False,
        },
        always_resample=(params.method == 'meta'),
    )

    logger.info('===== Start training ====')

    for i_iteration in manager.iterate():
        logger.info(f'Iteration {i_iteration} ...')
        wandb_metrics = dict()

        tic = time.perf_counter()

        if n_inner == 0:
            assert params.method in ['fixed-teacher', 'no-teacher', 'perfect-teacher']
        else:
            assert n_inner > 0
            assert params.method == 'meta'

            logger.debug('Call train_meta_step')
            meta_train_loader = manager.current_meta_train_loader()
            meta_valid_loader = manager.current_meta_valid_loader()
            meta_metrics = train_meta_step(
                student, device, meta_train_loader,
                prepare_batch,
                inner_optimizer,
                n_inner, criterion, teacher,
                compute_teaching_loss=compute_teaching_loss,
                meta_opt=meta_optimizer,
                meta_params=meta_params,
                valid_loader=meta_valid_loader,
                teaching_coef=teaching_coef,
                label_coef=1.0,
                clip_meta_grad_value=1e8,
                weight_norm_dim=None,
                track_intermediate_losses=False,
                additional_metrics=None)
            logger.debug('train_meta_step done')

            all_meta_metrics.append(meta_metrics)
            if np.isnan(meta_metrics['meta/loss']):
                logger.info('meta_metrics["meta/loss"] is NaN! Aborting.')
                break

            logger.info(f'meta_metrics: {meta_metrics}')
            for mkey, mvalue in meta_metrics.items():
                wandb_metrics[mkey] = mvalue

        logger.debug('Call train_regular')
        print(f'train_batches_per_iteration: {train_batches_per_iteration}')
        train_metrics = train_regular(
            student, device, manager.current_train_loader(),
            prepare_batch,
            inner_optimizer,
            train_batches_per_iteration, criterion, teacher,
            compute_teaching_loss,
            teaching_coef=teaching_coef,
            label_coef=1.0,
            additional_metrics=None)
        train_metrics['i_iteration'] = i_iteration
        logger.info(f'train_metrics {train_metrics}')

        toc = time.perf_counter()

        if manager.should_test():
            logger.debug('Call run_test_epoch')
            test_metrics = run_test_epoch(student, device,
                                          manager.current_test_loader(),
                                          criterion, prepare_batch)
            all_test_metrics.append(test_metrics)
            logger.info(f'test_metrics {test_metrics}')

            for mkey, mvalue in {**train_metrics, **test_metrics}.items():
                wandb_metrics[mkey] = mvalue

        i_epoch = manager.current_frac_epoch()
        wandb_metrics['i_epoch'] = i_epoch
        wandb_metrics['step'] = i_iteration
        wandb.log(wandb_metrics)

        logger.info(f'step time: {toc - tic:.3} s')
        gc.collect()

    return all_meta_metrics, all_test_metrics
