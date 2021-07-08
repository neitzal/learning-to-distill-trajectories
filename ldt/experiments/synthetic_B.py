import gc
import time

import numpy as np
import torch
import wandb
from experiment_utils.miscutils.nputil import softmax
from loguru import logger
from torch import optim as optim

from ldt.datasets.synthetic_B import make_datasets
from ldt.meta.manager import MetaManager
from ldt.meta.metrics import MseLoss, MseFromExpectationLoss, PredictionEntropy, KLFromTransform
from ldt.meta.training import run_test_epoch, train_meta_step, train_regular
from ldt.models.synthetic_stochastic_vanilla.loss import prepare_batch, compute_teaching_loss
from ldt.models.synthetic_stochastic_vanilla.student import StudentModel
from ldt.models.synthetic_stochastic_vanilla.teacher import Teacher
from ldt.util.torchutil import make_entropy_regularized_CE


def train(params):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(params.seed)
    np.random.seed(params.seed)
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    if params.method != 'meta':
        params.n_inner = 0

    if params.method == 'meta':
        validation_split = params.validation_split
    else:
        validation_split = 0.0

    n_validation_examples = int(np.ceil(validation_split * params.n_train))
    valid_batch_size = n_validation_examples
    n_test_examples = 1000
    tests_per_epoch = 8
    test_batch_size = n_test_examples
    train_batches_per_iteration = 8
    clip_meta_grad_value = 1e8
    n_classes = 4
    privileged_dim = 32
    student = StudentModel(input_dim=params.input_dim,
                           hidden_dim=params.student_hidden_dim,
                           n_classes=n_classes)
    student.to(device)
    if params.method in ['meta', 'fixed-teacher']:
        teacher = Teacher(teacher_hidden_dim=params.teacher_hidden_dim,
                          privileged_dim=privileged_dim,
                          supervision_dim=n_classes)
        teacher.to(device)
    else:
        teacher = None

    if params.method == 'meta':
        meta_params = list(teacher.parameters())
        logger.info(f'len(meta_params): {len(meta_params)}')

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

    elif params.method in ['fixed-teacher', 'no-teacher']:
        meta_optimizer = None
    else:
        raise ValueError('Unknown method')

    # Needed to compute the KL metric
    privileged_transform = torch.nn.Linear(n_classes, privileged_dim, bias=False)

    train_ds_torch, test_ds_torch = make_datasets(params.input_dim,
                                                  n_classes,
                                                  privileged_transform,
                                                  params.n_train,
                                                  n_test_examples, 1234 + params.seed,
                                                  params.seed)
    inner_weight_decay = params.inner_weight_decay
    inner_momentum = params.inner_momentum
    inner_optimizer = optim.Adam(student.parameters(),
                                 lr=params.inner_lr,
                                 betas=(inner_momentum, 0.999),
                                 weight_decay=inner_weight_decay)

    # Baseline uses entropy regularization
    if params.method != 'no-teacher':
        params.entropy_reg_coef = 0.0

    criterion = make_entropy_regularized_CE(params.entropy_reg_coef)

    def privileged_to_p(privileged_data):
        privileged_data = np.asarray(privileged_data)
        W = privileged_transform.weight.detach().cpu().numpy()
        logits = (np.linalg.pinv(W) @ privileged_data.T).T
        ps = softmax(logits, axis=1)
        return ps

    additional_metrics = [
        MseLoss(),
        MseFromExpectationLoss(privileged_to_p),
        PredictionEntropy(),
        KLFromTransform(privileged_to_p)
    ]

    all_meta_metrics = []
    all_test_metrics = []

    manager = MetaManager(
        train_set=train_ds_torch,
        test_set=test_ds_torch,
        train_batch_size=params.batch_size,
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
            'train_iteration': True,
        },
        always_resample=(params.method == 'meta'),
    )

    logger.info('===== Start training ====')

    for i_iteration in manager.iterate():
        logger.info(f'Iteration  {i_iteration} ...')
        wandb_metrics = dict()

        if manager.should_test():
            logger.debug('Call run_test_epoch')
            test_metrics = run_test_epoch(student, device, manager.current_test_loader(),
                                          criterion,
                                          prepare_batch, additional_metrics)
            all_test_metrics.append(test_metrics)
            logger.info(f'test_metrics {test_metrics}')
            for mkey, mvalue in test_metrics.items():
                wandb_metrics[mkey] = mvalue

        tic = time.perf_counter()

        if params.n_inner == 0:
            assert params.method in ['fixed-teacher', 'no-teacher']
        else:
            assert params.n_inner > 0
            assert params.method == 'meta'
            meta_train_loader = manager.current_meta_train_loader()
            meta_valid_loader = manager.current_meta_valid_loader()

            logger.debug('Call train_meta_step')
            meta_metrics = train_meta_step(
                student, device, meta_train_loader, prepare_batch,
                inner_optimizer,
                params.n_inner, criterion, teacher,
                compute_teaching_loss=compute_teaching_loss,
                meta_opt=meta_optimizer,
                meta_params=meta_params,
                valid_loader=meta_valid_loader,
                teaching_coef=teaching_coef,
                label_coef=1.0,
                clip_meta_grad_value=clip_meta_grad_value,
                weight_norm_dim=None,
                track_intermediate_losses=False,
                additional_metrics=additional_metrics)
            logger.debug('train_meta_step done')

            all_meta_metrics.append(meta_metrics)

            logger.info(f'meta_metrics: {meta_metrics}')
            for mkey, mvalue in meta_metrics.items():
                wandb_metrics[mkey] = mvalue

        logger.debug('Call train_regular')
        train_metrics = train_regular(
            student, device, manager.current_train_loader(),
            prepare_batch,
            inner_optimizer,
            train_batches_per_iteration, criterion, teacher,
            compute_teaching_loss,
            teaching_coef=teaching_coef,
            label_coef=1.0,
            additional_metrics=additional_metrics)

        logger.info(f'train_metrics {train_metrics}')

        toc = time.perf_counter()

        for mkey, mvalue in train_metrics.items():
            wandb_metrics[mkey] = mvalue

        i_epoch = manager.current_frac_epoch()
        wandb_metrics['i_epoch'] = i_epoch
        wandb_metrics['step'] = i_iteration
        wandb.log(wandb_metrics)

        logger.info(f'step time: {toc - tic:.3} s')
        gc.collect()

    return all_meta_metrics, all_test_metrics
