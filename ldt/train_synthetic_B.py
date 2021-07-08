import argparse
from datetime import datetime

import torch
import wandb
from loguru import logger

from ldt.experiments import synthetic_B

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, required=True, choices=['meta', 'no-teacher', 'fixed-teacher'])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n-epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--input-dim', type=int, default=128, help='Input dimensionality of the task')
    parser.add_argument('--n-train', type=int, default=1000, help='Number of training examples')
    parser.add_argument('--inner-momentum', type=float, default=0.9, help='Momentum of inner optimizer')
    parser.add_argument('--batch-size', type=int, default=32, help='Size of all batches')
    parser.add_argument('--inner-lr', type=float, default=1e-3, help='Inner loop learning rate')
    parser.add_argument('--inner-weight-decay', type=float, default=1e-6, help='Inner loop weight decay')
    parser.add_argument('--teaching-coef', type=float, default=1e4, help='Teaching coefficient')
    parser.add_argument('--validation-split', type=float, default=0.5, help='Fraction of data used for validation.')
    parser.add_argument('--meta-momentum', type=float, default=0.9, help='Momentum of the meta-optimizer.')
    parser.add_argument('--n-inner', type=int, default=64, help='Number of inner-loop optimization steps.')
    parser.add_argument('--meta-lr', type=float, default=1e-3, help='Learning rate of meta-optimizer.')
    parser.add_argument('--student-hidden-dim', type=int, default=256, help='Network width of student.')
    parser.add_argument('--teacher-hidden-dim', type=int, default=256, help='Network width of student.')
    parser.add_argument('--entropy-reg-coef', type=float, default=0.0, help='Regularization coef for method "no-teacher"')

    args = parser.parse_args()
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    logger.info(f'torch.__version__: {torch.__version__}')
    wandb.init(project='LDT-synthetic-B', name=timestamp)
    synthetic_B.train(args)
