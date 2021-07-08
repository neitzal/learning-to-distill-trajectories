from math import ceil

import numpy as np
from loguru import logger
from torch.utils.data import Sampler, Dataset
from torch.utils.data.dataloader import DataLoader


class ListSampler(list, Sampler):
    """
    Deterministic "sampler": simply provides its elements in order
    """
    pass


class MetaManager:
    """
    Manages training, validation and tests loaders.
    """

    def __init__(self,
                 train_set: Dataset,
                 test_set: Dataset,
                 train_batch_size: int,
                 valid_batch_size: int,
                 test_batch_size: int,
                 train_steps_per_iteration: int,
                 n_epochs,
                 valid_idxs=None,
                 validation_split=None,
                 tests_per_epoch=1,
                 pin_memory=True,
                 num_workers=0,
                 rng=None,
                 drop_last=None,
                 always_resample=False,
                 meta_train_batch_size=None
                 ):
        """
        :param train_set: PyTorch Dataset with training examples
                         (validation will be taken from here)
        :param test_set: PyTorch Dataset with test examples (only for evaluation)
        :param train_batch_size: Batch size for inner-training loop
        :param valid_batch_size: Batch size used to compute the validation loss
                                 (should not affect the results)
        :param test_batch_size: Batch size used to compute the test loss
        :param train_steps_per_iteration: Number of training batches per meta-step
        :param n_epochs: Total number of times of iterating over the training set
        :param valid_idxs: Provide a specific set of indices of the training set
                           which should be used for validation.
        :param validation_split: Fraction of training set used for validation.
                                Mutually exclusive with valid_idx.
        :param tests_per_epoch: How many times per epoch "should_test" happens
        :param pin_memory: Will passed through to the DataLoaders
        :param num_workers: Will passed through to the DataLoaders
        :param rng: Instance of a random number generator
        :param drop_last: dictionary with two entries:
                'train_batch': drop the last batch of an epoch if it is smaller
                    than the requested batch size
                'train_iteration': drop the last iteration if the train loader
                    is too small to accommodate
                Only applies if always_resample is set to False.
        :param always_resample: Resample a new meta_train/meta_validation split
                at each iteration (i.e., more often than every epoch!)
        """
        self.always_resample = always_resample
        self.tests_per_epoch = tests_per_epoch
        self.train_batch_size = train_batch_size
        if meta_train_batch_size is not None:
            self.meta_train_batch_size = meta_train_batch_size
        else:
            self.meta_train_batch_size = train_batch_size

        self.valid_batch_size = valid_batch_size
        self.test_set = test_set
        self.train_set = train_set
        self.test_batch_size = test_batch_size

        if rng is None:
            rng = np.random.RandomState()

        self.rng = rng

        n_train = len(self.train_set)
        if valid_idxs is not None:
            if validation_split is not None:
                raise ValueError('Please only specify one of validation_split '
                                 'and valid_idxs')
            self.valid_idxs = valid_idxs
        elif validation_split is None:
            raise ValueError('Please provide one of validation_split or '
                             'valid_idxs.')
        else:
            n_valid = int(round(validation_split * n_train))
            self.valid_idxs = self.rng.choice(np.arange(n_train),
                                              size=n_valid,
                                              replace=False)

        self.n_valid = len(self.valid_idxs)
        self.n_train = n_train   # This includes the validation indices

        self.train_idxs = np.array([i for i in range(n_train)
                                    if i not in self.valid_idxs])

        if (len(self.train_idxs) < train_batch_size
                or len(self.train_idxs) < self.meta_train_batch_size):
            raise ValueError(f'Training set too small ({len(self.train_idxs)}) '
                             f'for requested batch size: '
                             f'train_batch_size {train_batch_size}, '
                             f'meta_train_batch_size {meta_train_batch_size}')

        if drop_last is None:
            drop_last = {
                'train_batch': False,
                'train_iteration': False,
            }

        assert isinstance(drop_last, dict)
        self.drop_last = drop_last

        self.train_steps_per_iteration = min(
            train_steps_per_iteration,
            int(ceil(len(self.train_idxs) / train_batch_size))
        )

        if self.train_steps_per_iteration != train_steps_per_iteration:
            logger.warning(f'Requested train_steps_per_iteration={train_steps_per_iteration}. '
                           f'However, since train_batch_size={train_batch_size} '
                           f'and len(self.train_idxs)={len(self.train_idxs)}, '
                           f'will only perform {self.train_steps_per_iteration} '
                           f'steps per iteration.')

        self.n_epochs = n_epochs
        self.pin_memory = pin_memory
        self.num_workers = num_workers

        self._meta_train_loader = None
        self._meta_valid_loader = None
        self._train_loader = None
        self._test_loader = DataLoader(
            self.test_set, batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )

        self._should_test = False

    def iterate(self):
        train_examples_per_iter = min(
            self.train_batch_size * self.train_steps_per_iteration,
            len(self.train_idxs)
        )
        if self.drop_last['train_iteration'] and not self.always_resample:
            iterations_per_epoch = len(self.train_idxs) // train_examples_per_iter
        else:
            iterations_per_epoch = int(ceil(len(self.train_idxs) / train_examples_per_iter))

        n_iterations = self.n_epochs * iterations_per_epoch

        tests_per_iteration = self.tests_per_epoch / iterations_per_epoch
        logger.info(f'Tests per iteration: {tests_per_iteration}')
        test_counter = 0
        epoch_train_idxs = self.rng.permutation(self.train_idxs)
        lower = 0

        for i in range(n_iterations):

            self._current_frac_epoch = i / iterations_per_epoch
            self.epoch_counter = int(np.floor(self._current_frac_epoch))

            if self.always_resample:
                self.valid_idxs = self.rng.choice(np.arange(self.n_train),
                                             size=self.n_valid,
                                             replace=False)
                meta_train_idxs = np.array([i for i in range(self.n_train)
                                            if i not in self.valid_idxs])
                train_idxs = meta_train_idxs[:train_examples_per_iter]

            else:
                if self.drop_last['train_iteration']:
                    new_epoch = lower > (len(self.train_idxs) - train_examples_per_iter)
                else:
                    # Check if the current position exceeds the training set size.
                    # If we are dropping the last batch, we need to take this
                    # into account too.
                    n_max = len(self.train_idxs)
                    if self.drop_last['train_batch']:
                        n_max -= (self.train_batch_size - 1)
                    new_epoch = lower >= n_max

                if new_epoch:
                    epoch_train_idxs = self.rng.permutation(self.train_idxs)
                    lower = 0

                meta_train_idxs = np.roll(epoch_train_idxs, shift=-lower)
                train_idxs = epoch_train_idxs[lower:lower + train_examples_per_iter]

            assert train_idxs[0] == meta_train_idxs[0], 'train_idxs and meta_train_idxs are not aligned'

            # Set loaders
            self._meta_train_loader = DataLoader(
                self.train_set, batch_size=self.meta_train_batch_size,
                sampler=ListSampler(meta_train_idxs),
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                drop_last=bool(self.drop_last['train_batch']),
            )
            if len(self.valid_idxs) > 0:
                self._meta_valid_loader = DataLoader(
                    self.train_set, batch_size=self.valid_batch_size,
                    sampler=ListSampler(self.valid_idxs),
                    num_workers=self.num_workers,
                    pin_memory=self.pin_memory,
                )
            else:
                print('No meta-valid loader')
                self._meta_valid_loader = None

            self._train_loader = DataLoader(
                self.train_set, batch_size=self.train_batch_size,
                sampler=ListSampler(train_idxs),
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                drop_last=bool(self.drop_last['train_batch']),
            )

            last_iteration = i == n_iterations - 1
            if last_iteration or test_counter < (i + 1) * tests_per_iteration:
                self._should_test = True
                test_counter += 1
            else:
                self._should_test = False
            lower += train_examples_per_iter
            yield i

    def current_epoch(self):
        return self.epoch_counter

    def current_meta_train_loader(self):
        return self._meta_train_loader

    def current_meta_valid_loader(self):
        return self._meta_valid_loader

    def current_train_loader(self):
        return self._train_loader

    def current_test_loader(self):
        return self._test_loader

    def current_frac_epoch(self):
        return self._current_frac_epoch

    def should_test(self):
        return self._should_test