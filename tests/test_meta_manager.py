from collections import defaultdict

import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset

from ldt.meta.manager import MetaManager


def _loader_to_list(dl):
    output = []
    for batch in dl:
        assert len(batch) == 1, 'expected TensorDataset with one tensor'
        output.append(batch[0].tolist())
    return output


def _flatten_once(xs):
    return [x for sublist in xs for x in sublist]


def _independent_draws(*args, n):
    for i in range(n):
        yield tuple(a() for a in args)


class TestMetaManager:
    # # For grid-based tests, use the following decorators instead
    # @pytest.mark.parametrize('train_set_size', [14, 17])
    # @pytest.mark.parametrize('train_batch_size', [2, 7])
    # @pytest.mark.parametrize('valid_batch_size', [3, 6])
    # @pytest.mark.parametrize('train_steps_per_iteration', [1, 2, 5])
    # @pytest.mark.parametrize('validation_split', [0.2, 0.5])
    # @pytest.mark.parametrize('drop_last_train_batch', [0, 1])
    # @pytest.mark.parametrize('drop_last_train_iteration', [0, 1])
    # @pytest.mark.parametrize('meta_train_batch_size', [2, 5])
    # @pytest.mark.parametrize('always_resample', [0, 1])
    @pytest.mark.parametrize(argnames=[
        'train_set_size',
        'train_batch_size',
        'valid_batch_size',
        'train_steps_per_iteration',
        'validation_split',
        'drop_last_train_batch',
        'drop_last_train_iteration',
        'meta_train_batch_size',
        'always_resample',
    ],
        argvalues=_independent_draws(
            lambda: np.random.randint(25, 40),  # train_set_size
            lambda: np.random.randint(1, 10),  # train_batch_size
            lambda: np.random.randint(1, 10),  # valid_batch_size
            lambda: np.random.randint(1, 10),  # train_steps_per_iteration
            lambda: np.random.uniform(0.1, 0.6),  # validation_split
            lambda: np.random.choice([0, 1]),  # drop_last_train_batch
            lambda: np.random.choice([0, 1]),  # drop_last_train_iteration
            lambda: np.random.randint(1, 10),  # meta_train_batch_size
            lambda: np.random.choice([0, 1]),  # always_resample
            n=1000,
        )

    )
    def test_invariants(self,
                        train_set_size,
                        train_batch_size,
                        valid_batch_size,
                        train_steps_per_iteration,
                        validation_split,
                        drop_last_train_batch,
                        drop_last_train_iteration,
                        meta_train_batch_size,
                        always_resample,
                        ):

        test_set_size = 100
        n_epochs = 5

        train_set = TensorDataset(10 + torch.arange(train_set_size))
        test_set = TensorDataset(10000 + torch.arange(test_set_size))
        manager = MetaManager(
            train_set, test_set,
            train_batch_size=train_batch_size,
            valid_batch_size=valid_batch_size,
            test_batch_size=4,
            train_steps_per_iteration=train_steps_per_iteration,
            n_epochs=n_epochs,
            validation_split=validation_split,
            drop_last={'train_batch': drop_last_train_batch,
                       'train_iteration': drop_last_train_iteration,
                       },
            always_resample=always_resample,
            meta_train_batch_size=meta_train_batch_size,
            num_workers=0,

        )
        seen_meta_train_examples = set()
        seen_meta_valid_examples = set()
        seen_train_examples = set()

        # Keep track of training examples seen during each epoch
        epoch_training_examples = defaultdict(list)

        for i, i_iteration in enumerate(manager.iterate()):

            # i_iteration is a list of ascending integers (0, 1, 2, 3, ...)
            assert i_iteration == i

            meta_train_batches = _loader_to_list(manager.current_meta_train_loader())
            meta_valid_batches = _loader_to_list(manager.current_meta_valid_loader())
            train_batches = _loader_to_list(manager.current_train_loader())
            examples_meta_train = set(_flatten_once(meta_train_batches))
            examples_meta_valid = set(_flatten_once(meta_valid_batches))
            examples_train = set(_flatten_once(train_batches))

            epoch_training_examples[manager.current_epoch()].extend(_flatten_once(train_batches))

            seen_meta_train_examples |= examples_meta_train
            seen_meta_valid_examples |= examples_meta_valid
            seen_train_examples |= examples_train

            # Check that the empirical size of the validation set is equal to validation_split * train_set_size
            expected_validation_set_size = int(round(validation_split * train_set_size))
            assert len(examples_meta_valid) == expected_validation_set_size

            # Validation and training examples must always be disjoint
            assert not (examples_meta_train & examples_meta_valid)

            # Validation and training examples cover the full training set
            if not drop_last_train_batch:
                assert set(train_set.tensors[0].tolist()) == examples_meta_train | examples_meta_valid

            # Batches have the required length
            for batch in meta_train_batches[:-1]:
                assert len(batch) == meta_train_batch_size
            if drop_last_train_batch:
                assert len(meta_train_batches[-1]) == meta_train_batch_size



            # Every iteration, we should have at most train_steps_per_iteration
            # batches. If the overall dataset is too small, we can get fewer
            assert len(train_batches) <= train_steps_per_iteration
            if (drop_last_train_iteration and
                    (train_set_size - expected_validation_set_size) >= train_batch_size * train_steps_per_iteration):
                assert len(train_batches) == train_steps_per_iteration

            # The train loader examples are a prefix of the meta_train_loader examples
            n_prefix = min(len(_flatten_once(train_batches)),
                           len(_flatten_once(meta_train_batches)))
            assert (_flatten_once(meta_train_batches)[:n_prefix] ==
                    _flatten_once(train_batches)[:n_prefix])


        if always_resample:
            # Statistically after 5 epochs, we should expect that there is some overlap
            assert len(seen_train_examples & seen_meta_valid_examples) > 0
        else:
            assert not seen_train_examples & seen_meta_valid_examples
            assert not seen_meta_train_examples & seen_meta_valid_examples

        # All epochs should have equal numbers of examples
        n_examples_per_epoch = [len(examples) for examples in epoch_training_examples.values()]
        assert len(set(n_examples_per_epoch)) == 1

        example_sequences = [tuple(examples) for examples in epoch_training_examples.values()]
        assert len(set(example_sequences)) > 1


    def test_standard(self):
        seed = 0
        train_set_size = 17
        test_set_size = 5
        train_batch_size = 3
        test_batch_size = 4
        n_epochs = 3
        train_steps_per_iteration = 2
        train_set = TensorDataset(10 + torch.arange(train_set_size))
        valid_idx = list(range(1, train_set_size, 2))
        test_set = TensorDataset(100 + torch.arange(test_set_size))
        drop_last = {
            'train_batch': False,
            'train_iteration': False,
        }
        expected_traces = {'i_iteration': [0, 1, 2, 3, 4, 5],
                           'meta_train_batches':
                               [[[24, 14, 12], [18, 26, 22], [16, 10, 20]],
                                [[16, 10, 20], [24, 14, 12], [18, 26, 22]],
                                [[16, 12, 24], [22, 26, 10], [18, 14, 20]],
                                [[18, 14, 20], [16, 12, 24], [22, 26, 10]],
                                [[20, 14, 16], [18, 12, 10], [26, 24, 22]],
                                [[26, 24, 22], [20, 14, 16], [18, 12, 10]]],
                           'meta_valid_batches':
                               [[[11, 13, 15], [17, 19, 21], [23, 25]],
                                [[11, 13, 15], [17, 19, 21], [23, 25]],
                                [[11, 13, 15], [17, 19, 21], [23, 25]],
                                [[11, 13, 15], [17, 19, 21], [23, 25]],
                                [[11, 13, 15], [17, 19, 21], [23, 25]],
                                [[11, 13, 15], [17, 19, 21], [23, 25]]],
                           'test_batches':
                               [[[100, 101, 102, 103], [104]],
                                [[100, 101, 102, 103], [104]],
                                [[100, 101, 102, 103], [104]],
                                [[100, 101, 102, 103], [104]],
                                [[100, 101, 102, 103], [104]],
                                [[100, 101, 102, 103], [104]]],
                           'train_batches':
                               [[[24, 14, 12], [18, 26, 22]],
                                [[16, 10, 20]],
                                [[16, 12, 24], [22, 26, 10]],
                                [[18, 14, 20]],
                                [[20, 14, 16], [18, 12, 10]],
                                [[26, 24, 22]]]
                           }

        self._test(seed, train_set, test_set, valid_idx,
                   train_batch_size, test_batch_size, train_steps_per_iteration,
                   n_epochs, expected_traces, drop_last)


    def test_drop_last(self):
        seed = 0
        train_set_size = 17
        test_set_size = 5
        train_batch_size = 3
        test_batch_size = 4
        n_epochs = 3
        train_steps_per_iteration = 2
        train_set = TensorDataset(10 + torch.arange(train_set_size))
        valid_idx = list(range(1, train_set_size, 2))
        test_set = TensorDataset(100 + torch.arange(test_set_size))
        drop_last = {
            'train_batch': True,
            'train_iteration': False,
        }
        expected_traces = {'i_iteration': [0, 1, 2, 3, 4, 5],
                           'meta_train_batches':
                               [[[24, 14, 12], [18, 26, 22], [16, 10, 20]],
                                [[16, 10, 20], [24, 14, 12], [18, 26, 22]],
                                [[16, 12, 24], [22, 26, 10], [18, 14, 20]],
                                [[18, 14, 20], [16, 12, 24], [22, 26, 10]],
                                [[20, 14, 16], [18, 12, 10], [26, 24, 22]],
                                [[26, 24, 22], [20, 14, 16], [18, 12, 10]]],
                           'meta_valid_batches':
                               [[[11, 13, 15], [17, 19, 21], [23, 25]],
                                [[11, 13, 15], [17, 19, 21], [23, 25]],
                                [[11, 13, 15], [17, 19, 21], [23, 25]],
                                [[11, 13, 15], [17, 19, 21], [23, 25]],
                                [[11, 13, 15], [17, 19, 21], [23, 25]],
                                [[11, 13, 15], [17, 19, 21], [23, 25]]],
                           'test_batches':
                               [[[100, 101, 102, 103], [104]],
                                [[100, 101, 102, 103], [104]],
                                [[100, 101, 102, 103], [104]],
                                [[100, 101, 102, 103], [104]],
                                [[100, 101, 102, 103], [104]],
                                [[100, 101, 102, 103], [104]]],
                           'train_batches':
                               [[[24, 14, 12], [18, 26, 22]],
                                [[16, 10, 20]],
                                [[16, 12, 24], [22, 26, 10]],
                                [[18, 14, 20]],
                                [[20, 14, 16], [18, 12, 10]],
                                [[26, 24, 22]]]
                           }

        self._test(seed, train_set, test_set, valid_idx,
                   train_batch_size, test_batch_size, train_steps_per_iteration,
                   n_epochs, expected_traces, drop_last)

    def test_too_many_train_steps_per_iteration(self):
        seed = 0
        train_set_size = 17
        test_set_size = 5
        train_batch_size = 3
        test_batch_size = 4
        n_epochs = 2
        train_steps_per_iteration = 100
        train_set = TensorDataset(10 + torch.arange(train_set_size))
        valid_idx = list(range(1, train_set_size, 2))
        test_set = TensorDataset(100 + torch.arange(test_set_size))
        drop_last = {
            'train_batch': False,
            'train_iteration': False,
        }
        expected_traces = {'i_iteration': [0, 1],
                           'meta_train_batches':
                               [[[24, 14, 12], [18, 26, 22], [16, 10, 20]],
                                [[16, 12, 24], [22, 26, 10], [18, 14, 20]]],
                           'meta_valid_batches':
                               [[[11, 13, 15], [17, 19, 21], [23, 25]],
                                [[11, 13, 15], [17, 19, 21], [23, 25]]],
                           'test_batches':
                               [[[100, 101, 102, 103], [104]],
                                [[100, 101, 102, 103], [104]]],
                           'train_batches':
                               [[[24, 14, 12], [18, 26, 22], [16, 10, 20]],
                                [[16, 12, 24], [22, 26, 10], [18, 14, 20]]]
                           }

        self._test(seed, train_set, test_set, valid_idx,
                   train_batch_size, test_batch_size, train_steps_per_iteration,
                   n_epochs, expected_traces, drop_last)

    def test_fractional_last_batch(self):
        seed = 0
        train_set_size = 13
        test_set_size = 5
        train_batch_size = 3
        test_batch_size = 4
        n_epochs = 3
        train_steps_per_iteration = 2
        train_set = TensorDataset(10 + torch.arange(train_set_size))
        valid_idx = list(range(1, train_set_size, 2))
        test_set = TensorDataset(100 + torch.arange(test_set_size))
        drop_last = {
            'train_batch': False,
            'train_iteration': False,
        }
        expected_traces = {
            'i_iteration': [0, 1, 2, 3, 4, 5],
             'meta_train_batches': [[[22, 14, 12], [16, 10, 20], [18]],
                                    [[18, 22, 14], [12, 16, 10], [20]],
                                    [[12, 10, 22], [16, 18, 14], [20]],
                                    [[20, 12, 10], [22, 16, 18], [14]],
                                    [[22, 16, 20], [12, 14, 18], [10]],
                                    [[10, 22, 16], [20, 12, 14], [18]]],
             'meta_valid_batches': [[[11, 13, 15], [17, 19, 21]],
                                    [[11, 13, 15], [17, 19, 21]],
                                    [[11, 13, 15], [17, 19, 21]],
                                    [[11, 13, 15], [17, 19, 21]],
                                    [[11, 13, 15], [17, 19, 21]],
                                    [[11, 13, 15], [17, 19, 21]]],
             'test_batches': [[[100, 101, 102, 103], [104]],
                              [[100, 101, 102, 103], [104]],
                              [[100, 101, 102, 103], [104]],
                              [[100, 101, 102, 103], [104]],
                              [[100, 101, 102, 103], [104]],
                              [[100, 101, 102, 103], [104]]],
             'train_batches': [[[22, 14, 12], [16, 10, 20]],
                               [[18]],
                               [[12, 10, 22], [16, 18, 14]],
                               [[20]],
                               [[22, 16, 20], [12, 14, 18]],
                               [[10]]]
        }

        self._test(seed, train_set, test_set, valid_idx,
                   train_batch_size, test_batch_size, train_steps_per_iteration,
                   n_epochs, expected_traces, drop_last)

    def test_drop_last_iteration(self):
        seed = 0
        train_set_size = 18
        test_set_size = 5
        train_batch_size = 3
        test_batch_size = 5
        n_epochs = 3
        train_steps_per_iteration = 2
        train_set = TensorDataset(10 + torch.arange(train_set_size))
        valid_idx = list(range(1, train_set_size, 2))


        test_set = TensorDataset(100 + torch.arange(test_set_size))
        drop_last = {
            'train_batch': False,
            'train_iteration': True,
        }
        expected_traces = {
            'i_iteration': [0, 1, 2],
             'meta_train_batches': [[[24, 14, 12], [18, 26, 22], [16, 10, 20]],
                                    [[16, 12, 24], [22, 26, 10], [18, 14, 20]],
                                    [[20, 14, 16], [18, 12, 10], [26, 24, 22]]],
             'meta_valid_batches': [[[11, 13, 15], [17, 19, 21], [23, 25, 27]],
                                    [[11, 13, 15], [17, 19, 21], [23, 25, 27]],
                                    [[11, 13, 15], [17, 19, 21], [23, 25, 27]]],
             'test_batches': [[[100, 101, 102, 103, 104]],
                              [[100, 101, 102, 103, 104]],
                              [[100, 101, 102, 103, 104]]],
             'train_batches': [[[24, 14, 12], [18, 26, 22]],
                               [[16, 12, 24], [22, 26, 10]],
                               [[20, 14, 16], [18, 12, 10]],
                               ]
        }

        self._test(seed, train_set, test_set, valid_idx,
                   train_batch_size, test_batch_size, train_steps_per_iteration,
                   n_epochs, expected_traces, drop_last)


    def test_should_test_every_epoch(self):
        seed = 0
        train_set_size = 18
        test_set_size = 5
        train_batch_size = 3
        test_batch_size = 5
        n_epochs = 5
        train_steps_per_iteration = 3
        train_set = TensorDataset(10 + torch.arange(train_set_size))
        valid_idx = list(range(1, train_set_size, 2))
        tests_per_epoch = 1

        test_set = TensorDataset(100 + torch.arange(test_set_size))
        drop_last = {
            'train_batch': False,
            'train_iteration': False,
        }
        expected_traces = {
            'i_iteration': [0, 1, 2, 3, 4],
            'should_test': [True, True, True, True, True],
        }

        self._test(seed, train_set, test_set, valid_idx,
                   train_batch_size, test_batch_size, train_steps_per_iteration,
                   n_epochs, expected_traces, drop_last, tests_per_epoch)

    def test_should_test_every_other_epoch(self):
        seed = 0
        train_set_size = 18
        test_set_size = 5
        train_batch_size = 3
        test_batch_size = 5
        n_epochs = 5
        train_steps_per_iteration = 3
        train_set = TensorDataset(10 + torch.arange(train_set_size))
        valid_idx = list(range(1, train_set_size, 2))
        tests_per_epoch = 0.5

        test_set = TensorDataset(100 + torch.arange(test_set_size))
        drop_last = {
            'train_batch': False,
            'train_iteration': False,
        }
        expected_traces = {
            'i_iteration': [0, 1, 2, 3, 4],
            'should_test': [True, False, True, False, True],
        }

        self._test(seed, train_set, test_set, valid_idx,
                   train_batch_size, test_batch_size, train_steps_per_iteration,
                   n_epochs, expected_traces, drop_last, tests_per_epoch)


    def test_should_test_never(self):
        seed = 0
        train_set_size = 18
        test_set_size = 5
        train_batch_size = 3
        test_batch_size = 5
        n_epochs = 5
        train_steps_per_iteration = 3
        train_set = TensorDataset(10 + torch.arange(train_set_size))
        valid_idx = list(range(1, train_set_size, 2))
        tests_per_epoch = 0.0

        test_set = TensorDataset(100 + torch.arange(test_set_size))
        drop_last = {
            'train_batch': False,
            'train_iteration': False,
        }
        expected_traces = {
            'i_iteration': [0, 1, 2, 3, 4],
            'should_test': [False, False, False, False, True],
        }

        self._test(seed, train_set, test_set, valid_idx,
                   train_batch_size, test_batch_size, train_steps_per_iteration,
                   n_epochs, expected_traces, drop_last, tests_per_epoch)

    def test_should_test_fractional(self):
        seed = 0
        train_set_size = 18
        test_set_size = 5
        train_batch_size = 3
        test_batch_size = 5
        n_epochs = 16
        train_steps_per_iteration = 3
        train_set = TensorDataset(10 + torch.arange(train_set_size))
        valid_idx = list(range(1, train_set_size, 2))
        tests_per_epoch = 0.75

        test_set = TensorDataset(100 + torch.arange(test_set_size))
        drop_last = {
            'train_batch': False,
            'train_iteration': False,
        }
        expected_traces = {
            'i_iteration': list(range(n_epochs)),
            'should_test': [1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1],
        }

        self._test(seed, train_set, test_set, valid_idx,
                   train_batch_size, test_batch_size, train_steps_per_iteration,
                   n_epochs, expected_traces, drop_last, tests_per_epoch)


    def test_should_test_fractional_multiple_iterations_per_epoch(self):
        seed = 0
        train_set_size = 18
        test_set_size = 5
        train_batch_size = 3
        test_batch_size = 5
        n_epochs = 8
        train_steps_per_iteration = 2
        train_set = TensorDataset(10 + torch.arange(train_set_size))
        valid_idx = list(range(1, train_set_size, 2))
        tests_per_epoch = 0.75

        test_set = TensorDataset(100 + torch.arange(test_set_size))
        drop_last = {
            'train_batch': False,
            'train_iteration': False,
        }
        expected_traces = {
            'i_iteration': list(range(16)),
            'should_test': [1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1],
        }

        self._test(seed, train_set, test_set, valid_idx,
                   train_batch_size, test_batch_size, train_steps_per_iteration,
                   n_epochs, expected_traces, drop_last, tests_per_epoch)


    def _test(self, seed, train_set, test_set, valid_idx, train_batch_size,
              test_batch_size, train_steps_per_iteration,
              n_epochs, expected_traces, drop_last,
              tests_per_epoch=0):
        manager = MetaManager(
            train_set=train_set,
            test_set=test_set,

            train_batch_size=train_batch_size,
            valid_batch_size=train_batch_size,
            test_batch_size=test_batch_size,
            train_steps_per_iteration=train_steps_per_iteration,
            n_epochs=n_epochs,
            valid_idxs=valid_idx,
            tests_per_epoch=tests_per_epoch,

            pin_memory=True,
            num_workers=1,

            rng=np.random.RandomState(seed),
            drop_last=drop_last,
            always_resample=False,

        )
        traces = defaultdict(list)
        for i_iteration in manager.iterate():
            traces['i_iteration'].append(i_iteration)

            traces['meta_train_batches'].append(_loader_to_list(manager.current_meta_train_loader()))
            traces['meta_valid_batches'].append(_loader_to_list(manager.current_meta_valid_loader()))
            traces['train_batches'].append(_loader_to_list(manager.current_train_loader()))
            traces['test_batches'].append(_loader_to_list(manager.current_test_loader()))
            traces['should_test'].append(manager.should_test())

        traces = {k: v for k, v in traces.items() if k in expected_traces}


        assert traces == expected_traces

    def test_no_duplicate_valid_idxs(self):
        train_set_size = 512
        test_set_size = 256
        validation_split = 0.4
        expected_validation_set_size = int(round(validation_split * train_set_size))
        manager = MetaManager(
            train_set=TensorDataset(10 + torch.arange(train_set_size)),
            test_set=TensorDataset(100 + torch.arange(test_set_size)),

            train_batch_size=32,
            valid_batch_size=64,
            test_batch_size=128,

            train_steps_per_iteration=3,
            n_epochs=5,

            validation_split=validation_split,

            tests_per_epoch=2,

            pin_memory=True,
            num_workers=1,

            rng=np.random.RandomState(123),
            drop_last={'train_batch': False,
                       'train_iteration': False},

            always_resample=False,
        )
        assert len(manager.valid_idxs) == len(set(manager.valid_idxs))
        assert len(set(manager.valid_idxs)) == expected_validation_set_size

    def test_last_incomplete_epoch(self):
        """
        Special situtation for
        drop_last = {
            "train_batch": True,
            "train_iteration": False,
        }:

        If there is a residual iteration with one batch which is too small,
        then this last iteration should actually be dropped.
        """
        validation_split = 0.5

        train_batch_size = 32
        train_steps_per_iteration = 8

        # Make a training set with a size such that some integer number of
        # iterations fit, and the residual is less than a batch
        train_set_size = int(round((512 + 31) * 1/(1 - validation_split)))

        always_resample = False
        drop_last = {'train_batch': True,
                     'train_iteration': False}

        manager = MetaManager(
            train_set=TensorDataset(10 + torch.arange(train_set_size)),
            test_set=TensorDataset(1000 + torch.arange(256)),

            train_batch_size=train_batch_size,
            valid_batch_size=64,
            test_batch_size=128,

            train_steps_per_iteration=train_steps_per_iteration,
            n_epochs=9,

            validation_split=validation_split,

            tests_per_epoch=2,

            pin_memory=True,
            num_workers=1,

            rng=np.random.RandomState(123),
            drop_last=drop_last,

            always_resample=always_resample,
        )
        for i_epoch in manager.iterate():
            train_batches = _loader_to_list(manager.current_train_loader())
            assert len(train_batches) > 0
