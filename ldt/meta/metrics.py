import numpy as np
from experiment_utils.miscutils.nputil import softmax


class BaseMetric:
    def __init__(self, key):
        self._key = key

    def __call__(self, predictions, labels, privileged_data):
        return self._call(predictions, labels, privileged_data)

    def _call(self, predictions, labels, privileged_data):
        raise NotImplementedError()

    @property
    def key(self):
        return self._key


class MseLoss(BaseMetric):
    def __init__(self):
        super().__init__(key='mse')

    def _call(self, predictions, labels, privileged_data):
        predictions = np.asarray(predictions)
        labels = np.asarray(labels)

        if predictions.ndim != 2:
            return np.nan

        ps = softmax(predictions, axis=1)

        assert np.allclose(ps.sum(axis=1), np.ones(ps.shape[0]))

        expectations = (np.arange(0, ps.shape[1])[np.newaxis, :] * ps).sum(axis=1)

        assert expectations.min() >= 0
        assert expectations.max() <= (ps.shape[1] - 1) + 1e-4, f'{expectations.max()}, {ps.shape[1] - 1}, {ps}'

        assert expectations.shape == labels.shape

        return ((expectations - labels) ** 2).mean()


class MseFromExpectationLoss(BaseMetric):
    def __init__(self, privileged_to_p):
        super().__init__(key='mse_from_expectation')
        self.privileged_to_p = privileged_to_p

    def _call(self, predictions, labels, privileged_data):
        predictions = np.asarray(predictions)
        true_ps = self.privileged_to_p(privileged_data)

        if predictions.ndim != 2:
            return np.nan

        ps = softmax(predictions, axis=1)

        assert np.allclose(ps.sum(axis=1), np.ones(ps.shape[0]))

        expectations = (np.arange(0, ps.shape[1])[np.newaxis, :] * ps).sum(axis=1)

        true_expectations = (np.arange(0, ps.shape[1])[np.newaxis, :] * true_ps).sum(axis=1)

        assert expectations.min() >= 0
        assert expectations.max() <= (ps.shape[1] - 1) + 1e-4, f'{expectations.max()}, {ps.shape[1] - 1}, {ps}'

        assert expectations.shape == true_expectations.shape

        return ((expectations - true_expectations) ** 2).mean()


class PredictionEntropy(BaseMetric):
    def __init__(self):
        super().__init__(key='prediction_entropy')

    def _call(self, predictions, labels, privileged_data):
        predictions = np.asarray(predictions)
        ps = softmax(predictions, axis=1)
        return (-ps * np.log(ps + 1e-12)).sum(axis=1).mean()


class KLFromTransform(BaseMetric):
    def __init__(self, privileged_to_p, epsilon=1e-12):
        super().__init__(key='kl')
        self.privileged_to_p = privileged_to_p
        self.epsilon = epsilon

    def _call(self, predictions, labels, privileged_data):
        predictions = np.asarray(predictions)
        true_ps = self.privileged_to_p(privileged_data)

        if predictions.ndim != 2:
            return np.nan

        ps = softmax(predictions, axis=1)

        assert np.allclose(ps.sum(axis=1), np.ones(ps.shape[0]))

        kls = (true_ps * np.log(true_ps / (ps + self.epsilon) + self.epsilon)).sum(axis=1)
        return kls.mean()
