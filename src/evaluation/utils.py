from typing import Any, Union

import cloudpickle
import numpy as np


class CloudpickleWrapper(object):
    """A cloudpickle wrapper used in SubprocVectorEnv."""

    def __init__(self, data: Any) -> None:
        self.data = data

    def __getstate__(self) -> str:
        return cloudpickle.dumps(self.data)

    def __setstate__(self, data: str) -> None:
        self.data = cloudpickle.loads(data)


class RunningMeanStd(object):
    """Calculates the running mean and std of a data stream.

    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    """

    def __init__(
        self,
        mean: Union[float, np.ndarray] = 0.0,
        std: Union[float, np.ndarray] = 1.0
    ) -> None:
        self.mean, self.var = mean, std
        self.count = 0

    def update(self, x: np.ndarray) -> None:
        """Add a batch of item into RMS with the same shape, modify mean/var/count."""
        batch_mean, batch_var = np.mean(x, axis=0), np.var(x, axis=0)
        batch_count = len(x)

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + delta**2 * self.count * batch_count / total_count
        new_var = m_2 / total_count

        self.mean, self.var = new_mean, new_var
        self.count = total_count