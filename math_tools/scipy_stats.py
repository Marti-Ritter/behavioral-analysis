from typing import Iterable

import numpy as np
from .array_funcs import np_sample_along_first_axis


def sample_mixture(distributions_list, sample_size, coefficients=None, random_state=None):
    if coefficients is None:
        coefficients = np.array([1] * len(distributions_list))
    else:
        coefficients = np.array(coefficients)

    if random_state is not None:
        np.random.seed(random_state)

    num_distr = len(distributions_list)
    data = np.zeros((sample_size, num_distr))
    for idx, distr in enumerate(distributions_list):
        data[:, idx] = distr.rvs(sample_size, random_state=random_state)
    random_idx = np.random.choice(np.arange(num_distr), size=(sample_size,), p=coefficients / sum(coefficients))
    sample = data[np.arange(sample_size), random_idx]
    return sample


class FrozenDistributionMixture:
    """
    This class is a wrapper for a mixture of distributions. It is a frozen distribution, so it can be used in
    scipy.stats functions. It only works with continuous distributions from scipy, i.e. (hopefully) any rv_continuous.
    """
    def __init__(self, distributions, coefficients=None):
        """
        Initialize the FrozenDistributionMixture class. The distributions must be a list of frozen distributions
        from scipy.stats. The coefficients must be an array of the same length as distributions. If coefficients is
        None, then a uniform distribution is assumed.

        :param distributions: An iterable of frozen distributions from scipy.stats
        :type distributions: Iterable of scipy.stats.rv_continuous
        :param coefficients: An iterable of the same length as distributions, containing the coefficients of the
        mixture distribution. If None, then a uniform distribution is assumed.
        :type coefficients: Iterable of (float or int)
        """
        self.distributions_list = list(distributions)
        if coefficients is None:
            self.coefficients = np.array([1] * len(distributions))
        else:
            self.coefficients = np.array(coefficients)

        for feature in ['pdf', 'logpdf', 'cdf', 'logcdf', 'sf', 'logsf', 'ppf', 'isf']:
            # _f introduced to avoid late binding, otherwise all functions would have the same value for feature
            # https://stackoverflow.com/a/3431699... found through Copilot and ChatGPT
            setattr(self, feature, lambda x, _f=feature: self._combined_feature(_f, x))

        self.rvs = lambda size=1, random_state=None: self._combined_sample(size, random_state)

    def _combined_feature(self, feature, x):
        """
        This function returns the feature of the mixture distribution at values x. It then normalizes the result by
        dividing by the sum of the coefficients. For specific features, see the scipy.stats documentation.

        For all intents and purposes, this function is just the weighted average of the features of the distributions
        in the mixture, weighted by the coefficients.

        :param feature: feature of the distribution to be evaluated
        :type feature: str
        :param x: values at which the feature is evaluated
        :type x: float or numpy.ndarray
        :return: feature of the mixture distribution at values x
        :rtype: float or numpy.ndarray
        """
        feature_sum = sum(
            [coeff * getattr(distr, feature)(x) for coeff, distr in zip(self.coefficients, self.distributions_list)])

        return feature_sum / sum(self.coefficients)

    def _combined_sample(self, size, random_state):
        """
        This function returns a sample from the mixture distribution.

        :param size: size of the sample
        :type size: int or tuple of ints
        :param random_state: Random state for the sampling and the choice across the distributions
        :type random_state: int or None
        :return: sample from the mixture distribution
        :rtype: numpy.ndarray
        """

        if isinstance(size, int):
            # size is an int, so we can use the faster implementation, which wont work with a multidimensional sample
            # this is about twice as fast as the other implementation and will be probably be used most of the time
            sample = sample_mixture(self.distributions_list, size, self.coefficients, random_state=random_state)

        else:
            distribution_samples = np.empty((len(self.distributions_list), *size))
            for idx, distr in enumerate(self.distributions_list):
                distribution_samples[idx] = distr.rvs(size=size, random_state=random_state)
            sample = np_sample_along_first_axis(distribution_samples, p=self.coefficients / sum(self.coefficients),
                                                random_state=random_state)
        return sample
