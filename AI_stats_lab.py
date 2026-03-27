import math
import numpy as np


def bernoulli_log_likelihood(data, theta):
    data = list(data)
    if len(data) == 0:
        raise ValueError("Data must not be empty.")
    if not (0 < theta < 1):
        raise ValueError("theta must satisfy 0 < theta < 1.")
    for x in data:
        if x not in (0, 1):
            raise ValueError("Data must contain only 0s and 1s.")

    return sum(x * math.log(theta) + (1 - x) * math.log(1 - theta) for x in data)


def bernoulli_mle_with_comparison(data, candidate_thetas=None):
    data = list(data)
    if len(data) == 0:
        raise ValueError("Data must not be empty.")
    for x in data:
        if x not in (0, 1):
            raise ValueError("Data must contain only 0s and 1s.")

    if candidate_thetas is None:
        candidate_thetas = [0.2, 0.5, 0.8]

    n = len(data)
    num_successes = int(sum(data))
    num_failures = n - num_successes
    mle = num_successes / n

    log_likelihoods = {}
    for theta in candidate_thetas:
        log_likelihoods[theta] = bernoulli_log_likelihood(data, theta)

    best_candidate = max(candidate_thetas, key=lambda t: log_likelihoods[t])

    return {
        "mle": mle,
        "num_successes": num_successes,
        "num_failures": num_failures,
        "log_likelihoods": log_likelihoods,
        "best_candidate": best_candidate,
    }


def poisson_log_likelihood(data, lam):
    data = list(data)
    if len(data) == 0:
        raise ValueError("Data must not be empty.")
    if lam <= 0:
        raise ValueError("lam must be > 0.")
    for x in data:
        if x < 0 or not float(x).is_integer():
            raise ValueError("Data must contain nonneg integers.")

    return sum(x * math.log(lam) - lam - math.lgamma(x + 1) for x in data)


def poisson_mle_analysis(data, candidate_lambdas=None):
    data = list(data)
    if len(data) == 0:
        raise ValueError("Data must not be empty.")
    for x in data:
        if x < 0 or not float(x).is_integer():
            raise ValueError("Data must contain nonneg integers.")

    if candidate_lambdas is None:
        candidate_lambdas = [1.0, 3.0, 5.0]

    n = len(data)
    total_count = int(sum(data))
    sample_mean = total_count / n
    mle = sample_mean

    log_likelihoods = {}
    for lam in candidate_lambdas:
        log_likelihoods[lam] = poisson_log_likelihood(data, lam)

    best_candidate = max(candidate_lambdas, key=lambda l: log_likelihoods[l])

    return {
        "mle": mle,
        "sample_mean": sample_mean,
        "total_count": total_count,
        "n": n,
        "log_likelihoods": log_likelihoods,
        "best_candidate": best_candidate,
    }