from PyRT_Common import *
from math import exp
import numpy as np
from numpy import ndarray
from abc import ABC, abstractmethod


class CovarianceFunction(ABC):

    @abstractmethod
    def eval(self, omega_i, omega_j):
        pass


class SobolevCov(CovarianceFunction):
    def __init__(self, s=1.4):
        self.s = s

    def eval(self, omega_i, omega_j):
        s = self.s
        r = Length(omega_i - omega_j)
        return (2 ** (2 * s - 1)) / s - r ** (2 * s - 2)


class SECov(CovarianceFunction):
    def __init__(self, l, noise):
        self.l = l
        self.noise = noise

    def eval(self, omega_i, omega_j):
        r = Length(omega_i - omega_j)
        return exp(-(r ** 2) / (2 * self.l ** 2))


class GP:

    def __init__(self, cov_func, p_func, noise_=0.01):
        self.cov_func = cov_func
        self.p_func = p_func
        self.noise = noise_
        self.samples_pos = None
        self.samples_val = None
        self.invQ = None
        self.z = None
        self.weights = None

    def add_sample_pos(self, samples_pos_):
        self.samples_pos = samples_pos_
        self.invQ = self.compute_inv_Q()
        self.z = self.compute_z()
        self.weights = self.z @ self.invQ

    def add_sample_val(self, samples_val_):
        self.samples_val = samples_val_

    def compute_inv_Q(self):
        n = len(self.samples_pos)
        Q = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                Q[i, j] = self.cov_func.eval(self.samples_pos[i], self.samples_pos[j])

        Q += np.eye(n) * self.noise ** 2
        return np.linalg.inv(Q)

    def compute_z(self):
        uniform_pdf = UniformPDF()
        ns_z = 50000
        sample_set_z, probab = sample_set_hemisphere(ns_z, uniform_pdf)
        ns = len(self.samples_pos)
        z_vec = np.zeros(ns)

        for i in range(ns):
            omega_i = self.samples_pos[i]
            integrand_values = np.zeros(ns_z)
            for j in range(ns_z):
                omega_j = sample_set_z[j]
                integrand_values[j] = self.cov_func.eval(omega_i, omega_j) * self.p_func(omega_j)

            z_vec[i] = np.mean(integrand_values / probab)

        return z_vec

    def compute_integral_BMC(self):
        return np.dot(self.weights, self.samples_val)
