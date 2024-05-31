from PyRT_Common import *
from GaussianProcess import GP, SECov
import matplotlib.pyplot as plt
import numpy as np

def collect_samples(function_list, sample_pos_):
    sample_values = []
    for i in range(len(sample_pos_)):
        val = 1
        for j in range(len(function_list)):
            val *= function_list[j].eval(sample_pos_[i])
        sample_values.append(RGBColor(val, 0, 0))
    return sample_values

def compute_estimate_cmc(sample_prob_, sample_values_):
    cmc_estimate = 0.0
    for sample_prob, sample_value in zip(sample_prob_, sample_values_):
        cmc_estimate += sample_value.r / sample_prob

    cmc_estimate /= len(sample_prob_)
    return cmc_estimate

def compute_estimate_cmc_is(sample_prob_, sample_values_):
    cmc_is_estimate = 0.0
    for sample_prob, sample_value in zip(sample_prob_, sample_values_):
        cmc_is_estimate += sample_value.r / sample_prob

    cmc_is_estimate /= len(sample_prob_)
    return cmc_is_estimate

def compute_estimate_bmc(sample_pos_, sample_values_, gp):
    gp.add_sample_pos(sample_pos_)
    gp.add_sample_val([sv.r for sv in sample_values_])
    return gp.compute_integral_BMC()

def compute_estimate_bmc_is(sample_pos_, sample_values_, gp):
    gp.add_sample_pos(sample_pos_)
    gp.add_sample_val([sv.r for sv in sample_values_])
    return gp.compute_integral_BMC()

# Main Script Section

methods_label = [('MC', 'o'), ('MC IS', 'v'), ('BMC', 'x'), ('BMC IS', '1')]
n_methods = len(methods_label)

l_i = Constant(1)
kd = 1
brdf = Constant(kd)
cosine_term = CosineLobe(3)
integrand = [l_i, brdf, cosine_term]  # l_i * brdf * cos

uniform_pdf = UniformPDF()
exponent = 1
cosine_pdf = CosinePDF(exponent)

ground_truth = cosine_term.get_integral()
print('Ground truth: ' + str(ground_truth))

ns_min = 20
ns_max = 101
ns_step = 20
estimates_per_sample_count = 1000
ns_vector = np.arange(start=ns_min, stop=ns_max, step=ns_step)
n_estimates = 1
n_samples_count = len(ns_vector)

results = np.zeros((n_samples_count, n_methods))

for k, ns in enumerate(ns_vector):
    print(f'Computing estimates using {ns} samples')

    cmc_error_sum = 0.0
    cmc_is_error_sum = 0.0
    bmc_error_sum = 0.0
    bmc_is_error_sum = 0.0

    for _ in range(n_estimates):
        estimate_error_sum = 0.0
        for _ in range(estimates_per_sample_count):
            sample_set, sample_prob = sample_set_hemisphere(ns, uniform_pdf)
            sample_values = collect_samples(integrand, sample_set)
            cmc_estimate = compute_estimate_cmc(sample_prob, sample_values)
            cmc_error_sum += abs(ground_truth - cmc_estimate)

            sample_set_is, sample_prob_is = sample_set_hemisphere(ns, cosine_pdf)
            sample_values_is = collect_samples(integrand, sample_set_is)
            cmc_is_estimate = compute_estimate_cmc_is(sample_prob_is, sample_values_is)
            cmc_is_error_sum += abs(ground_truth - cmc_is_estimate)

        avg_error_cmc = cmc_error_sum / (n_estimates * estimates_per_sample_count)
        results[k, 0] = avg_error_cmc

        avg_error_cmc_is = cmc_is_error_sum / (n_estimates * estimates_per_sample_count)
        results[k, 1] = avg_error_cmc_is

        # Bayesian Monte Carlo
        sample_set, sample_prob = sample_set_hemisphere(ns, uniform_pdf)
        sample_values = collect_samples(integrand, sample_set)
        cov_func = SECov(l=0.5, noise=0.01)
        gp = GP(cov_func=cov_func, p_func=lambda x: 1)
        bmc_estimate = compute_estimate_bmc(sample_set, sample_values, gp)
        avg_error_bmc = abs(ground_truth - bmc_estimate)
        results[k, 2] = avg_error_bmc

        # Bayesian Monte Carlo Importance Sampling
        sample_set_is, sample_prob_is = sample_set_hemisphere(ns, cosine_pdf)
        sample_values_is = collect_samples(integrand, sample_set_is)
        gp_is = GP(cov_func=cov_func, p_func=lambda x: CosineLobe(1).eval(x))
        bmc_is_estimate = compute_estimate_bmc_is(sample_set_is, sample_values_is, gp_is)
        avg_error_bmc_is = abs(ground_truth - bmc_is_estimate)
        results[k, 3] = avg_error_bmc_is

    print(f'Average absolute error Classical Monte Carlo for {ns} samples: {avg_error_cmc}')
    print(f'Average absolute error MC Importance Sampling for {ns} samples: {avg_error_cmc_is}')
    print(f'Average absolute error Bayesian Monte Carlo for {ns} samples: {avg_error_bmc}')
    print(f'Average absolute error Bayesian Monte Carlo Importance Sampling for {ns} samples: {avg_error_bmc_is}')

for k in range(len(methods_label)):
    method = methods_label[k]
    plt.plot(ns_vector, results[:, k], label=method[0], marker=method[1])
plt.legend()
plt.xlabel('Number of Samples')
plt.ylabel('Absolute Error')
plt.title('Comparison of MC, MC IS, BMC, and BMC IS Methods')
plt.savefig('figure.png')
plt.show()
