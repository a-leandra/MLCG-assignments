from PyRT_Common import *
from GaussianProcess import GP, SECov  # Import the Gaussian Process and covariance functions
import matplotlib.pyplot as plt
import numpy as np

def collect_samples(function_list, sample_pos_):
    sample_values = []
    for i in range(len(sample_pos_)):
        val = 1
        for j in range(len(function_list)):
            val *= function_list[j].eval(sample_pos_[i])
        sample_values.append(RGBColor(val, 0, 0))  # for convenience, we'll only use the red channel
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

def compute_estimate_bmc_is(sample_pos_, sample_values_, gp, cosine_term):
    # Add sample positions and values to GP
    gp.add_sample_pos(sample_pos_)
    gp.add_sample_val([sv.r for sv in sample_values_])
    
    # Compute the z vector
    ns = len(sample_pos_)
    z_vec = np.zeros(ns)
    for i in range(ns):
        omega_i = sample_pos_[i]
        # Integrate over the hemisphere
        z_vec[i] = cosine_term.eval(omega_i)
    
    gp.z = z_vec
    return gp.compute_integral_BMC()

# ----------------------------- #
# ---- Main Script Section ---- #
# ----------------------------- #

# #################################################################### #
# STEP 0                                                               #
# Set-up the name of the used methods, and their marker (for plotting) #
# #################################################################### #
methods_label = [('MC', 'o'), ('MC IS', 'v'), ('BMC', 'x'), ('BMC IS', '1')]
n_methods = len(methods_label) # number of tested monte carlo methods

# ######################################################## #
#                   STEP 1                                 #
# Set up the function we wish to integrate                 #
# We will consider integrals of the form: L_i * brdf * cos #
# ######################################################## #
l_i = Constant(1)
kd = 1
brdf = Constant(kd)
cosine_term = CosineLobe(1)
integrand = [l_i, brdf, cosine_term]  # l_i * brdf * cos

# ############################################ #
#                 STEP 2                       #
# Set-up the pdf used to sample the hemisphere #
# ############################################ #
uniform_pdf = UniformPDF()
exponent = 1
cosine_pdf = CosinePDF(exponent)

# ###################################################################### #
# Compute/set the ground truth value of the integral we want to estimate #
# NOTE: in practice, when computing an image, this value is unknown      #
# ###################################################################### #
ground_truth = cosine_term.get_integral()  # Assuming that L_i = 1 and BRDF = 1
print('Ground truth: ' + str(ground_truth))

# ################### #
#     STEP 3          #
# Experimental set-up #
# ################### #
ns_min = 20  # minimum number of samples (ns) used for the Monte Carlo estimate
ns_max = 101  # maximum number of samples (ns) used for the Monte Carlo estimate
ns_step = 20  # step for the number of samples
estimates_per_sample_count = 1000
ns_vector = np.arange(start=ns_min, stop=ns_max, step=ns_step)  # the number of samples to use per estimate
n_estimates = 1  # the number of estimates to perform for each value in ns_vector
n_samples_count = len(ns_vector)

# Initialize a matrix of estimate error at zero
results = np.zeros((n_samples_count, n_methods))  # Matrix of average error

# ################################# #
#          MAIN LOOP                #
# ################################# #
for k, ns in enumerate(ns_vector):
    print(f'Computing estimates using {ns} samples')

    cmc_error_sum = 0.0
    cmc_is_error_sum = 0.0
    bmc_error_sum = 0.0
    bmc_is_error_sum = 0.0

    for _ in range(n_estimates):
        estimate_error_sum = 0.0
        for _ in range(estimates_per_sample_count):
            # Classical Monte Carlo Sampling
            sample_set, sample_prob = sample_set_hemisphere(ns, uniform_pdf)
            sample_values = collect_samples(integrand, sample_set)
            cmc_estimate = compute_estimate_cmc(sample_prob, sample_values)
            cmc_error_sum += abs(ground_truth - cmc_estimate)

            # Monte Carlo Importance Sampling
            sample_set_is, sample_prob_is = sample_set_hemisphere(ns, cosine_pdf)
            sample_values_is = collect_samples(integrand, sample_set_is)
            cmc_is_estimate = compute_estimate_cmc_is(sample_prob_is, sample_values_is)
            cmc_is_error_sum += abs(ground_truth - cmc_is_estimate)

    avg_error_cmc = cmc_error_sum / (n_estimates * estimates_per_sample_count)
    results[k, 0] = avg_error_cmc

    avg_error_cmc_is = cmc_is_error_sum / (n_estimates * estimates_per_sample_count)
    results[k, 1] = avg_error_cmc_is

    # Bayesian Monte Carlo
    cov_func = SECov(l=0.5, noise=0.01)
    gp = GP(cov_func=cov_func, p_func=lambda x: 1)
    gp.add_sample_pos(sample_set)
    gp.add_sample_val([sv.r for sv in sample_values])
    bmc_estimate = gp.compute_integral_BMC()

    avg_error_bmc = abs(ground_truth - bmc_estimate)
    results[k, 2] = avg_error_bmc

    # Bayesian Monte Carlo Importance Sampling
    bmc_is_estimate = compute_estimate_bmc_is(sample_set_is, sample_values_is, gp, cosine_term)
    avg_error_bmc_is = abs(ground_truth - bmc_is_estimate)
    results[k, 3] = avg_error_bmc_is

    print(f'Average absolute error Classical Monte Carlo for {ns} samples: {avg_error_cmc}')
    print(f'Average absolute error MC Importance Sampling for {ns} samples: {avg_error_cmc_is}')
    print(f'Average absolute error Bayesian Monte Carlo for {ns} samples: {avg_error_bmc}')
    print(f'Average absolute error Bayesian Monte Carlo Importance Sampling for {ns} samples: {avg_error_bmc_is}')

# ################################################################################################# #
# Create a plot with the average error for each method, as a function of the number of used samples #
# ################################################################################################# #
for k in range(len(methods_label)):
    method = methods_label[k]
    plt.plot(ns_vector, results[:, k], label=method[0], marker=method[1])
plt.legend()
plt.xlabel('Number of Samples')
plt.ylabel('Absolute Error')
plt.title('Comparison of MC, MC IS, BMC, and BMC IS Methods')
plt.savefig('figure.png')  # Save figure to file
plt.show()
