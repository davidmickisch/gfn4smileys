# %%
import numpy as np

true_mean = 0.666
num_samples = 128
# draw monte carlo samples from bernoulli distribution
def monte_carlo_bernoulli(true_mean, num_samples):
    return np.random.binomial(1, true_mean, num_samples)

samples = monte_carlo_bernoulli(true_mean, num_samples)

# %%
# plot running average of samples
import matplotlib.pyplot as plt
plt.plot(np.cumsum(samples) / np.arange(1, num_samples + 1))

# %%
# plot absolute error from true mean
plt.plot(np.abs(np.cumsum(samples) / np.arange(1, num_samples+1) - true_mean))

# %%
# print absolute error from true mean for all samples
print(np.abs(np.sum(samples) / num_samples - true_mean))

# %%
