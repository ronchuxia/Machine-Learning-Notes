In UCB, we estimated a range that the mean lies within. In Thompson Sampling, we estimate a distribution over the mean.

**Thompson Sampling:**
Given samples, estimate a distribution over the mean $\mu$.
$$p(\mu | r) \propto p(r | \mu) p(\mu)$$

# Gaussian Reward
Assume both reward and mean reward follows **Gaussian distribution**:
$$p(r | \mu) \sim \mathcal N(\mu, \sigma)$$
$$p(\mu) \sim \mathcal N(\mu_0, \sigma_0)$$
Then the posterior also follows **Gaussian distribution**:
$$p(\mu|r) \sim \mathcal N(\mu', \sigma')$$
$$\mu' = \frac{1}{\frac{1}{\sigma_0^2} + \frac{1}{\sigma^2}}(\frac{\mu_0}{\sigma_0^2} + \frac{r}{\sigma^2})$$
$$(\sigma')^2 = \left( \frac{1}{\sigma_0^2} + \frac{1}{\sigma^2} \right)^{-1}$$

Repeat the above process for any new observations.

# Bernoulli Reward
Assume reward follows **Bernoulli distribution**:
$$p(r | \theta) = \theta^r (1 - \theta)^{1-r}$$
Assume mean reward follows **Beta distribution**:
$$p(\theta) = \frac{\Gamma(\alpha + \beta)}{\Gamma(\alpha) + \Gamma(\beta)} \theta^{\alpha-1} (1 - \theta)^{\beta-1}$$
$$\Gamma(n) = (n - 1)!$$
Then the posterior also follows a **Beta distribution**:
$$\begin{aligned}
p(\theta | r) & \propto p(r|\theta) p(\theta)\\
& \propto \theta^r (1 - \theta)^{1-r} \theta^{\alpha-1} (1 - \theta)^{\beta-1}\\
& \propto \theta^{(r + \alpha) - 1} (1 - \theta)^{(1-r+\beta)-1}\\
& \propto \theta^{\alpha' - 1} (1 - \theta)^{\beta' - 1}
\end{aligned}$$

**Conjugate prior:**
https://en.wikipedia.org/wiki/Conjugate_prior

# BayesUCB
**BayesUCB** uses the formulas for conjugate priors to estimate a distribution over the mean (the posterior), and picks an arm based on an optimistic estimate, using the posterior:
$$\text{BayesUCB}_i = \mu'_i + k\sigma'_i$$

# Thompson Sampling
1. Estimate a posterior for each arm.
2. For each arm, sample a mean from its posterior distribution.
3. Choose the arm with the highest **sampled** mean.
4. Take the action, observe the reward and go back to step 1.

Actions with low uncertainty will be exploited.
Actions with high uncertainty will be explored.