On-policy: require data (trajectories) collected from current/recent policy.
Off-policy: can use data (trajectories) collected from any policy.

# Off-Policy MC
With **importance sampling**, we can use trajectories collected from another policy $b(a | s)$ (e.g. an older policy) to estimate the value function $V^\pi(s)$ of the current policy $\pi(a|s)$.
$$\begin{aligned}
V^\pi(s) & = \mathbb E_\pi \left[ R | s_0 = s \right]\\
& = \mathbb E_b \left[ \frac{p_\pi(\tau)}{p_b(\tau)} R | s_0 = s \right]\\
& = \mathbb E_b \left[ \frac{\prod_{k=t}^{T-1} \pi(a_k | s_k) p(s_{k+1} | s_k, a_k)}{\prod_{k=t}^{T-1} b(a_k | s_k) p(s_{k+1} | s_k, a_k)} R | s_0 = s \right]\\
& = \mathbb E_b \left[ \frac{\prod_{k=t}^{T-1} \pi(a_k | s_k)}{\prod_{k=t}^{T-1} b(a_k | s_k)} R | s_0 = s \right]
\end{aligned}$$

In theory, $V^\pi(s)$ will converge to the actual value. However, it may require many samples to converge. There are many tricks to reduce the number of samples required:
- Discounting-aware importance sampling.
- Per-reward importance sampling.

For continuous states and/or actions:
$$\rho_i = \prod_t \frac{\pi(a_t | s_t)}{b(a_t | s_t)}$$
$$y_i = \sum_t \gamma^t r_t$$
$$\text{Loss} = \frac{1}{n} \sum_i \rho_i \left( Q_\theta(s_i, a_i) - y_i \right)^2$$

# Off-Policy TD
**Q-Learning:**
$$y = r(s) + \gamma \max_{a'} Q_\theta(s', a')$$
With this learning target, Q-learning will converge to the optimal Q function $Q^*(s, a)$.

**Expected SARSA**:
$$y = r(s) + \gamma \mathbb E_{a' \sim \pi(a'|s')} [Q_\theta(s', a')]$$
- Lower variance than SARSA.
- Off-policy.

For continuous actions, we can use Monte Carlo to estimate the expectation.
1. Sample M actions from the current policy $\pi$: 
$$a_1', a_2', \cdots, a_M' \sim \pi(s')$$
2. Compute the expected SARSA target:
$$y = r(s) + \gamma \frac{1}{M} \sum_m Q_\theta(s', a_m')$$

# Soft Actor-Critic (SAC)
Sometimes we might want to learn a stochastic policy for the following reasons:
- If the environment is partially observed, stochastic policy may have better performance than deterministic policy.
- Policy gradient requires a distribution of actions.
- Stochastic policies explore naturally.

To learn a stochastic policy, we add noise to a deterministic policy:
$$\pi_b(s) = \pi_\theta(s) + \epsilon$$
e.g. Gaussian noise. The policy network outputs the mean $\pi_\theta(s)$ and variance $\sigma(s)$.

The variance will often decrease to 0 to try to maximize rewards. However, this could be a problem when:
- The policy is stuck in a local optima.
- The environment changes.
- No exploration!

To prevent the variance from going to 0 too fast, we can **add an entropy term to the reward** to encourage a large variance.
1. Add an entropy term to the return.
$$\text{Return} = \left( \sum_t \gamma^t r(s_t, a_t) \right) + \alpha \mathcal{H}_\pi(a_t | s_t)$$
2. Add an entropy term to every reward.
$$\text{Return} = \sum_t \gamma^t (r(s_t, a_t) + \alpha \mathcal H_\pi(s_t, a_t))$$

Option 2 is better because the policy is rewarded for having high future entropy. Maximizing the future entropy leads the policy to find robust trajectories.

We can show that option 2 will converge to a policy such that:
$$\pi(a | s) \propto e^{Q^\pi(s, a)}$$
This is called **Maximum Entropy RL.**

**Soft Actor-Critic (SAC)** is an off-policy method combining expected SARSA with maximum entropy RL.

NOTE: The entropy coefficient is very important.