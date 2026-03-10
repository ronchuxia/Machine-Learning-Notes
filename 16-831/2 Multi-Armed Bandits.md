- greedy
- $\epsilon$-greedy
- UCB
# Multi-Armed Bandits
Multi-armed bandits: At each time step, choose an action, observe a reward, system restarts.

Application: Generate several choices and select the best choice.

Problem definition:
Reward follows a conditional distribution:
$$r \sim p(R | A = a)$$
We define the value of action $a$ as the expected reward given action $a$:
$$q(a) = \mathbb{E}[R | A = a]$$
We want to choose the action with the maximum mean reward:
$$a = \arg\max_a q(a)$$

However, we can only observe sampled rewards, therefore we don't know the actual reward distribution. 

We can estimate of value as sample averages.
$$Q_t(a) = \frac{\sum_{i=1}^{t-1} R_i \cdot \mathbf{1}_{A_i = a}}{\sum_{i=1}^{t-1} \mathbf{1}_{A_i = a}}$$

# Greedy Action Selection
Choose the greedy best action.
$$A_t^* = \arg\max_a Q_t(a)$$
- If $A_t = A_t^*$: Exploiting.
- If $A_t \neq A_t^*$: Exploring.

**Explore vs exploit conflict**:
- Early vs late: Explore more in the early stages.
- Risk: Less risk to explore actions that are close to the greedy best action.
- Uncertainty: Explore actions with higher uncertainty.

# $\epsilon$-Greedy Action Selection
**$\varepsilon$-greedy action selection**:
- With probability $1 - \epsilon$, choose the greedy action
- With probability $\epsilon$, choose an action uniformly at random

The 10-armed bandit testbed: $\varepsilon$-greedy finds a better action.

# Action-Value Function
**Online update of action-value function**:
$$Q_{n+1} = Q_n + \frac{1}{n} \left[ R_n - Q_n \right]$$

**Non-stationary update of action-value function**:
$$\begin{align}
Q_{n+1} & = Q_n + \alpha [R_n - Q_n]\\
& = (1-\alpha)^n Q_1 + \sum_{i=1}^n \alpha (1 - \alpha)^{n-i} R_i
\end{align}$$
- Effect of bias from initial value $Q_1$ gets smaller and smaller.
- Recent rewards are weighted more.

**Optimistic initial values**:
$$Q_1(a) = 5$$
Encourages exploration in the initial steps. Will lead to better performance later.

NOTE: Optimistic initial values only encourages exploration in the initial steps. This is not very helpful for non-stationary problems.

# Upper Confidence Bound (UCB)
The exploration method should adapt to the samples collected.

If given the uncertainty, we should follow the **Upper Confidence Bound (UCB)** rule: pick the one with the **highest optimistic estimate**:
$$\text{UCB}_i = \hat \mu_i + \epsilon_i$$

To measure the uncertainty $\epsilon$, we want to say, with probability $1 - \delta$, the true mean $\mu$ is at most $\hat \mu + \epsilon$:
$$P(\mu < \hat \mu + \epsilon) \geq 1 - \delta$$
Assume gaussian distribution:
$$P(|\hat \mu - \mu| \geq \epsilon) \leq 2e^{-N\epsilon^2 \ (2\sigma^2)}$$
Therefore, the uncertainty:
$$\epsilon = \sqrt{\frac{2\sigma^2}{N} \log \frac{1}{\delta}}$$
- $N$: Number of samples.
- $\sigma$: Std of reward, assumed to be known.

Choose the most optimistic action.

How to choose error rate $\delta$:
- Too large: Often select suboptimal arm, reduced performance.
- Too small: Estimates not sufficiently optimistic, may never select the optimal arm.

Reduce $\delta$ over time:
$$\delta = \frac{1}{t^2}$$
UCB:
$$\text{UCB}_i = \hat \mu_i + c \sqrt{\frac{4\sigma_i^2 \log t}{N_i}}$$
If the variance $\sigma_i$ is unknown, this can be treated as a hyperparameter:
$$\text{UCB}_i = \hat \mu_i + c \sqrt{\frac{\log t}{N_i}}$$

# Gradient Bandits
Learn a distribution over actions that maximizes the expected reward. Use gradient ascent to update the parameters.