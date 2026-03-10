# Q Function
$$Q^\pi(s, a) = E_{a_{1:\infty}} \left[ \sum_i \gamma^ir_i | s_0 = s, a_0 = a \right]$$

Property:
$$\pi^*(s) = \arg\max_a Q^{\pi^*}(s, a)$$
But we don't have the Q-function for the optimal policy.

# Policy Iteration
Requirments (approximate policy iteration):
- Action: continuous or discrete.
- State: continuous or discrete.

![[Policy Iteration.png]]

**Policy iteration**:
Repeat until convergence:
- **Policy evaluation**.
- **Policy improvement**.

# Policy Evaluation: Monte Carlo
For continuous states and/or actions, we can use a neural network to represent the Q-function. The learning target can be computed using Monte Carlo Estimation.
$$\mathrm{Loss} = \frac{1}{n} \sum(Q_\theta(s, a) - R)$$

# Policy Improvement
Proposed new policy:
$$\pi'(s) = \arg\max_a Q^\pi(s, a)$$

If we take the new action $\pi'(s)$ at state $s$, then follow the old policy, the expected return will increase. In fact, according to the policy improvement theorem, if we always take the new action $\pi'(s)$ at state $s$ (meaning we adopt the new policy $\pi'$), the expected return will also increase. Thus the new policy is better than the old policy.

**Policy improvement theorem**.
$$\begin{align}
V^{\pi_{i}}(s) & = R(s, \pi_{i}(s)) + \gamma \sum_{s' \in S} P(s' | s, \pi_{i}(s)) V^{\pi_{i}}(s')\\
& \leq \max_{a \in A} Q^{\pi_i}(s, a)\\
& = R(s, \pi_{i+1}(s)) + \gamma \sum_{s' \in S} P(s' | s, \pi_{i+1}(s)) V^{\pi_i}(s')\\
& \leq R(s, \pi_{i+1}(s)) + \gamma \sum_{s' \in S} P(s' | s, \pi_{i+1}(s)) \max_{a \in A} Q^{\pi_i}(s')\\
& = R(s, \pi_{i+1}(s)) + \gamma \sum_{s' \in S} P(s' | s, \pi_{i+1}(s)) \left[ R(s', \pi_{i+1}(s')) + \gamma \sum_{s'' \in S} P(s'' | s', \pi_{i+1}(s')) V^{\pi_i}(s'') \right]\\
& = R(s, \pi_{i+1}(s)) + \gamma \sum_{s' \in S} P(s' | s, \pi_{i+1}(s)) R(s', \pi_{i+1}(s')) + \gamma^2 \sum_{s' \in S} P(s' | s, \pi_{i+1}(s)) \sum_{s'' \in S} P(s'' | s', \pi_{i+1}(s')) V^{\pi_i}(s'')\\
& \cdots\\
& \leq V^{\pi_{i+1}}(s)
\end{align}$$

For discrete state and action spaces, tabular policies and value functions, policy iteration will converge to the optimal policy.

NOTE: For continuous actions, replace $\arg\max$ with gradient ascent.

# Policy Evaluation: TD (Temporal Difference)
Recursive relationship (**Bellman equations**):
$$V^\pi(s) = r + \gamma \sum_a \pi(a | s) \sum_{s'} p(s' | s, a) V^\pi(s')$$
$$Q^\pi(s, a) = r + \pi(a|s) \sum_{s'} p(s'|s, a) \sum_{a'} Q^\pi(s', a')$$

Dynamic programming (**Bellman operator**):
1. Initialize $V^\pi(s), Q^\pi(s, a)$ randomly.
2. Repeat until convergence:
	- For each state $s$:
$$V^\pi(s) \leftarrow r + \gamma \sum_a \pi(a | s) \sum_{s'} p(s' | s, a) V^\pi(s')$$
	- For each state $s$ and action $a$:
$$Q^\pi(s, a) \leftarrow r + \pi(a|s) \sum_{s'} p(s'|s, a) \sum_{a'} Q^\pi(s', a')$$

$V^\pi$ and $Q^\pi$ will converge to the actual values. But this requires:
- Known dynamics
- Discrete states
- Discrete actions

**Bootstrapping (Temporal Difference Learning):**
1. Initialize $V^\pi(s), Q^\pi(s, a)$ randomly.
2. Repeat until convergence:
	1. Execute the policy $\pi$ from state $s$ and get a sequence $(s, a, r, s')$.
	2. Bootstrap:
$$V(s)^\pi \leftarrow r + \gamma V^\pi(s')$$
$$Q^\pi(s, a) \leftarrow r + \gamma Q^\pi(s', a')$$

$V^\pi$ and $Q^\pi$ will converge to the actual values.

For unknown dynamics and continuous states and/or actions, we can use a neural network to represent the Q-function. The learning target can be computed using Temporal Difference Learning.
$$\mathrm{Loss} = \frac{1}{n} \sum_i (Q(s_i, a_i) - y_i)^2$$
$$y_i = r_i + Q(s_i', a_i')$$

NOTE: After we update $Q(s,a)$ with one gradient step, the targets change! We have to recompute them!

# Generalized Policy Iteration
In policy evaluation, we update $Q^\pi(s, a)$ until it converges to the actual value. This might take a long time. But actually we don't need the value function to be very precise to improve the policy.

**Generalized policy iteration**:
Repeat until convergence:
1. Policy evaluation:
	- Repeat for n steps (instead of repeat until convergence):
		1. Execute the policy $\pi$ from state $s$ and get a sequence $(s, a, r, s')$.
		2. Update $Q^\pi(s, a)$.
2. Policy improvement.

# Generalized Advantage Estimation (Lambda Returns)
0-step return:
$$R_t^{(0)} = V(s_t)$$

1-step return (**TD(0)**):
$$R_t^{(1)} = r_t + \gamma V(s_{t+1})$$
High bias, low variance.

2-step return (**TD(1)**):
$$R_t^{(2)} = r_t + \gamma r_{t+1} + \gamma^2 V(s_{t+2})$$

n-step return (**TD(n-1)**):
$$R_t^{(n)} = \sum_{k=0}^{n-1} \gamma^k r_{t+k} + \gamma^{n} V(s_{t+n})$$

**MC (TD($\infty$)):**
$$R_t^{(\infty)} = \sum_{k=0}^\infty \gamma^k r_{t+k}$$
High variance, low bias.

The **$\lambda$-return** is a weighted average of all TD targets:
$$R_t^\lambda = (1 - \lambda) \sum_{n = 1}^\infty \lambda^{n-1} R_t^{(n)}$$

Consider a truncated trajectory of T steps (the trajectory has infinite steps, but we truncate the first T steps). To compute the $\lambda$-returns efficiently, we can use the following recursive relationship:
1. Use the value function of the last state to start bootstrapping.
$$R_T^\lambda = V(s_T)$$
2. Compute $\lambda$-returns backwards.
$$R_t^\lambda = r_t + \gamma \left( (1 - \lambda) V(s_{t+1}) + \lambda R_{t+1}^\lambda \right)$$

**Generalized advantage estimation (GAE)** estimates the advantage $A_t = Q(s_t, a_t) − V(s_t)$ using the lambda return.
$$A_t^\lambda = R_t^\lambda - V(s_t)$$

It is equivalent to the following formulation:
$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$
$$A_t^\lambda = \sum_{k=0}^\infty (\gamma \lambda)^k \delta_{t+k}$$
Which can be computed efficiently using the following recursive relationship:
$$A_t^\lambda = \delta_t + \gamma \lambda A_{t+1}^\lambda$$