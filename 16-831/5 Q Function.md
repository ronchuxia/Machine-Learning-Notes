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

# Policy Evaluation: Monte Carlo
We can use a neural network to represent the Q-function. The learning target can be computed using Monte Carlo Estimation.
$$\mathrm{Loss} = \frac{1}{n} \sum(Q_\theta(s, a) - R)$$

# Policy Improvement
Proposed new policy:
$$\pi'(s) = \arg\max_a Q^\pi(s, a)$$

If we take the new action $\pi'(s)$ at state $s$, then follow the old policy, the expected return will increase. In fact, according to the policy improvement theorem, if we always take the new action $\pi'(s)$ at state $s$ (meaning we adopt the new policy $\pi'$), the expected return will also increase. Thus the new policy is better than the old policy.

Policy improvement theorem.
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

Note: For continuous actions, replace $\arg\max$ with gradient ascent.

# Policy Evaluation: TD (Temporal Difference)
Recursive relationship (Bellman equations):
$$V(s) \leftarrow r + \gamma \sum_a \pi(a | s) \sum_{s'} p(s' | s, a) V(s')$$
$$Q(s, a) \leftarrow r + \pi(a|s) \sum_{s'} p(s'|s, a) \sum_{a'} Q(s', a')$$
Repeat the above process, $V^\pi$ and $Q^\pi$ will converge. But this requires:
- Known dynamics
- Discrete states
- Discrete actions

Bootstrapping:
$$V(s) \leftarrow r + \gamma V(s')$$
$$Q(s, a) \leftarrow r + \gamma Q(s', a')$$
Repeat the above process, $V^\pi$ and $Q^\pi$ will converge. 

For unknown dynamics and continuous states and actions,
$$\mathrm{Loss} = \frac{1}{n} \sum (Q(s_i, a_i) - y_i)^2$$
$$y_i = r_i + Q(s_i', a_i')$$

Note: After we update Q(s,a) with one gradient step, the targets change! We have to recompute them!