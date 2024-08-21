# Markov Process/Markov Chain
Sequence of random states with Markov property
- S is a (finite) set of states $(s \in S)$ 
- P is dynamics/transition model that specifices $p(s_{t+1} = s' | s_t = s)$

状态有限时，P 可以写成矩阵的形式：
$$P = \begin{bmatrix}
p(s_1 | s_1) & p(s_2 | s_1) & \cdots & p(s_N | s_1) \\
p(s_1 | s_2) & p(s_2 | s_2) & \cdots & p(s_N | s_2) \\
\vdots & \ddots & \cdots & \vdots \\
p(s_1 | s_N) & p(s_2 | s_N) & \cdots & p(s_N | s_N) \\
\end{bmatrix}$$

# Markov Reward Process
Markov Chain + rewards
- S is a (finite) set of states $(s \in S)$ 
- P is dynamics/transition model that specifices $p(s_{t+1} = s' | s_t = s)$
- R is a reward function $R(s_t = s) = E[r_t|s_t = s]$ 
- Discount factor $\gamma \in [0, 1]$
## Return & Value Function
- Horizon H
	- Number of time steps in each episode 
	- Can be infinite 
	- Otherwise called finite Markov reward process 
- Return $G_t$
	- Discounted sum of rewards from time step t to horizon H 
	- $G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots + \gamma^{H-1} r_{t+H-1}$
- State Value Function $V(s)$
	- Expected return from starting in state s 
	- $V(s) = E[G_t|s_t = s] = E[r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots + \gamma^{H-1} r_{t+H-1}|s_t = s]$
## Analytic Solution for Computing Value of a MRP
State Value Function 满足贝尔曼方程：
$$V(s) = R(s) + \gamma \sum_{s'} P(s' | s) V(s')$$
对于有限状态 MRP，可以将上式写成矩阵的形式：
$$V = R + \gamma P V$$
因此：
$$V = (I - \gamma P)^{-1} R$$
这个过程需要求矩阵的逆，复杂度为 $O(N^3)$。

注：P 是行随机矩阵，其特征值在 0 到 1 之间。因此，当 $\gamma < 1$ 时，$I - \gamma P$ 是可逆的。
## Iterative Algorithm for Computing Value of a MRP
除了上述方法，也可以使用迭代法求解 MRP 的价值函数：
- Initialize $V_0(s) = 0$ for all s 
- For k = 1 until convergence 
	- For all s in S 
$$V_k(s) = R(s) + γ \sum_{s'∈S} P(s'|s)V_{k−1}(s')$$
每次迭代的复杂度为 $O(N^2)$。

可以证明，迭代法会收敛到唯一解。

# Markov Decision Process
Markov Reward Process + actions
- S is a (finite) set of states ($s \in S$) 
- A is a (finite) set of actions ($a ∈ A$)
- P is dynamics/transition model that specifices $p(s_{t+1} = s' | s_t = s, a_t = a)$
- R is a reward function $R(s_t = s, a_t = a) = E[r_t|s_t = s, a_t = a]$ 
- Discount factor $\gamma \in [0, 1]$
## Policy
Policy specifies what action to take in each state.
$$π(a|s) = P(a_t = a|s_t = s)$$

MDP $(S, A, P, R, \gamma)$ + Policy $π(a|s)$ = MRP $(S, P^\pi, R^\pi, \gamma)$ 
其中：
- $R^π(s) = \sum_{a∈A} π(a|s) R(s, a)$ 
- $P^π(s'|s) = \sum_{a∈A} π(a|s) P(s'|s, a)$

这表明我们可以使用迭代法计算一个 MDP 策略的价值函数：
$$V_k^\pi(s) = \sum_{a \in A} \pi(a | s) \left[ R(s, a) + γ \sum_{s'∈S} P(s'|s, a) V^\pi_{k−1}(s') \right]$$
对于确定策略：
$$V^\pi_k(s) = R(s, \pi(s)) + γ \sum_{s'∈S} P(s'|s, \pi(s))V^\pi_{k−1}(s')$$
## Optimal Policy
$$\pi^*(s) = \arg\max_\pi V^\pi(s)$$
可以证明：
- 最优价值函数唯一
- 对于 infinite-horizon MDP
	- 最优策略是确定策略
	- 最优策略是静态策略
	- 最优策略不唯一
## Policy Iteration
Policy iteration computes infinite horizon value of a policy and then improves that policy.
**Policy Iteration = Policy evaluation + Policy imporvement.**
- Set i = 0 
- Initialize $π_0(s)$ randomly for all states s 
- While $i == 0$ or $||π_i − π_{i−1}||_1 > 0$ (L1-norm, measures if the policy changed for any state): 
	- $V^{π_i} ←$ Policy evaluation of $π_i$ 
	- $π_{i+1} ←$ Policy improvement 
	- i = i + 1
### Policy evaluation
- Loop until convergence:
	- Use **Bellman backup operator for a particular policy** on value function
$$B^{\pi_i} V^{\pi_i}_{k+1}(s) = r(s, \pi_i(s)) + \gamma \sum_{s' \in S} P(s' | s, \pi_i(s)) V^{\pi_i}_k(s')$$
可以证明，Bellman backup operator for a particular policy 是一个 contraction operator（[[A1]]）。因此，迭代法会收敛到唯一解。
### Policy improvement
- Compute state-action value $Q^{\pi}(s, a)$ of a policy $π_i$
	- Expected return from starting in state s, take action a, then follow the policy π
$$Q^{\pi_i}(s, a) = R(s, a) + \gamma \sum_{s' \in S} P(s' | s, a) V^{\pi_i}(s')$$
- Compute new policy $π_{i+1}$, for all $s ∈ S$
$$\pi_{i + 1}(s) = \arg\max_{a \in A} Q^{\pi_i}(s, a)$$

Policy improvement 表明，在状态 s，如果采取新的动作 a，然后再采取原来的策略 $\pi$，价值函数会比原来的大，即：
$$R(s, \pi_{i + 1}(s)) + \gamma \sum_{s' \in S} P(s' | s, \pi_{i + 1}(s))) V^{\pi_i}(s') \geq R(s, \pi_{i}(s)) + \gamma \sum_{s' \in S} P(s' | s, \pi_{i}(s))) V^{\pi_i}(s') = V^{\pi_i}(s)$$

事实上，如果全程都采取新的策略，价值函数也是增大的，即**价值函数单调递增**：
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

又因为策略的总数有限，所以 **policy iteration 的迭代次数有限**。
## Value Iteration
Value iteration maintains optimal value of starting in a state s if have a finite number of steps k left in the episode.
- Set k = 1
- Initialize $V_0(s) = 0$ for all states s 
- Loop until convergence: (for ex. $||V_{k+1} − V_k||_∞ ≤ ϵ$) 
	- For each state s, use **Bellman backup operator** on value function
$$V_{k+1}(s) = BV_k(s) =  \max_{a \in A} \left[ R(s, a) + γ \sum_{s'∈S} P(s'|s, a) V_k(s') \right]$$
$$\pi_{k + 1} = \arg\max_{a \in A} \left[ R(s, a) + \gamma \sum_{s' \in S} P(s' | s, a) V_{k+1}(s') \right]$$
对于 infinite-horizon MDP，迭代直到收敛。对于 finite-horizon MDP，迭代直到用完所有时间步。

Value iteration 相当于从最后一步开始，不断往前迭代，计算每一步的 optimal value function。这种方法称为 **bootstrap**。

可以证明，Bellman backup operator 是一个 contraction operator（[[ps4]]）。因此，value iteration 会收敛到唯一解。

注意 value iteration 与 policy iteration 的不同：
- Value iteration 可以计算 finite-horizon MDP 的价值函数，policy iteration 不能计算 finite-horizon MDP 的价值函数
- Value iteration 每一步使用的都是最优价值函数，policy iteration 每一步使用的都是当前策略的价值函数
## Bellman backup operator
上述算法中分别用到了两种 Bellman backup operator。Bellman backup operator 作用于一个价值函数，返回一个新的价值函数：
- Bellman backup operator $B$
$$BV(s) = \max_{a \in A} \left[ R(s, a) + \gamma \sum_{s' \in S} P(s' | s, a) V(s')\right]$$
- Bellman backup operator for a particular policy $B^\pi$
$$B^\pi V(s) = R(s, \pi(s)) + \gamma \sum_{s' \in S} P(s' | s, \pi(s)) V(s')$$

下面总结 Bellman backup operator 的一些性质：

|                                | $B$                 | $B^\pi$                              |
| :----------------------------- | ------------------- | ------------------------------------ |
| 计算某个策略的价值函数（policy evaluation） |                     | $V^\pi = B^\pi B^\pi \cdots B^\pi V$ |
| 计算最优价值函数（value iteration）      | $V^* = BB\cdots BV$ |                                      |
| 不动点                            | $BV^* = V^*$        | $B^\pi V^\pi = V^\pi$                |

假设 $\pi$ 是由 $V$ 导出的贪心策略，即：
$$\pi(s) = \arg\max_a [R(s, a) + \gamma \sum_{s' \in S} P(s' | s, a) V(s')]$$
则有：
$$B^\pi V = BV$$
Policy improvement 中，$\pi_{i+1}$ 是由 $V^{\pi_i}$ 导出的贪心策略，因此，$B^{\pi_{i+1}}V^{\pi_i} = BV^{\pi_i}$ 

注：Policy improvement 中，$BV^{\pi_i} \neq V^{\pi_{i+1}}$。










