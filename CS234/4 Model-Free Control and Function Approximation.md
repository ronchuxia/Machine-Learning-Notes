# Generalized Policy Improvement
在 policy improvement 中，需要对所有的 $s \in S$ 和 $a \in A$ 计算 $Q^\pi(s, a)$。然而，对于确定策略 $\pi$，我们无法通过采样对 $Q^\pi(s, a)$ 进行估计。因此，在 model-free 的情况下，我们采取随机策略：
$$\pi(a|s) = \cases{\arg\max_a Q(s, a) \ \text{w. prob} \ 1 - \epsilon + \frac{\epsilon}{|A|} \\ a' \neq \arg\max_a Q(s, a) \ \text{w. prob} \ \frac{\epsilon}{|A|}}$$
即以 $1 - \epsilon$ 的概率采取最优动作，以 $\epsilon$ 的概率采取随机动作。这种策略称为 $\epsilon$-greedy policy。使用 $\epsilon$-greedy policy 允许我们探索不同动作的奖励。

可以证明，对于任意的 $\epsilon$-greedy policy $\pi_i$，由 $Q^{\pi_i}$ 导出的 $\epsilon$-greedy policy $\pi_{i + 1}$，其价值函数是单调递增的，即 $V^{\pi_{i + 1}} \geq V^{\pi_i}$。

控制方法有两种：
- **on-policy** control: Learn to estimate and evaluate a policy from experience obtained from following that policy
- **off-policy** control: Learn to estimate and evaluate a policy using experience gathered from following a different policy

# Monte Carlo Control
这是一种 on-policy control 方法。
- Initialize $Q(s, a) = 0, N(s, a) = 0 \ ∀(s, a)$, Set $ϵ = 1, k = 1$, 
- $π_k = ϵ\text{-greedy}(Q)$ // Create initial ϵ-greedy policy 
- loop 
	- Sample k-th episode $(s_{k,1}, a_{k,1}, r_{k,1}, s_{k, 2},  . . . , s_{k, T})$ given $π_k$
	- $G_{k, t} = r_{k, t} + γ r_{k, t+1} + γ^2 r_{k, t+2} + · · · + γ^{T−t} r_{k, T}$ 
	- for $t = 1, . . . , T$ do
		- if First visit to $(s, a)$ in episode k then 
			- $N(s, a) = N(s, a) + 1$ 
			- $Q(s_t, a_t) = Q(s_t, a_t) + \frac{1}{N(s, a)} (G_{k, t} − Q(s_t , a_t))$
		- end if 
	- end for
	- $k = k + 1, ϵ = 1/k$ 
	- $π_k = ϵ\text{-greedy}(Q)$ // Policy improvement 
- end loop

当策略满足 **GLIE（Greedy in the Limit of Infinite Exploration）条件**：
1. Infinite Exploration（无限探索）：每一个 state-action pair $(s,a)$ 都会被尝试无限多次
$$\lim_{i \rightarrow \infty} N_i(s, a) \rightarrow \infty$$
3. Greedy in the Limit（最终贪婪）：随着时间的推移，策略变得越来越贪婪
$$\lim_{i \rightarrow \infty} \pi_i(s) \rightarrow \arg\max_a Q(s, a) \ \text{with probability 1}$$
时，GLIE Monte-Carlo Control using Tabular Representations 会收敛到 optimal state-action value function $Q^*(s, a)$。

注：满足 $\epsilon_i = 1 / i$ 的 $\epsilon$-greedy policy 是一个满足 GLIE 条件的策略。

# Temporal Difference Control
## SARSA
这是一种 on-policy control 方法。
- Set initial ϵ-greedy policy π, t = 0, initial state $s_t = s_0$ 
- Take $a_t ∼ π(s_t)$ // Sample action from policy
- Observe $(r_t, s_{t+1})$ 
- loop
	- Take action $a_{t+1} ∼ π(s_{t+1})$
	- Observe $(r_{t+1}, s_{t+2})$ 
	- $Q(s_t, a_t) ← Q(s_t, a_t) + α (r_t + γ Q(s_{t+1}, a_{t+1}) − Q(s_t , a_t))$ 
	- $π(s_t) = \arg\max_a Q(s_t, a)$ w.prob 1 − ϵ, else random 
	- $t = t + 1$ 
- end loop

当：
- 策略满足 GLIE 条件
- $\alpha$ 满足 Robbins-Munro sequence
时，SARSA 会收敛到 optimal state-action value function $Q^*(s, a)$。 
## Q-Learning
这是一种 off-policy control 方法。
- Set initial ϵ-greedy policy π, t = 0, initial state $s_t = s_0$ 
- Take $a_t ∼ π(s_t)$ // Sample action from policy
- Observe $(r_t, s_{t+1})$ 
- loop
	- Take action $a_{t+1} ∼ π(s_{t+1})$
	- Observe $(r_{t+1}, s_{t+2})$ 
	- $Q(s_t, a_t) ← Q(s_t, a_t) + α (r_t + γ \max_a Q(s_{t+1}, a) − Q(s_t , a_t))$ 
	- $π(s_t) = \arg\max_a Q(s_t, a)$ w.prob 1 − ϵ, else random 
	- $t = t + 1$ 
- end loop

当：
- 每一个 state-action pair $(s,a)$ 都会被尝试无限多次
- $\alpha$ 满足 Robbins-Munro sequence
时，Q-Learning 会收敛到 optimal state-action value function $Q^*(s, a)$。 

当：
- 策略满足 GLIE 条件
- $\alpha$ 满足 Robbins-Munro sequence
时，Q-Learning 会收敛到 optimal policy $\pi^*(s, a)$。

Q-Learning 与 SARSA 的区别：
- SARSA 在更新 Q 值时采取当前策略下的实际动作，行为策略与目标策略一致，这使得 SARSA 更加稳定，但收敛速度可能较慢
- Q-Learning 在更新 Q 值时采取当前 Q 值下的最优动作，行为策略与目标策略不一致，这使得 Q-Learning 较不稳定（对于某些动作过于乐观），但收敛速度可能较快 

# Value Function Approximation
当状态或动作空间很大时，我们无法对每个 state-action pair 的价值函数进行准确的估计。因此，需要使用 value function approximation 近似价值函数。Value function approximation 最小化 MSE。
## Monte Carlo Value Function Approximation
在 Monte Carlo Policy Evaluation 中，$G_t$ 是对 $V(s)$ 的估计。因此，MC VFA 可以被归约为在一组 state-return pair $(s_t, G_t)$ 上进行监督学习。
## Temporal Difference Value Function Approximation
在 Temporal Difference Policy Evaluation 中，$r_t + \gamma \hat V(s'; w)$ 是对 $V(s)$ 的估计。因此 TD VFA 可以被归约为在一组 data pair $(s_t, r_t + \gamma \hat V(s_{t+1}; w))$ 上进行监督学习。

# Control Using Value Function Approximation 
## Control Using General Value Function Approximators
Interleave
- Approximate policy evaluation using value function approximation
	- Monte Carlo
$$∆w = α(G_t − \hat Q(s_t, a_t; w)) ∇_w \hat Q(s_t, a_t; w)$$
	- SARSA
$$∆w = α(r_t + \gamma \hat Q(s_{t + 1}, a_{t + 1}; w) − \hat Q(s_t, a_t; w)) ∇_w \hat Q(s_t, a_t; w)$$
	- Q-Learning
$$∆w = α(r_t + \gamma \max_a Q(s_{t + 1}, a; w) − \hat Q(s_t, a_t; w)) ∇_w \hat Q(s_t, a_t; w)$$
- Perform ϵ-greedy policy improvement

Control using value function approximation 交替进行（近似的）Bellman backup 和 value function fitting。其中 Bellman backup 是 contraction operator，而 value function fitting 则有可能是 expasion operator。因此 control using value function approximation 有可能会导致震荡，不一定收敛。
## Deep Q-Learning
直接将神经网络用于拟合价值函数的 Q-Learning 存在两个问题：
- Experience tuple $(s, a, r, s')$ 是按生成顺序处理的，彼此之间存在较强的相关性，这导致神经网络的学习不稳定
- Q-target $r_t + \gamma \max_a Q(s_{t+1}, a; w)$ 是不断变化的，这导致神经网络的学习不稳定

DQN 针对这两个问题，对 Q-Learning 进行了改进：
- **Experience replay**：DQN 将所有的 experience tuple 保存在一个 replay buffer 中，每次从 replay buffer 中采样一个 experience tuple 用于参数更新
- **Fixed Q-targets**：DQN 使用另一组参数 $w^-$ 计算 Q-target，$w^-$ 在多次迭代间保持不变，仅在每隔一定次数的迭代后更新一次

DQN 算法：[[A2]]



