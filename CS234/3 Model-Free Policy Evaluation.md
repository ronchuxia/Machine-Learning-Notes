# Bias, Variance and MSE
统计值 $\hat \theta$ 是参数 $\theta$ 的估计量：
- Bias
$$Bias_\theta(\hat \theta) = E_{x|\theta}[\hat \theta] - \theta$$
- Variance
$$Var(\hat\theta) = E_{x|\theta}\left[(\hat\theta - E_{x|\theta}[\hat \theta])^2\right]$$
- MSE (Mean Squared Error)
$$MSE(\hat\theta) = E_{x|\theta}\left[(\hat\theta - \theta)^2\right] = Bias_\theta(\hat\theta)^2 + Var(\hat\theta)$$

无偏估计量和一致估计量：
- 无偏估计量
$$Bias_\theta(\hat\theta) = E_{x|\theta}[\hat \theta] - \theta = 0$$
- 一致估计量
$$\lim_{n \rightarrow \infty} Pr(|\hat\theta - \theta| > \epsilon) = 0$$
无偏估计量不一定一致（比如某个估计量，随样本数量增加，其方差不变），一致估计量不一定无偏（比如未添加贝塞尔校正的样本方差）。

# Monte Carlo (MC) Policy Evaluation
对于 episodic MDP（有终点的 MDP，但时间步数量不固定），可以通过对 episodes 进行采样，用 Monte Carlo 方法估计价值函数。
## First-Visit Monte Carlo (MC) On Policy Evaluation
- Initialize $N(s) = 0, G(s)=0 \ \forall s \in S$
- Loop 
	- Sample episode i $= s_{i,1}, a_{i,1}, r_{i,1}, s_{i,2}, a_{i,2}, r_{i,2}, ..., s_{i,T_i} , a_{i,T_i}, r_{i,T_i}$ 
	- Define $G_{i,t} = r_{i,t} + \gamma r_{i, t+1} + \gamma^2 r_{i, t+2} + \cdots + \gamma^{T_i - t} r_{i, T_i}$ as return from time step t onwards in ith episode
	- For each time step t until $T_i$ ( the end of the episode i) 
		- If this is the first time t that state s is visited in episode i 
			- Increment counter of total first visits: $N(s) = N(s)+1$ 
			- Increment total return $G(s) = G(s) + G_{i,t}$ 
			- Update estimate $V^\pi(s) = G(s)/N(s)$
性质：
- 无偏估计量
- 一致估计量
## Every-Visit Monte Carlo (MC) On Policy Evaluation
- Initialize $N(s) = 0, G(s)=0 \ \forall s \in S$
- Loop 
	- Sample episode i $= s_{i,1}, a_{i,1}, r_{i,1}, s_{i,2}, a_{i,2}, r_{i,2}, ..., s_{i,T_i} , a_{i,T_i}, r_{i,T_i}$ 
	- Define $G_{i,t} = r_{i,t} + \gamma r_{i, t+1} + \gamma^2 r_{i, t+2} + \cdots + \gamma^{T_i - t} r_{i, T_i}$ as return from time step t onwards in ith episode
	- For each time step t until $T_i$ ( the end of the episode i) 
		- If state s is the state visited at time step t in episodes i 
			- Increment counter of total visits: $N(s) = N(s)+1$ 
			- Increment total return $G(s) = G(s) + G_{i,t}$ 
			- Update estimate $V^\pi(s) = G(s)/N(s)$
性质：
- 有偏估计量：因为多次 visit 中相同的 state 是相关的，不再 iid
- 一致估计量
- 方差一般比 first-visit MC 小
## Incremental Monte Carlo (MC) On Policy Evaluation
- Initialize $N(s) = 0, V^\pi(s)=0 \ \forall s \in S$
- Loop 
	- Sample episode i $= s_{i,1}, a_{i,1}, r_{i,1}, s_{i,2}, a_{i,2}, r_{i,2}, ..., s_{i,T_i} , a_{i,T_i}, r_{i,T_i}$ 
	- Define $G_{i,t} = r_{i,t} + \gamma r_{i, t+1} + \gamma^2 r_{i, t+2} + \cdots + \gamma^{T_i - t} r_{i, T_i}$ as return from time step t onwards in ith episode
	- For each time step t until $T_i$ ( the end of the episode i) 
		- If state s is the state visited at time step t in episodes i 
			- Increment counter of total visits: $N(s) = N(s)+1$ 
			- Update estimate $V^\pi(s) = \frac{1}{N(s)} [V^\pi(s) (N(s) - 1) + G(s)]= V^\pi(s) + \frac{1}{N(s)}[G(s) - V^\pi(s)]$

一般地，incremental MC 可以写成：
$$V^\pi(s) = V^\pi(s) + \alpha [G(s) - V^\pi(s)]$$
- 当 $\alpha = \frac{1}{N(s)}$ 时，incremental MC 与 every-visit MC 等价
- 当 $\alpha$ 满足一定条件时，一致估计量

注：使用 Monte Carlo 方法也可以对 $Q(s, a)$ 进行估计。

# Temporal Difference (TD) Policy Evaluation
Temporal difference learning 是另一种估计价值函数的方法：
- 它是 Monte Carlo 方法（sampling）与动态规划方法（bootstrapping）的结合
- 它可以估计 episodic 和 infinite-horizon non-episodic MDP 的价值函数

- Initialize $V^\pi(s) = 0, \ \forall s \in S$ 
- Loop 
	- Sample tuple $(s_t, a_t, r_t, s_{t+1})$ 
	- $V^\pi(s_t) = V^\pi(s_t) + \alpha [r_t + \gamma V^\pi(s_{t + 1}) - V^\pi(s_t)]$

性质：
- 有偏估计量：TD learning 相当于将 incremental MC 中的 $G_t$ 用 $r_t + \gamma V^\pi(s_{t+1})$ 代替，使用 bootstrap 估计 $G_t$。由于使用了 bootstrap，TD learning 对 $V^\pi$ 的估计会受到初始值的影响。因此，TD learning 对 $V^\pi$ 的估计是有偏的
- 当 $\alpha$ 满足 Robbins-Munro sequence 时，一致估计量
- 方差一般比 MC 小

注：Robbins-Munro sequence：
$$\sum_{t = 1}^\infty \alpha_t = \infty$$
$$\sum_{t=1}^\infty \alpha_t^2 < \infty$$

TD 与 MC 的比较：
- 与 MC 相比，TD 无需等到 episode 结束，而是每遇到一个 SARS 四元组时，就对价值函数进行一次更新。因此，TD 通常收敛更快
- 与 MC 相比，TD 利用了马尔可夫性质。因此，当过程具有马尔可夫性质时，TD 的效果通常比 MC 更好
- 与 MC 相比，TD 使用的是由 bootstrap 估计的累积奖励，而不是实际的累积奖励。因此，当更新次数相同时，TD 的数据利用率通常比 MC 低

# Certainty Equivalence with Dynamic Programming Policy Evaluation
After each $(s_i, a_i, r_i, s_{i+1})$ tuple 
- Recompute MLE MDP model for (s, a) 
$$\hat P(s' |s, a) = \frac{1}{N(s, a)} \sum_{k=1}^i 1\{s_k = s, a_k = a,s_{k+1} = s'\}$$
$$\hat r(s, a) = \frac{1}{N(s, a)} \sum_{k=1}^i 1\{s_k = s, a_k = a\}r_k$$
- Compute $V^\pi$ using MLE MDP (using any dynamic programming method)

性质：
- 无偏估计量
- 一致估计量

当我们掌握的数据足以较为精确地估计状态转移概率和奖励函数时，可以使用这种方法。

# Batch Policy Evaluation
- Incremental policy evaluation (online policy evaluation) 涉及到在与环境交互的过程中逐步更新状态价值估计
- Batch policy evaluation (offline policy evaluation) 涉及使用一个固定的、预先收集的数据集来评估一个策略

上述讨论的都是 incremental policy evaluation。对于 batch policy evaluation，可以对数据集中的 episodes 进行有放回采样，用 MC 或 TD 估计价值函数。在这种情况下：
- MC 会收敛到使 MSE 最小的价值函数
- TD 会收敛到 MLE MDP 的价值函数

# Comparison
下面对比一下各种 policy evaluation 的方法：

|                                                   | DP  | MC  | TD  |
| ------------------------------------------------- | --- | --- | --- |
| Usable when no models of current domain           |     | 是   | 是   |
| Handles continuing (non-episodic) domains         | 是   |     | 是   |
| Handles non-Markov domains                        |     | 是   |     |
| Converge to true value in limit (for tabular MDP) | 是   | 是   | 是   |
| Unbiased estimate of value                        |     | 是   |     |
注：Tabular MPD：状态空间和动作空间有限，且对于每一个 SAS 三元组，都有一个确切的概率和奖励值。


