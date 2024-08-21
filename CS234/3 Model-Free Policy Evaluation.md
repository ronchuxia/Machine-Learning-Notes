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
对于 episodic MDP（有终点的 MDP，但时间步数量不固定），可以通过采样，用 Monte Carlo 方法估计价值函数。
## First-Visit Monte Carlo (MC) On Policy Evaluation
Initialize $N(s) = 0, G(s)=0 \ \forall s \in S$
Loop 
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
Initialize $N(s) = 0, G(s)=0 \ \forall s \in S$
Loop 
- Sample episode i $= s_{i,1}, a_{i,1}, r_{i,1}, s_{i,2}, a_{i,2}, r_{i,2}, ..., s_{i,T_i} , a_{i,T_i}, r_{i,T_i}$ 
- Define $G_{i,t} = r_{i,t} + \gamma r_{i, t+1} + \gamma^2 r_{i, t+2} + \cdots + \gamma^{T_i - t} r_{i, T_i}$ as return from time step t onwards in ith episode
- For each time step t until $T_i$ ( the end of the episode i) 
	- If state s is the state visited at time step t in episodes i 
		- Increment counter of total visits: $N(s) = N(s)+1$ 
		- Increment total return $G(s) = G(s) + G_{i,t}$ 
		- Update estimate $V^\pi(s) = G(s)/N(s)$
性质：
- 有偏估计量（因为多次 visit 中相同的 state 是相关的，不再 iid），但方差比 first-visit MC 小
- 一致估计量
## Incremental Monte Carlo (MC) On Policy Evaluation
Initialize $N(s) = 0, V^\pi(s)=0 \ \forall s \in S$
Loop 
- Sample episode i $= s_{i,1}, a_{i,1}, r_{i,1}, s_{i,2}, a_{i,2}, r_{i,2}, ..., s_{i,T_i} , a_{i,T_i}, r_{i,T_i}$ 
- Define $G_{i,t} = r_{i,t} + \gamma r_{i, t+1} + \gamma^2 r_{i, t+2} + \cdots + \gamma^{T_i - t} r_{i, T_i}$ as return from time step t onwards in ith episode
- For each time step t until $T_i$ ( the end of the episode i) 
	- If state s is the state visited at time step t in episodes i 
		- Increment counter of total visits: $N(s) = N(s)+1$ 
		- Update estimate $V^\pi(s) = \frac{1}{N(s)} [V^\pi(s) (N(s) - 1) + G(s)]= V^\pi(s) + \frac{1}{N(s)}[G(s) - V^\pi(s)]$

可以改写成 $V^\pi(s) = V^\pi(s) + \alpha [G(s) - V^\pi(s)]$。当 $\alpha = \frac{1}{N(s)}$ 时，incremental MC 与 every-visit MC 等价。

TD
delay alpha to ensure convergence