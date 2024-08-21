# Off Policy Evaluation And Causal Inference
有时我们需要在不执行策略的情况下评价策略的表现（比如医疗），这种任务称为 off-policy evaluation 或 causal inference。

假设我们采取随机策略 $\pi(s, a) = p(a | s, \pi)$。$\pi_0$ 为基线策略，由观测数据算得。$\pi_1$ 为目标策略，由 MDP 习得。我们已知在分布 $\pi_0$ 下采样的多条 MDP 所得的奖励（即观测数据），并想要据此预测在分布 $\pi_1$ 下采样的 MDP 所得的期望奖励。

1. Regression estimator：
用 MDP 估计的奖励函数估计期望奖励：
$$E_{s \sim p(s), a \sim \pi_1(s, a)} \left[ \hat R(s, a) \right]$$

2. Importance sampling：
重要性采样是一种统计方法，用于估计随机变量的期望值或其他统计量，特别是当我们难以直接从目标分布中采样时。它通过从一个已知的分布（称为提议分布或重要性分布）中采样来估计另一个目标分布下的期望值。在这里，我们难以从目标策略 $\pi_1$ 中采样，因此转而从基线策略 $\pi_0$ 中采样，用于估计 $\pi_1$ 下的期望奖励：
$$E_{s \sim p(s), a \sim \pi_0(s, a)} \left[ \frac{\pi_1(s, a)}{\hat \pi_0(s, a)} R(s, a) \right]$$
证明当 $\hat \pi_0 = \pi_0$ 时，上式等于期望奖励：
$$\begin{align}
E_{s \sim p(s), a \sim \pi_0(s, a)} \left[ \frac{\pi_1(s, a)}{\pi_0(s, a)} R(s, a) \right] & = \sum_{(s, a)} \frac{\pi_1(s, a)}{\pi_0(s, a)} R(s, a) p(s) \pi_0(s, a)\\
& = \sum_{(s, a)} R(s, a) p(s) \pi_1(s, a)\\
& = E_{s \sim p(s), a \sim \pi_1(s, a)}[R(s, a)]
\end{align}$$

3. Weighted importance sampling：
$$\frac{E_{s \sim p(s), a \sim \pi_0(s, a)} \left[ \frac{\pi_1(s, a)}{\hat \pi_0(s, a)} R(s, a) \right]}{E_{s \sim p(s), a \sim \pi_0(s, a)} \left[ \frac{\pi_1(s, a)}{\hat \pi_0(s, a)} \right]}$$
证明当 $\hat \pi_0 = \pi_0$ 时，上式等于期望奖励：
$$\begin{align}
\frac{E_{s \sim p(s), a \sim \pi_0(s, a)} \left[ \frac{\pi_1(s, a)}{\pi_0(s, a)} R(s, a) \right]}{E_{s \sim p(s), a \sim \pi_0(s, a)} \left[ \frac{\pi_1(s, a)}{\pi_0(s, a)} \right]} & = \frac{E_{s \sim p(s), a \sim \pi_1(s, a)}[R(s, a)]}{\sum_{(s, a)} 1 \cdot p(s) \pi_1(s, a)}\\
& = E_{s \sim p(s), a \sim \pi_1(s, a)}[R(s, a)]
\end{align}$$
在观测数据有限时，我们只能使用观测数据的加权和代替期望，此时，weighted importance sampling 的估计值是有偏差的。

4. Doubly robust：
$$E_{s \sim p(s), a \sim \pi_0(s, a)} \left[ E_{a \sim \pi_1(s, a)} \left[ \hat R (s, a) \right] + \frac{\pi_1(s, a)}{\hat \pi_0(s, a)} \left( R(s, a) - \hat R(s, a) \right) \right]$$
证明当 $\hat \pi_0 = \pi_0$ 时，上式等于期望奖励：
$$\begin{align}
& E_{s \sim p(s), a \sim \pi_0(s, a)} \left[ E_{a \sim \pi_1(s, a)} \left[ \hat R(s, a) \right] + \frac{\pi_1(s, a)}{\pi_0(s, a)} \left( R(s, a) - \hat R(s, a) \right) \right]\\ 
= & E_{s \sim p(s), a \sim \pi_1(s, a)} \left[ \hat R(s, a) \right] + \sum_{(s, a)} \left( R(s, a) - \hat R(s, a) \right) p(s) \pi_1(s, a)\\
= & E_{s \sim p(s), a \sim \pi_1(s, a)} \left[ \hat R(s, a) \right] + E_{s \sim p(s), a \sim \pi_1(s, a)} \left[ R(s, a) - \hat R(s, a) \right]\\
= & E_{s \sim p(s), a \sim \pi_1(s, a)} [R(s, a)]
\end{align}$$
证明当 $\hat R(s, a) = R(s, a)$ 时，上式等于期望奖励：
$$E_{s \sim p(s), a \sim \pi_0(s, a)} \left[ E_{a \sim \pi_1(s, a)} \left[ \hat R (s, a) \right] \right] = E_{s \sim p(s), a \sim \pi_1(s, a)} [R(s, a)]$$

5. (1) Importance sampling，适用于估计 $\pi_0$ 容易，估计 $R$ 复杂的情况。(2) Regression estimator，适用于估计 $\pi_0$ 复杂，估计 $R$ 容易的情况。

# PCA
证明最小化投影数据和原始数据的均方误差的单位向量 $u$ 就是原始数据的第一主成分。

首先，$x^{(i)}$ 在单位向量 $u$ 方向上的投影 $f_u(x^{(i)})$ 为：
$$f_u(x^{(i)}) = \frac{u^T x^{(i)}}{u^T u} \cdot u = u^T x^{(i)} u$$

其次，最小化投影数据和原始数据的均方误差的单位向量 $u$ 为：
$$\begin{align}
& \arg\min_u \sum_{i = 1}^m \left( x^{(i)} - u^T x^{(i)} u \right)^T \left( x^{(i)} - u^T x^{(i)} u \right)\\
= & \arg\min_u \sum_{i = 1}^m \left( x^{(i)} \right)^T x^{(i)} - 2 u^T \left( x^{(i)} \right)^T u x^{(i)} + u^T \left( x^{(i)} \right)^T u u^T x^{(i)} u\\
= & \arg\min_u \sum_{i = 1}^m - 2 u^T \left( x^{(i)} \right)^T u x^{(i)} + u^T \left( x^{(i)} \right)^T u u^T x^{(i)} u\\
= & \arg\min_u \sum_{i = 1}^m - 2 \left( x^{(i)} \right)^T u u^T x^{(i)} + \left( x^{(i)} \right)^T u u^T x^{(i)} u^T u\\
= & \arg\min_u \sum_{i = 1}^m - \left( x^{(i)} \right)^T u u^T x^{(i)}\\
= & \arg\min_u \sum_{i = 1}^m - u^T x^{(i)} \left( x^{(i)} \right)^T u\\
= & \arg\max_u u^T \Sigma u
\end{align}$$

# ICA
1. Unmixing matrix $W$ 的对数似然函数为：
$$l(W) = \sum_{i = 1}^m \left( \sum_{j = 1}^d \log p \left( w_j^T x^{(i)} \right) + \log |W| \right)$$
证明对于服从标准高斯分布的源数据 $s_j \sim \mathcal{N}(0, 1)$，$W$ 的闭形解存在歧义性：
$$\begin{align}
\nabla_W l(W) & = \nabla_W \left( \text{sum} \left( \log \frac{1}{\sqrt{2\pi}} \exp \left( - \frac{1}{2} \left( XW^T \right)^2 \right) \right) + m \log |W| \right)\\
& = \nabla_W \left( \text{sum} \left( - \frac{1}{2} \left( XW^T \right)^2 \right) + m \log |W| \right)\\
& = - \left( XW^T \right)^T X + m \left( W^{-1} \right)^T\\
& = - W X^T X + m \left( W^{-1} \right)^T\\
& = 0
\end{align}$$
所以：
$$W^TW = m (X^T X)^{-1}$$
对于任意的正交矩阵 $R$，$W' = RW$：
$${W'}^T W' = W^T W$$
即 $W$ 和 $W'$ 都是可行解。

2. 推导对于服从标准拉普拉斯分布的源数据 $s_j \sim \mathcal{L}(0, 1)$，随机梯度上升的更新规则：
$$\begin{align}
\nabla_W l(W) & = \nabla_W \left( \text{sum} \left( \log \frac{1}{2} \exp \left( - |Wx^{(i)}| \right) \right) + \log |W| \right)\\
& = \nabla_W \left( \text{sum} \left( - |Wx^{(i)}| \right) + \log |W| \right)\\
& = - \text{sign} \left( Wx^{(i)} \right) \left( x^{(i)} \right)^T + \left( W^{-1} \right)^T
\end{align}$$
随机梯度上升的更新规则为：
$$W := W + \alpha \left( - \text{sign} \left( Wx^{(i)} \right) \left( x^{(i)} \right)^T + \left( W^{-1} \right)^T \right)$$

# Markov Decision Processes
考虑一个有限状态、有限动作、且折扣因子 $\gamma < 1$ 的 MDP。令 $B$ 为 Bellman update operator，即如果 $V' = B(V)$，那么：
$$V'(s) = R(s) + \gamma \max_{a \in A} \sum_{s' \in S} P_{sa}(s') V(s')$$

1. 证明此时对于任意两个有界的 $V_1$ 和 $V_2$：
$$||B(V_1) - B(V_2)||_\infty \leq \gamma ||V_1 - V_2||_\infty$$
其中：
$$||V||_\infty = \max_{s \in S} |V(s)|$$

证明：
$$\begin{align}
||B(V_1) - B(V_2)||_\infty & = \gamma \left| \left| \max_{a \in A} \left[R(s, a) + \sum_{s' \in S} P_{sa}(s') V_1(s')\right] - \max_{a' \in A} \left[ R(s, a') + \sum_{s' \in S}P_{sa'}(s') V_2(s') \right] \right| \right|_\infty\\
& \leq \gamma \max_{a \in A} \left|\left| \left[R(s, a) + \sum_{s' \in S} P_{sa}(s') V_1(s')\right] - \left[ R(s, a) + \sum_{s' \in S}P_{sa}(s') V_2(s') \right] \right|\right|_\infty\\
& = \gamma \max_{a \in A} \left| \left| \sum_{s' \in S} P_{sa}(s')[V_1(s') - V_2(s')] \right| \right|_\infty\\
& \leq \gamma \max_{a \in A} \left| \left| \sum_{s' \in S} P_{sa}(s') |V_1(s') - V_2(s')| \right| \right|_\infty\\
& = \gamma ||V_1(s) - V_2(s)||_\infty\\
\end{align}$$

这表明 Bellman update operator 是一个 $\gamma$-contraction in the max-norm。

注：可以证明：
$$|\max_{a \in A} Q(s, a) - \max_{a' \in A}Q'(s, a')| \leq \max_{a \in A}|Q(s, a) - Q'(s, a)|$$

2. 证明此时 Bellman update operator 只有一个不动点。

假设 $V_1$ 和 $V_2$ 都是 Bellman update operator 的不动点，则：
$$||B(V_1) - B(V_2)||_\infty = ||V_1 - V_2||_\infty \leq \gamma ||V_1 - V_2||_\infty$$
因为 $\gamma < 1$，所以要让上式成立，必有 $V_1 = V_2$。

这表明 value iteration 几何收敛到唯一的最优价值函数。
