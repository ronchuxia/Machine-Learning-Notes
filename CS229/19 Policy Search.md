无需计算最优价值函数，直接拟合策略。

Policy search 输出 **stochastic policy** $\pi: S \times A \mapsto R$，$\pi(s, a)$ 表示在状态 $s$ 下采取动作 $a$ 的概率。

Policy search 最大化期望奖励：
$$\max_\theta E_{s_0, a_0, s_1, a_1, \cdots, s_T, a_T}[R(s_0, a_0) + R(s_1, a_1) + \cdots + R(s_T, a_T) | \pi_\theta]$$

Notations：
- $\tau = s_0, a_0, s_1, a_1, \cdots, s_T, a_T$
- $R(\tau) = R(s_0, a_0) + R(s_1, a_1) + \cdots + R(s_T, a_T)$

Policy search 使用 **REINFORCE 算法**最大化价值函数：
Repeat {
	Sample $\tau = s_0, a_0, s_1, a_1, \cdots, s_T, a_T$.
	Compute $R(\tau) = R(s_0, a_0) + R(s_1, a_1) + \cdots + R(s_T, a_T)$.
	Update $\theta := \theta + \alpha \left[ \frac{\nabla_\theta \pi_\theta(s_0, a_0)}{\pi_\theta(s_0, a_0)} + \frac{\nabla_\theta \pi_\theta(s_1, a_1)}{\pi_\theta(s_1, a_1)} + \cdots \frac{\nabla_\theta \pi_\theta(s_T, a_T)}{\pi_\theta(s_T, a_T)} \right] R(\tau)$.
}

期望奖励的梯度：
$$\begin{align}
\nabla_\theta E_\tau[R(\tau)] & = \nabla_\theta \sum_\tau P_\theta(\tau) R(\tau)\\
& = \nabla_\theta \sum_\tau P(s_0) \pi_\theta(s_0, a_0) P_{s_0a_0}(s_1) \pi_\theta(s_1, a_1) \cdots P_{s_{T - 1}a_{T - 1}}(s_T) \pi_\theta(s_T, a_T) R(\tau)\\
& = \sum_\tau [P(s_0) \nabla_\theta(\pi_\theta(s_0, a_0)) P_{s_0a_0}(s_1) \pi_\theta(s_1, a_1) \cdots P_{s_{T - 1}a_{T - 1}}(s_T) \pi_\theta(s_T, a_T) + P(s_0) \pi_\theta(s_0, a_0) P_{s_0a_0}(s_1) \nabla_\theta(\pi_\theta(s_1, a_1)) \cdots P_{s_{T - 1}a_{T - 1}}(s_T) \pi_\theta(s_T, a_T) + \cdots + P(s_0) \pi_\theta(s_0, a_0) P_{s_0a_0}(s_1) \pi_\theta(s_1, a_1) \cdots P_{s_{T - 1}a_{T - 1}}(s_T) \nabla_\theta(\pi_\theta(s_T, a_T))] R(\tau)\\
& = \sum_\tau P(s_0) \pi_\theta(s_0, a_0) P_{s_0a_0}(s_1) \pi_\theta(s_1, a_1) \cdots P_{s_{T - 1}a_{T - 1}}(s_T) \pi_\theta(s_T, a_T) \left[ \frac{\nabla_\theta \pi_\theta(s_0, a_0)}{\pi_\theta(s_0, a_0)} + \frac{\nabla_\theta \pi_\theta(s_1, a_1)}{\pi_\theta(s_1, a_1)} + \cdots + \frac{\nabla_\theta \pi_\theta(s_T, a_T)}{\pi_\theta(s_T, a_T)} \right] R(\tau)\\
& = \sum_\tau P_\theta(\tau) \left[ \frac{\nabla_\theta \pi_\theta(s_0, a_0)}{\pi_\theta(s_0, a_0)} + \frac{\nabla_\theta \pi_\theta(s_1, a_1)}{\pi_\theta(s_1, a_1)} + \cdots + \frac{\nabla_\theta \pi_\theta(s_T, a_T)}{\pi_\theta(s_T, a_T)} \right] R(\tau)\\
& = E_\tau \left[ \left[ \frac{\nabla_\theta \pi_\theta(s_0, a_0)}{\pi_\theta(s_0, a_0)} + \frac{\nabla_\theta \pi_\theta(s_1, a_1)}{\pi_\theta(s_1, a_1)} + \cdots + \frac{\nabla_\theta \pi_\theta(s_T, a_T)}{\pi_\theta(s_T, a_T)} \right] R(\tau) \right]\\
\end{align}$$
因此，REINFORCE 算法中参数的更新量的期望等于 expected payoff 的梯度，即 **REINFORCE 算法是一种随机梯度上升算法**。

优点：适用于 POMDP。

缺点：收敛较慢。对于需要长期规划的 MDP，效果一般不如 value iteration。