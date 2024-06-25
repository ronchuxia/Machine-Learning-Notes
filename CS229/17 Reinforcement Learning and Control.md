# Markov Decision Processes
一个马尔可夫决策过程是一个元组 $(S, A. \{P_{sa}\}, \gamma, R)$，其中：
- $S$ 是**状态**（states）集合
- $A$ 是**动作**（actions）集合
- $P_{sa}$ 是**状态转移概率**（state transition probabilities）。对于每个状态 $s \in S$ 和每个动作 $a \in A$，$P_{sa}$ 是状态空间上的一个概率分布，描述了在状态 $s$ 采取动作 $a$ 后转移到的状态的概率分布
- $\gamma$ 是**折扣因子**（discount factor）。折扣因子用于权衡即时奖励和未来奖励
- $R: S \times A \mapsto \mathbb{R}$ 是**奖励函数**（reward function）。奖励函数描述了在状态 $s$ 采取动作 $a$ 后获得的**即时奖励**。奖励函数有时只是状态的函数，即 $R: S \mapsto \mathbb{R}$

注：马尔可夫决策过程基于**马尔可夫性质**，即 $p(s_{t + 1} | s_t, s_{t - 1}, \cdots, s_0) = p(s_{t + 1} | s_t)$。马尔可夫性质使得状态转移概率和奖励函数只依赖于当前的状态和动作，而与过去的状态和动作序列无关。

MDP 首先从状态 $s_0$ 开始，采取动作 $a_0$ 后，转移到状态 $s_1$，$s_1$ 采样自分布 $P_{s_0 a_0}$。接着从状态 $s_1$ 开始，采取动作 $a_1$ 后，转移到状态 $s_2$，$s_2$ 采样自分布 $P_{s_1 a_1}$。以此类推。

整个过程的**累积奖励**是每一步的奖励的折现和：
$$R(s_0, a_0) + \gamma R(s_1, a_1) + \gamma^2 R(s_2, a_2) + \cdots$$
如果奖励函数只是状态的函数：
$$R(s_0) + \gamma R(s_1) + \gamma^2 R(s_2) + \cdots$$
为了方便，下面都使用状态奖励函数。

一个**策略**（policy） $\pi: S \mapsto A$ 是一个将状态映射到动作的函数。执行策略 $\pi$ 表示每当我们处于状态 $s$ 时，我们都会采取动作 $a = \pi(s)$。定义策略 $\pi$ 的**价值函数**（value function）：
$$V^\pi(s) = E[R(s_0) + \gamma R(s_1) + \gamma^2 R(s_2) + \cdots | s_0 = s, \pi]$$
为从状态 $s$ 开始，执行策略 $\pi$ 所得累积奖励的期望。

可以证明，对于固定的策略 $\pi$，它的价值函数满足 **Bellman 方程**：
$$V^\pi(s) = R(s) + \gamma \sum_{s' \in S} P_{s \pi(s)}(s') V^\pi(s')$$

对于有限状态 MDP，Bellman 等式给出了 $|S|$ 个 $|S|$ 元线性方程。解这个线性方程组就能得到策略 $\pi$ 的价值函数 $V^\pi(s)$。

定义**最优价值函数**（optimal value function）：
$$V^*(s) = \max_\pi V^\pi (s)$$
为从状态 $s$ 开始，执行任意策略所能达到的价值函数的最大值。

可以证明，最优价值函数满足 **Bellman 最优性方程**：
$$V^*(s) = R(s) + \max_{\alpha \in A} \gamma \sum_{s' \in S} P_{sa}(s') V^*(s')$$

根据 Bellman 最优性方程，定义策略 $\pi^*: S \mapsto A$：
$$\pi^*(s) = \arg\max_{\alpha \in A} \sum_{s' \in S} P_{sa}(s') V^*(s')$$
则对于所有的 $s \in S$，有：
$$V^*(s) = V^{\pi^*(s)}(s) \geq V^\pi(s)$$
即 $\pi^*$ 就是最优策略。

# Value Iteration and Policy Iteration
对于有限状态、有限动作 MDP，**value iteration** 的过程如下：
1. For each state $s$, initialize $V (s) := 0.2$. 
2. Repeat until convergence { 
	For every state, update $V(s) := R(s) + \max_{a \in A} \gamma \sum_{s'} P_{sa}(s') V (s')$.
	}
Value iteration 可以看作是不断根据 Bellman 最优性方程更新价值函数。

可以证明，value iteration 中，$V$ 会收敛到 $V^*$。得到 $V^*$ 后，就可以得到 $\pi^*$。

对于有限状态、有限动作 MDP，**policy iteration** 的过程如下：
1. Initialize $\pi$ randomly. 
2. Repeat until convergence { 
	1. Let $V(s) := V^\pi(s)$. 
	2. For each state $s$, let $π(s) := \arg\max_{a \in A} \sum_{s'} P_{sa}(s') V(s')$. 
	}
Policy iteration 可以看作是不断根据当前的策略计算当前的价值函数，并根据当前的价值函数更新策略。

Value iteration 会收敛到局部最优解，但不会达到局部最优解。Policy iteration 在有限步迭代后会达到局部最优解。对于小的 MDP，policy iteration 收敛更快。对于大的 MDP，policy iteration 需要多次求解线性方程组，收敛较慢。因此，value iteration 使用得更多。

# Learning a Model for an MDP
如果状态转移概率未知，需要先进行实验，估计状态转移概率：
$$P_{sa}(s') = \frac{\text{\# times we took action a in state s and got to s'}}{\text{\# times we took action a in state s}}$$
如果分子或分母为 0，则状态转移概率可以估计为：$P_{sa}(s') = 1 / |s|$。

在进行优化时，可以通过添加一定的随机性或奖励新状态的发现来促使 MDP 探索更多可能的策略。


