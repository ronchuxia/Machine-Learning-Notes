首先，对 MDP 做一些一般化：
1. 如果奖励函数是状态和动作的函数，在预测时，假设系统处于状态 $s$，则选择动作：
$$\pi^*(s) = \arg\max_a R(s, a) + \gamma E_{s' \sim P_{sa}[V^*(s')]}$$
2. 如果状态转移概率或奖励函数随时间变化，则需要采取**动态策略**（non-stationary policy）：
$$\pi^{(t)}: S \mapsto A$$

# Finite-Horizon MDPs
一个 finite-horizon MDP 是一个元组 $(S, A. \{P^{(t)}_{sa}\}, T, R^{(t)})$。其中，$T$ 为 time horizon，表示 MDP 的长度。

Finite-horizon MDP 的累积奖励为：
$$R(s_0, a_0) + R(s_1, a_1) + \cdots + R(s_T, a_T)$$

注：因为累积奖励有界，所以不再需要 $\gamma$。

在 finite-horizon MDP 中，因为 time horizon 的限制，需要采取动态策略 $\pi$。定义策略 $\pi$ 价值函数：
$$V_t^\pi(s) = E[R^{(t)}(s_t, a_t) + R^{(t + 1)}(s_{t + 1}, a_{t + 1})\cdots + R^{(T)}(s_T, a_T) | s_t = s, \pi]$$
为在时间 $t$，从状态 $s$ 开始，执行策略 $\pi$ 所得累积奖励的期望。

定义最优价值函数：
$$V_t^*(s) = \max_\pi V_t^\pi(s)$$

根据 Bellman 最优性方程，可以用动态规划求解 finite-horizon MDP 的最优价值函数：
1. compute $V^∗_T(s) := \max_a R^{(T)}(s, a)$. 
2. for $t = T − 1, T - 2, \cdots, 0$ : 
	compute $V^∗_t := \max_a \left[ R^{(t)(s, a)} + E_{s' \sim P_{sa}}[V_{t + 1}^*(s')] \right]$

# Linear Quadratic Regulation (LQR)
LQR 是一种特殊的 finite-horizon MDP，其最佳策略

LQR 基于以下假设：
1. 线性状态转移函数：
$$s_{t + 1} = A_t s_t + B_t a_t + w_t$$
$$w_t \sim \mathcal{N}(0, \Sigma_t)$$
2. 二次奖励函数：
$$R^{(t)}(s_t, a_t) = - s_t^T U_t s_t - a_t^T W_t a_t$$
其中，$U_t$ 和 $W_t$ 是正定矩阵，即奖励函数为负。二次奖励函数偏好靠近原点的状态。

使用前述的动态规划求解 LQR，可以证明：
1. 最优策略 $a_t^*$ 是状态 $s_t$ 的线性函数
2. 最优策略 $a_t^*$ 与 $\Sigma_t$ 无关（但最优价值函数与 $\Sigma_t$ 有关）

# From Non-Linear Dynamics to LQR
很多模型都可以转化为 LQR。
## Linearization of Dynamics
对于状态与动作变化不大（$s_t \approx \bar s$，$a_t \approx \bar a$）的系统（例如 inverted pendulum），对状态转移函数使用一阶泰勒展开：
$$s_{t + 1} \approx F(\bar s, \bar a) + \nabla_s F(\bar s, \bar a) \cdot (s_t - \bar s) + \nabla_a F(\bar s, \bar a) \cdot (a_t - \bar a)$$
则：
$$s_{t + 1} \approx A s_t + B a_t + \kappa$$
再为 $s$ 添加一个维度 1，就能获得线性状态转移函数。

对奖励函数使用二阶泰勒展开：
$$\begin{align}
R(s_t, a_t) & \approx R(\bar s, \bar a) + \nabla_s R(\bar s, \bar a)(s - \bar s) + \nabla_a R(\bar s, \bar a)(a - \bar a)\\
& + \frac{1}{2}(s - \bar s)^T H_{ss} (s - \bar s) + \frac{1}{2}(s - \bar s)^T H_{sa} (a - \bar a)\\
& + \frac{1}{2}(a - \bar a)^T H_{aa} (a - \bar a)
\end{align}$$
则：
$$R_t(s_t, a_t) = - s_t^T U_t s_t - a_t^T W_t a_t + \kappa$$
再为 $s$ 添加一个维度 1，就能获得二次奖励函数。
## Differential Dynamic Programming (DDP)
对于状态和动作变化较大的系统（例如 rocker），使用差分动态规划。将 MDP 划分化为多个离散的时间步，在每个时间步内使用泰勒展开（即状态转移概率和奖励函数随时间变化）。

# Linear Quadratic Gaussian (LQG)
在现实中，有时不能观测到完整的状态，这种情况称为 **partially obeservable MDP（POMDP）**。

一个 finite-horizon POMDP 是一个元组 $(S, O, A, \{P_{sa}\}, T, R)$。其中，$O$ 为观测层，变量 $o_t$ 表示 t 时刻的观测值：
$$o_t | s_t \sim O(o | s)$$
在 t 时刻，POMDP 根据此前以及当前共 t 个观测值，维护一个 belief state（当前状态基于此前以及当前共 t 个观测值的条件概率分布 $P(s_t | o_1, o_2, \cdots, o_t)$），并根据 belief state 选择动作。

LQG 是 LQR 在 POMDP 下的推广，LQG 使用 Kalman filter 对 belief state 进行维护。

对于一般的 POMDP，belief state 难以维护，建议使用 policy gradient 直接根据观测值选择动作。





