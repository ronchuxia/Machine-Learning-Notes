对于 POMDP，无法根据当前观测确定当前状态，如果采取确定策略，价值函数不一定最优。因此，可以使用随机策略。Policy based RL 可以学习随机策略，通过优化随机策略，最大化价值函数。

# Policy Gradient
[[19 Policy Search]]
对于 episodic MDP，可以使用 Monte Carlo 方法估计价值函数的梯度：
$$\begin{align}
\nabla_\theta V(\theta) & = \nabla_\theta E_\tau [R(\tau)]\\
& = E_\tau \left[ R(\tau) \sum_{t = 0}^{T-1} \nabla_\theta \log \pi(a_t | s_t; \theta) \right]\\
& \approx \frac{1}{m} \sum_{i = 1}^m R(\tau^{(i)}) \sum_{t = 0}^{T-1} \nabla_\theta \log \pi(a_t | s_t; \theta) 
\end{align}$$

对于 non-episodic MDP，可以一般化为**策略梯度定理**：
对于任意的可微分策略 $\pi_\theta(s, a)$ 和策略目标函数 $J = J_1$（episodic reward），$J_{avR}$（average reward per time step），或 $\frac{1}{1 - \gamma} J_{avR}$（average value），策略梯度为：
$$\nabla_\theta J(\theta) = E_{\pi_\theta}[\nabla_\theta \log \pi(a | s; \theta) Q^{\pi_\theta}(s, a)]$$

# Policy Gradient with Temporal Structure
使用 Monte Carlo 方法估计的梯度是无偏的，但是方差较大。利用马尔可夫假设，可以减小梯度估计量的方差：
$$\begin{align}
\nabla_\theta V(\theta) & = \nabla_\theta E_\tau [R(\tau)]\\
& = \nabla_\theta E_\tau \left[ \sum_{t' = 0}^{T - 1} r_{t'} \right]\\
& = \sum_{t' = 0}^{T - 1} \nabla_\theta  E_\tau [r_{t'}] \\
& = \sum_{t' = 0}^{T-1} \nabla_\theta E_{s_0, a_0, s_1, a_1, \dots, s_{t'}, a_{t'}}[r_{t'}]\\
& = \sum_{t' = 0}^{T-1} E_{s_0, a_0, s_1, a_1, \dots, s_{t'}, a_{t'}} \left[ r_{t'} \sum_{t = 0}^{t'} \nabla_\theta \log \pi(a_t | s_t; \theta) \right]\\
& = \sum_{t' = 0}^{T-1} E_\tau \left[ r_{t'} \sum_{t = 0}^{t'} \nabla_\theta \log \pi(a_t | s_t; \theta) \right]\\
& = E_\tau \left[ \sum_{t' = 0}^{T - 1} r_{t'} \sum_{t = 0}^{t'} \nabla_\theta \log \pi(a_t | s_t; \theta) \right]\\
& = E_\tau \left[ \sum_{t = 0}^{T - 1} \nabla_\theta \log \pi(a_t | s_t; \theta) \sum_{t' = t}^{T - 1} r_{t'} \right]\\
& = E_\tau \left[ \sum_{t = 0}^{T - 1} \nabla_\theta \log \pi(a_t | s_t; \theta) G_t \right]\\
& \approx \frac{1}{m} \sum_{i = 1}^m \sum_{t = 0}^{T - 1} \nabla_\theta \log \pi( a_t^{(i)} | s_t^{(i)}; \theta) G_t^{(i)}
\end{align}$$

intuition: moving in the direction of the policy gradient pushes up the logprob of the action $a_t$, in proportion to how good $G_t$ is.

据此，得到 **REINFORCE 算法**：
- Initialize policy parameters $\theta$ arbitrarily
- for each episode $\{s_1, a_1, r_2, ···, s_{T-1}, a_{T-1}, r_T\} \sim \pi_\theta$ do
	- for t = 1 to T - 1 do
		- $\theta \leftarrow \theta + \alpha \nabla_\theta \log \pi(a_t | s_t; \theta) G_t$
	- endfor
- endfor
- return $\theta$

# Policy Gradient with Baseline
使用 REINFORCE 算法估计的梯度是无偏的，但是方差较大。通过引入 baseline，可以减小梯度估计量的方差：
$$\nabla_\theta V(\theta) = E_\tau \left[ \sum_{t = 0}^{T - 1} \nabla_\theta \log \pi(a_t | s_t; \theta) \left( \sum_{t' = t}^{T - 1} r_{t'} - b(s_t) \right) \right]$$

对于任意的 $b$，如果它仅是状态的函数，那么有：
$$\begin{align}
& E_\tau[\nabla_\theta \log \pi(a_t | s_t; \theta) b(s_t)]\\
= & E_{s_{0:t}, a_{0:(t-1)}} \left[ E_{s_{(t+1):T}, a_{t:(T-1)}} \left[ \nabla_\theta \log \pi(a_t | s_t; \theta) b(s_t) \right] \right]\\
= & E_{s_{0:t}, a_{0:(t-1)}} \left[ b(s_t) E_{s_{(t+1):T}, a_{t:(T-1)}} \left[ \nabla_\theta \log \pi(a_t | s_t; \theta) \right] \right]\\
= & E_{s_{0:t}, a_{0:(t-1)}} \left[ b(s_t) E_{a_t} \left[ \nabla_\theta \log \pi(a_t | s_t; \theta) \right] \right]\\
= & E_{s_{0:t}, a_{0:(t-1)}} \left[ b(s_t) \sum_{a_t} \pi(a_t | s_t; \theta) \frac{\nabla_\theta \pi(a_t | s_t; \theta)}{\pi(a_t | s_t; \theta)} \right]\\
= & E_{s_{0:t}, a_{0:(t-1)}} \left[ b(s_t) \sum_{a_t} \nabla_\theta \pi(a_t | s_t; \theta) \right]\\
= & E_{s_{0:t}, a_{0:(t-1)}} \left[ b(s_t) \nabla_\theta \sum_{a_t} \pi(a_t | s_t; \theta) \right]\\
= & E_{s_{0:t}, a_{0:(t-1)}} \left[ b(s_t) \nabla_\theta 1 \right]\\
= & 0
\end{align}$$
因此：
$$\begin{align}
\nabla_\theta V(\theta) & = \nabla_\theta E_\tau[R(\tau)]\\
& = E_\tau \left[ \sum_{t = 0}^{T - 1} \nabla_\theta \log \pi(a_t | s_t; \theta) \sum_{t' = t}^{T - 1} r_{t'} \right]\\
& = E_\tau \left[ \sum_{t = 0}^{T - 1} \nabla_\theta \log \pi(a_t | s_t; \theta) \left( \sum_{t' = t}^{T - 1} r_{t'} - b(s_t) \right) \right] 
\end{align}$$
即使用 baseline 估计的梯度是无偏的。

通常使用 expected return 作为 baseline：
$$b(s_t) \approx E[r_t + r_{t+1} + \cdots + r_{T-1}]$$
可以使用神经网络拟合 expected return。

注：并不是所有的 baseline 都会减小方差，比如常数。要减小方差，应选择与 expected return 相关性较高的 baseline。

Intuition: moving in the direction of the policy gradient with baseline pushes up the logprob of the action $a_t$, in proportion to how $G_t$ is better than expected.

据此，得到 **Vanilla Policy Gradient 算法**：
- Initialize policy parameter $\theta$, baseline $b$
- for iteration = 1, 2, ··· do 
	- Collect a set of trajectories by executing the current policy 
	- At each timestep $t$ in each trajectory $\tau^{(i)}$ , compute 
		- Return $G^{(i)}_t = \sum_{t' = t}^{T - 1} r_{t'}^{(i)}$, and
		- Advantage estimate $\hat A_t^{(i)} = G_t^{(i)} - b(s_t^{(i)})$
	- Re-fit the baseline, by minimizing $\sum_i \sum_t |b(s_t^{(i)}) - G_t^{(i)}|^2$
	- Update the policy, using a policy gradient estimate $\hat g = \nabla_\theta \log \pi(a_t | s_t; \theta) \hat A_t$
- endfor

# Actor-Critic Methods
在 policy gradient 中，我们使用 Monte Carlo 方法，使用 $G_t$ 估计价值函数。为了进一步减小方差，可以使用 actor-critic 方法。

Actor-critic 方法同时维护一个显式的策略和一个显式的价值函数，并同时更新两者。例如，可以使用 Temporal Difference 方法或 Function Approximation 方法估计价值函数。其中，显式的价值函数称为 critic。

使用 Temporal Difference 方法：
$$\nabla_\theta V(\theta) \approx E_\tau \left[ \sum_{t = 0}^{T - 1} \nabla_\theta \log \pi(a_t | s_t; \theta) (r_t + \gamma V(s_{t+1}) - b(s_t)) \right]$$

使用 Function Approximation 方法：
$$\nabla_\theta V(\theta) \approx E_\tau \left[ \sum_{t = 0}^{T - 1} \nabla_\theta \log \pi(a_t | s_t; \theta) (Q(s_t, a_t; w) - b(s_t)) \right]$$














