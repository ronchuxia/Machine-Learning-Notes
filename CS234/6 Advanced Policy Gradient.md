Policy gradient 的限制：
- Policy gradient 是 **on-policy** control，策略每更新一次，就要采样新的轨迹，样本效率较低
- Policy gradient 在参数空间进行优化，而不是在策略空间进行优化，参数更新的步长难以控制
	- 如果步长太小，策略更新的速度太慢
	- 如果步长太大，新策略与旧策略的差距太大，导致 performance collapse（已经学到的稳定策略被抛弃，智能体需要重新探索有效策略）

我们希望：
- 在进行策略更新时，更高效地使用在当前策略下采样的轨迹
- 在进行策略更新时，新策略与旧策略的差距不能太大

# Policy Performance Bounds
可以证明（[[A2]]）：
$$J(\pi') - J(\pi) = \frac{1}{1 - \gamma} E_{s' \sim d^{\pi'}, \ a \sim \pi'}[A^\pi(s, a)]$$
其中：
$$d^\pi(s) = (1 - \gamma) \sum_{t = 0}^\infty \gamma^t P(s_t = s | \pi)$$
$$A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)$$

上式表明了 $J(\pi')$ 与 $J(\pi)$ 之间的关系。但是，为了计算等式的右边，我们需要在新策略 $\pi'$ 下采样轨迹。在进行策略更新前，我们无法做到这一点。

可以进一步将上式改写：
$$\begin{align}
J(\pi') - J(\pi) & = \frac{1}{1 - \gamma} E_{s' \sim d^{\pi'}, \ a \sim \pi'}[A^\pi(s, a)]\\ 
& = \frac{1}{1 - \gamma} E_{s \sim d^{\pi'}} \left[ \sum_a \pi'(a | s) A^\pi(s, a) \right]\\
& = \frac{1}{1 - \gamma} E_{s \sim d^{\pi'}} \left[ \sum_a \pi(a | s) \frac{\pi'(a | s)}{\pi(a | s)} A^\pi(s, a) \right]\\
& = \frac{1}{1 - \gamma} E_{s \sim d^{\pi'}, \ a \sim \pi} \left[ \frac{\pi'(a | s)}{\pi(a | s)} A^\pi(s, a) \right] 
\end{align}$$
其中，最后一步称为 **importance sampling**。

通过 importance sampling，我们用 $a \sim \pi$ 代替了期望中的 $a \sim \pi'$，但期望中仍含有 $s \sim d^{\pi'}$。研究表明，**relative policy performance bounds**:
$$|J(\pi') - (J(\pi) + \mathcal{L}_\pi(\pi'))| \leq C \sqrt{E_{s \sim d^\pi}[D_{KL}(\pi' || \pi)[s]]}$$
其中，$C$ 是常数，且：
$$\begin{align}
\mathcal{L}_\pi(\pi') & = \frac{1}{1 - \gamma} E_{s \sim d^\pi, \ a \sim \pi} \left[ \frac{\pi'(a | s)}{\pi(a | s)} A^\pi(s, a) \right]\\
& = E_{\tau \sim \pi} \left[ \sum_{t = 0}^\infty \gamma^t \frac{\pi'(a_t | s_t)}{\pi(a_t | s_t)} A^\pi(s_t, a_t) \right]
\end{align}$$

上式表明，当新策略与旧策略的 KL 散度较小时，可以确保新策略的性能比旧策略有所提升。并且，我们可以使用在旧策略下采样的轨迹进行多次策略更新。

# Proximal Policy Optimization
## Adaptive KL Penalty
为了控制新策略与旧策略的 KL 散度，一种方法是对 KL 散度进行正则惩罚：
$$\theta_{k + 1} = \arg\max_\theta \mathcal{L}_{\theta_k}(\theta) - \beta_k \widetilde D_{kl}(\theta || \theta_k)$$
其中：
$$\mathcal{L_{\theta_k}}(\theta) = E_{\tau \sim \pi} \left[ \sum_{t = 0}^\infty \gamma^t r_t(\theta) A^{\pi_k} \right]$$
$$r_t(\theta) =  \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_k}(a_t | s_t)}$$
$$\widetilde D_{KL}(\theta || \theta_k) = E_{s \sim d^{\pi_k}}[D_{KL}(\pi_{\theta}(\cdot | s), \pi_{\theta_k}(\cdot | s))]$$

PPO with Adaptive KL Penalty:
- Input: initial policy parameters $\theta_0$, initial KL penalty $\beta_0$, target KL-divergence $\delta$
- for $k = 0, 1, 2, ...$ do
	- Collect set of partial trajectories $D_k$ on policy $\pi_k = \pi(\theta_k )$
	- Estimate advantages $\hat A_t^{\pi_k}$ t using any advantage estimation algorithm
	- Compute policy update by taking K steps of minibatch SGD (via Adam)
$$\theta_{k + 1} = \arg\max_\theta \mathcal{L}_{\theta_k}(\theta) - \beta_k \widetilde D_{kl}(\theta || \theta_k)$$
	- if $\widetilde D_{KL}(\theta_{k + 1} || \theta) \geq 1.5 \delta$ then
		- $\beta_{k + 1} = 2 \beta_k$
	- else if $\widetilde D_{KL}(\theta_{k + 1} || \theta) \leq \delta / 1.5$
		- $\beta_{k + 1} = \beta_k / 2$
	- end if
- end for

- Initial KL penalty not that important—it adapts quickly 
- Some iterations may violate KL constraint, but most don’t
## Clipped Objective
另一种方法是对目标函数进行裁剪：
$$\theta_{k + 1} = \arg\max_\theta \mathcal{L}_{\theta_k}^{CLIP}(\theta)$$
其中，$\epsilon$ 是超参数，且：
$$\mathcal{L}_{\theta_k}^{CLIP}(\theta) = E_{\tau \sim \pi_k} \left[ \sum_{t = 0}^T \gamma^t \min \left( r_t(\theta) \hat A_t^{\pi_k}, \text{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon) \hat A_t^{\pi_k} \right) \right]$$
$$r_t(\theta) = \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_k}(a_t | s_t)}$$

当 $A > 0$ 且 $r_t(\theta) > 1 - \epsilon$，或 $A < 0$ 且 $r_t(\theta) < 1 - \epsilon$ 时，梯度为 0。

PPO with Clipped Objective:
- Input: initial policy parameters $\theta_0$, clipping threshold $\epsilon$
- for $k = 0, 1, 2, ...$ do 
	- Collect set of partial trajectories $D_k$ on policy $\pi_k = \pi(\theta_k)$
	- Estimate advantages $\hat A_t^{\pi_l}$ using any advantage estimation algorithm
	- Compute policy update by taking K steps of minibatch SGD (via Adam)￥￥
$$\theta_{k + 1} = \arg\max_\theta \mathcal{L}_{\theta_k}^{CLIP}(\theta)$$
- end for

在 ChatGPT 的 RLHF 中，因为生成文本序列时需要进行采样，所以我们无法直接通过反向传播的方法优化奖励，所以需要进行梯度估计（如 REINFORCE 等）。OpenAI 使用 PPO 进行 RLHF。
[[梯度估计]]

注：Policy gradient 中，我们给出的是梯度的估计量，在使用自动微分库实现 policy gradient 时，可以根据梯度的估计量构造目标函数（[[A2]]）。而 PPO 中，我们给出的是目标函数。

