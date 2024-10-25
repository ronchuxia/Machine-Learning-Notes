# RLHF
训练 RL 时需要人类的介入，以让 RL agent 符合人类的偏好或价值观。
- Demonstration only (imitation learning)
- Pairwise labels (RLHF)
- Constant teaching (DAGGER)

对于人类，pairwise comparisons 比 scalar reward 更容易获得。

Feedback comes as preferences over model samples: $\mathcal{D} = \{x^i, y_w^i, y_l^i\}$
- $x^i$: prompt
- $y_w^i$: preferred response
- $y_l^i$: dispreferred response

**Bradley-Terry Model** connects rewards to preferences:
$$p(y_w \succ y_l | x) = \sigma(r(x, y_w) - r(x, y_l))$$

RLHF:
- Train the **reward model** by minimizing negative log likelihood:
$$\mathcal{L}_R(\phi, \mathcal{D}) = - E_{(x, y_w, y_l) \sim \mathcal{D}}[\log \sigma(r_\phi(x, y_w) - r_\phi(x, y_l))]$$
- Learn a policy $\pi_\theta$ achieving high reward while staying close to original model $\pi_{\text{ref}}$ (e.g. PPO):
$$\max_{\pi_\theta} E_{x \sim \mathcal{D}, y \sim \pi_\theta(y | x)}[r_\phi(x, y)] - \beta D_{\text{KL}}(\pi_\theta(\cdot | x) || \pi_{\text{ref}}(\cdot | x))$$

TODO: Secrets of RLHF in Large Language Models Part I: PPO, Zheng et.al. 2023

# DPO
根据 RLHF 的目标函数，我们可以推导出 closed-form optimal policy：
$$\pi^*(y | x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y | x) \exp \left( \frac{1}{\beta}r(x, y) \right)$$
其中：
$$Z(x) = \sum_y \pi_{\text{ref}}(y | x) \exp \left( \frac{1}{\beta} r(x, y) \right)$$

为了计算 $Z(x)$，我们需要对所有可能的 response $y$ 求和。但是，我们无法做到这一点。

重新整理上式，将奖励函数写成最优策略的函数：
$$r(x, y) = \beta \log \frac{\pi^*(y | x)}{\pi_{\text{ref}}(y | x)} + \beta \log Z(x)$$
注：这里的奖励函数可以是任意的奖励函数，最优策略则是这个奖励函数下的最优策略。

将其带回到 RLHF 训练奖励模型的损失函数中，我们得到 DPO 的损失函数：
$$\mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = - E_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w | x)}{\pi_{\text{ref}}(y_w | x)} - \beta \log \frac{\pi_\theta(y_l | x)}{\pi_{\text{ref}}(y_l | x)} \right) \right]$$








