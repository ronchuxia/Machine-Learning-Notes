# Deep Q-Network (DQN)
Requirements:
- Action: discrete.
- State: continuous or discrete.

In Q-learning, the target is created using the current estimate of $Q_\theta(s', a')$, which keeps changing during the learning process.

To make the target more stable, DQN uses a **target network** $Q_{\theta^-}(s', a')$. 
- It has the same architecture as $Q_\theta(s', a')$ but a different set of parameters. 
- Its parameters are periodically updated.
	- Hard update: every N steps, $\theta^- \leftarrow \theta$.
	- Soft update: every step, $\theta^- \leftarrow \tau \theta + (1 - \tau) \theta^-$.

Instead of using $Q_\theta(s', a')$ to create the target, DQN uses $Q_{\theta^-}(s', a')$ to create the target.
$$\theta = \arg\min_\theta \mathbb{E}_{(s, a, r, s')\sim D} \left[ \left(r + \gamma \max_{a' \in \mathcal A}Q_{\theta^-}(s', a') - Q_\theta(s, a) \right)^2 \right]$$

# Double DQN
Requirements:
- Action: discrete.
- State: continuous or discrete.

In DQN, the target is:
$$\begin{aligned}
y & = r(s) + \gamma \max_{a' \in \mathcal{A}}Q_{\theta^-}(s', a')\\
& = r(s) + \gamma Q_{\theta^-} \left(s', \arg\max_{a' \in \mathcal{A}} Q_{\theta^-}(s', a') \right)
\end{aligned}$$

Because $Q_\theta(s', a')$ is a noisy estimation of the Q function, the target will choose the action whose Q value is overestimated, which causes the target to be overestimated. This **overestimation** accumulates over training. This is called the **maximization bias**.

To remove the maximization bias, **double Q-learning** use two target networks Q1 and Q2. At each step, pick Q1 or Q2 at random to be updated. 

In **double DQN**, we reuse the **online network** $Q_{\theta}$ and the target network $Q_{\theta^-}$. The target is:
$$y = r(s) + \gamma Q_{\theta^-} \left( s', \arg\max_{a' \in \mathcal{A}} Q_\theta(s', a') \right)$$

# Deep Deterministic Policy Gradient (DDPG)
Requirements:
- Action: continuous.
- State: continuous or discrete.

**Deep Deterministic Policy Gradient (DDPG)** combines DQN with actor-critic to handle continuous action spaces.
- Actor: $\pi_\phi(s)$.
- Critic: $Q_\theta(s, a)$.

Critic update:
$$y = r(s) + \gamma Q_{\theta^-}(s', \pi_\phi(s'))$$

Actor update:
$$\nabla_\phi J = \mathbb{E}[\nabla_a Q_\theta(s, a)|_{a = \pi_\phi(s)} \cdot \nabla_\phi \pi_\phi(s)]$$

# Twin Delayed Deep Deterministic Policy Gradient (TD3)
DDPG also suffers from the maximum bias. **Twin delayed deep deterministic policy gradient (TD3)** removes the maximum bias by using two critic networks. At each step, both critic networks are updated.
$$y = r(s) + \min \left( Q_{\theta_1^-}(s', \pi_\phi(s')), Q_{\theta_2^-}(s', \pi_\phi(s')) \right)$$

# Extensions
- Prioritized experience replay: prioritize samples with high TD error.
- Distributional Q-learning: model the return/reward as a distribution.
- Dueling architecture: Separate outputs for $V(s)$ and $A(s, a)$.