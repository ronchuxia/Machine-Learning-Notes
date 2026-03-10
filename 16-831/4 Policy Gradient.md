RL methods:
- Model based
- Q-function based
- Gradient based

Policy gradient methods optimize a reinforcement learning agent's behavior by directly computing the gradient of the expected reward with respect to the policy parameters and updating them via gradient ascent.

Requirements:
- Action: continuous or discrete, stochastic, differentiable.
- State: continuous or discrete.

# Likelihood Ratio Policy Gradient
Policy:
$$\pi_\theta = \mathcal{N}(\mu_\theta(s), \sigma)$$

Objective:
$$\theta^* = \arg\max \mathbb{E}_{\tau \sim p_\theta(\tau)}[R(\tau)]$$

Gradient ascent:
$$U(\theta) = \mathbb{E}_{\tau \sim p_\theta(\tau)}[R(\tau)]$$
$$\theta \leftarrow \theta + \alpha \nabla_\theta U(\theta)$$

How to compute the gradient $\nabla_\theta U(\theta)$:
$$\begin{align}
\nabla_\theta U(\theta) & = \nabla_\theta \sum_\tau P_\theta(\tau) R(\tau)\\
& = \sum_\tau \nabla_\theta P_\theta(\tau) R(\tau)\\
& = \sum_\tau P_\theta(\tau) \frac{\nabla_\theta P_\theta(\tau)}{P_\theta(\tau)} R(\tau)\\
& = E_{\tau \sim P_\theta(\tau)} \left[ \frac{\nabla_\theta P_\theta(\tau)}{P_\theta(\tau)} R(\tau) \right]\\
& \approx \frac{1}{m} \sum_{i=1}^m \frac{\nabla_\theta P_\theta(\tau^{(i)})}{P_\theta(\tau^{(i)})} R(\tau^{(i)})
\end{align}$$

How to compute the gradient $\nabla_\theta P_\theta(\tau^{(i)})$:
$$p_\theta(\tau) = p(s_0) \prod_{t=0}^T \pi_\theta(a_t|s_t) p(s_{t+1}|s_t, a_t)$$
We need the dynamics $p(s_{t+1}|s_t, a_t)$. 

Options:
1. Analytical physics
2. Learn the dynamics
3. Likelihood ratio trick

**Likelihood ratio trick**:
$$\begin{aligned}
\frac{\nabla_\theta P_\theta(\tau^{(i)})}{P_\theta(\tau^{(i)})} & =
\nabla_\theta \log P_\theta(\tau^{(i)}) \\
& = \nabla_\theta \log \left[ p(s_1) \prod_{t=1}^H \pi_\theta(a_t|s_t) p(s_{t+1}|s_t, a_t) \right]\\
& = \sum_{t=0}^H \nabla_\theta \log \pi_\theta(a_t | s_t)
\end{aligned}$$

Therefore the the gradient $\nabla_\theta U(\theta)$:
$$\begin{aligned}
\nabla_\theta U(\theta) & = \frac{1}{m} \sum_{i=1}^m \frac{\nabla_\theta P_\theta(\tau^{(i)})}{P_\theta(\tau^{(i)})} R(\tau^{(i)})\\
& = \frac{1}{m} \sum_{i=1}^m \sum_{t=0}^H \nabla_\theta \log \pi_\theta (a_t | s_t) R(\tau^{(i)})
\end{aligned}$$

Gradient tries to:
- Increase probability of paths with positive R.
- Decrease probability of paths with negative R.

If all trajectories have positive R but some are larger than others, this will still work but it will take a long time to converge!

# Likelihood Ratio Policy Gradient: Baseline
The likelihood ratio policy gradient estimate is **unbiased but noisy**. It takes many samples to converge.

The value function:
$$V^\pi(s) = E_{a_t:\infty} [R | s_t = s]$$
Learning this function will allow us to significantly speed up how quickly we can learn a policy!

Consider **baseline** b:
$$\nabla_\theta U(\theta) \approx \frac{1}{m} \sum_{i=1}^m \sum_{t=0}^H \nabla_\theta \log \pi_\theta(a_t | s_t) (R(\tau_i) - b)$$
We can subtract any constant b without affecting the mean of the gradient estimator (meaning the gradient estimator is **still unbiased**). A good choice of b will **reduce the noise** in the gradient!

NOTE: The baseline b does not have to be a constant, e.g. the state-dependent expected return shown below, as long as it does not affect the mean of the gradient estimator (meaning the gradient estimator is still unbiased).

Some baseline choices:
- Constant baseline (average over all past trajectories).
$$b = \mathbb{E}[R(\tau)] \approx \frac{1}{m} \sum_{i=1}^m R(\tau^{(i)})$$
- Optimal constant baseline (minimize the variance of the estimated gradient).
- State-dependent expected return.
$$b(s_t) = V^\pi(s_t)$$

You can use the following methods to estimate $V^\pi(s_t)$:
- Monte Carlo.
- Neural Network.

# Likelihood Ratio Policy Gradient: Intuition
Repeat until convergence:
1. Sample m trajectories on robot.
2. Evaluate how good each trajectory was (compute the return).
3. Also compute the probability of each trajectory (with likelihood ratio trick to avoid need for dynamics model).
4. Update policy to make good trajectories more likely, bad trajectories less likely.

# Likelihood Ratio and Temporal Structure
The gradient estimator:
$$\begin{aligned}
\hat g & = \frac{1}{m} \sum_{i=1}^m \nabla_\theta \log P(\tau^{(i)}; \theta) (R(\tau^{(i)}) - b)\\
& = \frac{1}{m} \sum_{i=1}^m \left( \sum_{t=0}^H \nabla_\theta \log \pi_\theta(a_t^{(i)} | s_t^{(i)}) \right) \left( \sum_{t=0}^H R(s_t^{(i)}, a_t^{(i)}) - b \right)\\
& = \frac{1}{m} \sum_{i=1}^m \left[ \sum_{t=0}^H \nabla_\theta \log \pi_\theta(a_t^{(i)} | s_t^{(i)}) \left( \sum_{k=0}^{t-1} R(s_k^{(i)}, a_k^{(i)})  + \sum_{k=t}^H R(s_k^{(i)}, a_k^{(i)}) - b \right) \right]\\
& = \frac{1}{m} \sum_{i=1}^m \left[ \sum_{t=0}^H \nabla_\theta \log \pi_\theta(a_t^{(i)} | s_t^{(i)}) \left( \sum_{k=t}^H R(s_k^{(i)}, a_k^{(i)}) - b \right) \right]
\end{aligned}$$

Intuition: The past rewards $\sum_{k=0}^{t-1} R(s_k^{(i)}, a_k^{(i)})$ does not depend on the current action $a_t^{(i)}$.
Formal proof: $\mathbb{E}_\tau \left[ \sum_{t=0}^H \nabla_\theta \log \pi_\theta(a_t | s_t) \left( \sum_{k=0}^{t-1} R(s_k, a_k) \right) \right] = 0$.

NOTE: In fact, anything that does not depend on the current action $a_t^{(i)}$ can be chosen as the baseline, because it does not affect the mean of the gradient estimator.

Removing the terms that don't depend on current action can lower variance.

# TRPO (Trust Region Policy Optimization)
How do we choose the step size for policy gradient?
In RL, the agent generates its own data: Bad step size -> bad dataset -> cannot recover.

We might want to **constrain the policy based on the KL divergence**:
$$\max_\alpha \ \mathrm{s.t.} \ KL(P(\tau;\theta) || P(\tau; \theta + \alpha \nabla_\theta U(\theta))) \leq \epsilon$$

This is the idea behind TRPO (Trust Region Policy Optimization).

However, KL divergence can be slow to compute – PPO speeds it up!

# PPO (Proximal Policy Optimization)
Objective function of policy gradient with baseline:
$$L(\theta) = - \log \pi_\theta(a_t | s_t) \hat A_t$$
$$\hat A_t = R(\tau) - V^\pi(s_t)$$

Objective function of PPO:
$$L(\theta) = - \min \left[ r_t(\theta)\hat A_t, \mathrm{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat A_t \right]$$
$$r_t(\theta) = \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{\mathrm{old}}}(a_t | s_t) }$$
$$\hat A_t = R(\tau) - V^\pi(s_t)$$

![[PPO.png]]
- If action $a_t$ is good, $\hat A_t > 0$, we want to increase $r_t(\theta)$. But the objective function discourages $r_t(\theta)$ from being larger than $1 + \epsilon$.
- If action $a_t$ is bad, $\hat A_t < 0$, we want to decrease $r_t(\theta)$. But the objective function discourages $r_t(\theta)$ from being smaller than $1 - \epsilon$.

Compared with standard policy gradient, PPO employs **importance sampling**. This mathematical trick allows you to estimate the gradient of a new policy using data collected by an old policy. You can run multiple steps of gradient descent on the same batch of data.

Importance sampling:
$$\mathbb{E}_{x\sim P}​[f(x)] = \mathbb{E}_{x\sim Q}​ \left[ f(x) \frac{Q(x)}{P(x)} ​\right]$$

In PPO, $f(x) = \hat A$.

NOTE: If you let $\theta_{old} = \theta$, then $\nabla_\theta L_{\mathrm{PPO}}(\theta) = \nabla_\theta L_\mathrm{PolicyGradient}(\theta)$.