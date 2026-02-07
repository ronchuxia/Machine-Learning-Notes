# Review
**Unsupervised Learning**
Assumption: Data comes from distribution $p^*(\mathbf{x}_0)$
Goal: We learn a distribution $p_\theta(\mathbf{x}_0) \approx p^*(\mathbf{x}_0)$ that is easy to sample from, and sample $\mathbf{x}_0 \sim p_\theta(\mathbf{x}_0)$

**Latent Variable Models**
Sampling is easy: 
$$\mathbf{z} \sim \mathcal{N}(0, 1)$$
$$\mathbf{x}_0 = f_\theta(\mathbf{z})$$
Learning is hard:
$$\theta = \arg\max_\theta \sum_{i=1}^N \log p_\theta(\mathbf{x}^{(i)})$$$$p_\theta(\mathbf{x}_0) = \int_\mathbf{z} p_\theta(\mathbf{x} | \mathbf{z}) p_\theta(\mathbf{z}) d\mathbf{z}$$

# Diffusion Models
![[Diffusion Models.png]]
## Forward Process
Add noise to the image.
$$q_\phi(\mathbf{x}_{o:T}) = q_\phi(\mathbf{x}_0) \prod_{t=1}^T q_\phi(\mathbf{x}_t | \mathbf{x}_{t-1})$$
$$q_\phi(\mathbf{x}_0) = \text{data distribution}$$
$$q_\phi(\mathbf{x}_t | \mathbf{x}_{t-1}) \sim \mathcal{N}(\sqrt{\alpha_t} \mathbf{x}_{t-1}, (1 - \alpha_t)\mathbf{I})$$

**Gaussians (an Aside)**
Let $X \sim \mathcal{N}(\mu_x, \sigma_x^2)$ and $Y \sim \mathcal{N}(\mu_x, \sigma_y^2)$.
1. Sum of two Gaussians is a Gaussian: $X + Y \sim \mathcal{N}(\mu_x + \mu_y, \sigma_x^2 + \sigma_y^2)$.
2. Difference of two Gaussians is a Gaussian: $X - Y \sim \mathcal{N}(\mu_x - \mu_y, \sigma_x^2 + \sigma_y^2)$.

Let $X \sim \mathcal{N}(\mu_x, \sigma_x^2)$, $g(x) = ax+b$ be a linear/affine function and $Y = g(X)$.
3. Linear/Affine function of a Gaussian is a Gaussian: $Y \sim \mathcal{N}(a\mu_x + b, a^2\sigma_x^2)$.

Let $X \sim \mathcal{N}(\mu_x, \sigma_x^2)$ and $Z|X \sim \mathcal{N}(X, \sigma_z^2)$ (i.e. $Z = X + \epsilon, \epsilon = \mathcal{N}(0, \sigma_z^2)$).
4. Gaussian with a Gaussian mean has a Gaussian marginal: $Z \sim \mathcal{N}(\mu_x, \sigma_x^2 + \sigma_z^2)$.

Let $X \sim \mathcal{N}(\mu_x, \sigma_x^2)$, $g(x) = ax+b$ be a linear/affine function and and $Z | X \sim \mathcal{N}(\mu_z = g(X), \sigma_z^2)$ (i.e. $Z = g(X) + \epsilon, \epsilon \sim \mathcal{N} (0, \sigma_z^2)$).
5. Gaussian with a Gaussian mean passed through a linear/affine function has a Gaussian marginal: 
$$Z \sim \mathcal{N}(a\mu_x + b, a^2\sigma_x^2 + \sigma_z^2)$$

According to the forward process,
$$q_\phi(\mathbf{x}_1 | \mathbf{x}_0) \sim \mathcal{N}(\sqrt\alpha_1 \mathbf{x}_0, (1 - \alpha_1)\mathbf{I})$$
$$q_\phi(\mathbf{x}_2 | \mathbf{x}_1) \sim \mathcal{N}(\sqrt\alpha_2 \mathbf{x}_1, (1 - \alpha_2)\mathbf{I})$$
Therefore, 
$$\begin{aligned}
q_\phi(\mathbf{x}_2 | x_0) & = q_\phi(\mathbf{x}_2 | \mathbf{x}_1)q_\phi(\mathbf{x}_1 | \mathbf{x}_0)\\
& \sim \mathcal{N}(\sqrt{\alpha_2 \alpha_1} \mathbf{x}_0, \alpha_2 (1 - \alpha_1) \mathbf{I} + (1-\alpha_2)\mathbf{I})\\
& \sim \mathcal{N}(\sqrt{\alpha_2\alpha_1}\mathbf{x}_0, (1-\alpha_2\alpha_1)\mathbf{I})
\end{aligned}$$

Inductively, 
$$q_\phi(\mathbf x_T | \mathbf x_0) \sim \mathcal{N} (\sqrt{\bar \alpha_t} \mathbf x_0, (1 - \bar\alpha_t) \mathbf I)$$
$$\bar \alpha_t = \prod_{s=1}^t \alpha_s$$
is a **Gaussian distribution**.

**Noise Schedule**
We choose the sequence $\alpha_t$ to follow a fixed schedule such that:
$$1 > \bar \alpha_1​ > \bar \alpha_2 > \cdots > \bar \alpha_T​ \approx 0$$

Thus,
$$q_\phi(\mathbf{x}_T | \mathbf{x_0}) \sim \mathcal{N}(0, \mathbf{I})$$
is easy to sample from.

# Reverse Process
**Exact Reverse Process**
Given the forward process, we can use the Bayes Rule to derive the exact reverse process:
$$q_\phi(\mathbf{x}_{t-1} | \mathbf{x_t}) = \frac{q_\phi(\mathbf{x}_{t-1}, \mathbf{x}_t)}{q_\phi(\mathbf{x}_t)} = \frac{\int q_\phi(\mathbf{x}_{0:T}) d\mathbf{x}_{0:t-2} d\mathbf{x}_{t+1:T}}{\int q_\phi(\mathbf{x}_{0:T}) d\mathbf{x}_{0:t-1} d\mathbf{x}_{t+1:T}}$$
However, this is intractable because $q_\phi(x_0)$ is unknown!

**Learned Reverse Process**
$$p_\theta(\mathbf{x}_{0:T}) = p_\theta(\mathbf{x}_T) \prod_{t=1}^T p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t)$$
$$p_\theta(\mathbf{x}_T) \sim \mathcal{N}(0, \mathbf{I})$$
$$p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t) \sim \mathcal{N}(\mu_\theta(\mathbf{x}_t, t), \Sigma_\theta(\mathbf{x}_t, t))$$

NOTE: $p_\theta(\mathbf{x}_T)$ and $p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t)$ are Gaussian, but $\mu_\theta$ is a **non-linear** neural network, so $p_\theta(x_0)$ is **no longer a Gaussian**. It will be learned to simulate $q_\phi(x_0)$.

**Gaussians (an Aside)**
Let $X \sim \mathcal{N}(\mu_x, \sigma_x^2)$, $g(x) = ax+b$ be a linear/affine function and and $Z | X \sim \mathcal{N}(\mu_z = g(X), \sigma_z^2)$ (i.e. $Z = g(X) + \epsilon, \epsilon \sim \mathcal{N} (0, \sigma_z^2)$).
6. Gaussian with a Gaussian mean passed through a linear/affine function has a Gaussian posterior: 
$$X|Z \sim \mathcal{N}\left(\mu_x + \frac{a\sigma_x^2}{a^2\sigma_x^2 + \sigma_z^2}(z- (a\mu_x+b)), \left(\frac{1}{\sigma_x^2} + \frac{a^2}{\sigma_z^2} \right)^{-1} \right)$$
NOTE: $\sigma_z^2$ is the likelihood variance, not the marginal variance!

**Properties of the Forward and Exact Reverse Process**
1. Forward process:
$$\mathbf{x}_t = \sqrt{\bar\alpha_t} \mathbf{x}_0 + \sqrt{1 - \bar\alpha_t} \epsilon, \text{where } \epsilon \sim \mathcal{N}(0, \mathbf{I})$$
2. Estimating $q_\phi(\mathbf{x}_{t-1} | \mathbf{x}_t)$ is intractable. However, conditioning on $\mathbf{x}_0$ we can efficiently work with:
$$q(\mathbf{x}_{t-1} | \mathbf{x}_t, \mathbf{x}_0) \sim \mathcal{N}(\tilde \mu_q(\mathbf{x}_t, \mathbf{x}_0), \sigma_t^2 \mathbf{I})$$
$$\tilde \mu_q(\mathbf{x}_t, \mathbf{x}_0) = \frac{\sqrt{\bar \alpha_t} (1- \alpha_t)}{1 - \bar \alpha_t} \mathbf{x}_0 + \frac{\sqrt{\alpha_t} (1 - \bar \alpha_t)}{1 - \bar \alpha_t} \mathbf{x}_t$$
$$\sigma_t^2 = \frac{(1 - \bar\alpha_{t-1})(1 - \alpha_t)}{1 - \bar\alpha_t}$$
3. Combine 1 and 2, we get:
$$\tilde \mu_q (\mathbf{x}_t, \mathbf{x}_0) = \frac{1}{\sqrt{\alpha_t}} \left( \mathbf{x}_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar \alpha_t}} \epsilon \right)$$

**Parameterizing the Learned Reverse Process**
Given a training sample $\mathbf{x}_0$, we want $p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t)$ to be as close as possible to $q(\mathbf{x}_{t-1} | \mathbf{x}_t, \mathbf{x}_0)$.

We want $\Sigma_\theta(\mathbf{x}_t, t)$ to be close to $\sigma_t^2$. Rather than learn $\Sigma_\theta(\mathbf{x}_t, t)$, we use the property of the exact reverse process:
$$\Sigma_\theta(\mathbf{x}_t, t) = \frac{(1 - \bar\alpha_{t-1})(1 - \alpha_t)}{1 - \bar\alpha_t} \mathbf{I}$$
We want $\mu_\theta(\mathbf{x}_t, t)$ to be close to $\tilde\mu_q(\mathbf{x}_t, t)$. There are 3 ways to parameterize this:
1. Learn a network that approximates the $\tilde \mu_q(\mathbf{x}_t, t)$ directly from $\mathbf{x}_t$ and t:
$$\mu_\theta(\mathbf{x}_t, t) = \text{UNet}_\theta(\mathbf{x}_t, t)$$
2. (Property 2) Learn a network that approximates the real $\mathbf{x}_0$ from only $\mathbf{x}_t$ and t:
$$\mathbf{x}_\theta^{(0)} = \text{UNet}_\theta(\mathbf{x}_t, t)$$
$$\mu_\theta(\mathbf{x}_t, t) = \frac{\sqrt{\bar \alpha_t} (1- \alpha_t)}{1 - \bar \alpha_t} \mathbf{x}_\theta^{(0)} + \frac{\sqrt{\alpha_t} (1 - \bar \alpha_t)}{1 - \bar \alpha_t} \mathbf{x}_t$$
3. (Property 3) Learn a network that approximates the $\epsilon$ that gave rise to $\mathbf{x}_t$ from $\mathbf{x}_0$ in the forward process from  $\mathbf{x}_t$ and t:
$$\epsilon_\theta(\mathbf{x}_t, t) = \text{UNet}_\theta(\mathbf{x}_t, t)$$
$$\mu_\theta(\mathbf{x}_t, t) = \frac{1}{\sqrt{\alpha_t}} \left( \mathbf{x}_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar \alpha_t}} \epsilon_\theta(\mathbf{x}_t, t) \right)$$

# Diffusion Model Training
Option 3 is the best empirically.
```
initialize theta

for e in range(1, E) do:
	for x_0 in D do:
		t = Uniform(1, T)
		epsilon = Normal(0, I)
		x_t = sqrt(bar_alpha_t) * x_0 + sqrt(1 - bar_alpha_t) * epsilon
		epsilon_pred = unet(x_t, t)
		loss = (epsilon - epsilon_pred) ** 2
		theta -= loss.grad
```
