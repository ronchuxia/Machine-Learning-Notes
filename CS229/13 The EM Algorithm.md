# Jensen's Inequality
如果 $f$ 是凸函数，$X$ 是随机变量，则：
$$E[f(X)] \geq f(EX)$$

特别地，如果 $f$ 是严格凸函数，则 $E[f(X)] = f(EX)$ 当且仅当随机变量 $X$ 是常数。

# The EM algorithm
当模型含有隐变量时，显式地求解使 $l(\theta)$ 最大化的参数 $\theta$ 可能是困难的。

为了最大化 $l(\theta)$，EM 算法：
1. E-step（estimation）：构建 $l(\theta)$ 的下界
2. M-step（maximization）：最大化该下界
3. 重复 1，2 直到收敛

对 $i = 1, 2, \cdots, m$，如果 $Q_i$ 是与隐变量 $z$ 有关的某个概率分布，则：
$$\begin{align}
\sum_i \log p(x^{(i)}; \theta) & = \sum_i \log \sum_{z^{(i)}} p(x^{(i)}, z^{(i)}; \theta)\\
& = \sum_i \log \sum_{z^{(i)}} Q(z^{(i)}) \frac{p(x^{(i)}, z^{(i)}; \theta)}{Q_i(z^{(i)})}\\
& = \sum_i \log E_{z^{(i)} \sim Q_i}[\frac{p(x^{(i)}, z^{(i)}; \theta)}{Q_i(z^{(i)})}]\\
& \geq \sum_i E_{z^{(i)} \sim Q_i}[\log \frac{p(x^{(i)}, z^{(i)}; \theta)}{Q_i(z^{(i)})}]\\
& = \sum_i \sum_{z^{(i)}} Q_i(z^{(i)}) \log \frac{p(x^{(i)}, z^{(i)}; \theta)}{Q_i(z^{(i)})}
\end{align}$$
这给出了 $l(\theta)$ 的下界。

为了使 EM 算法中 $l(\theta)$ 单调递增，我们要求在当前 $\theta$ 处，下界与 $l(\theta)$ 的值相等。由 Jensen 不等式，为了取等号：
$$\frac{p(x^{(i)}, z^{(i)}; \theta)}{Q_i(z^{(i)})} = c$$
即：
$$\begin{align}
Q_i(z^{(i)}) & = \frac{p(x^{(i)}, z^{(i)}; \theta)}{\sum_z p(x^{(i)}, z; \theta)}\\
& = p(z^{(i)} | x^{(i)}; \theta)
\end{align}$$
据此，我们构建了 $l(\theta)$ 的一个下界。

完整的 EM 算法如下：
1. E-step：对 $i = 1, 2, \cdots, m$，计算条件概率
$$Q_i(z^{(i)}) := p(z^{(i)} | x^{(i)}; \theta)$$
2. M-step：更新参数
$$\theta := \arg\max_\theta \sum_i \sum_{z^{(i)}} Q_i(z^{(i)}) \log \frac{p(x^{(i)}, z^{(i)}; \theta)}{Q_i(z^{(i)})}$$

因为在 EM 算法中 $l(\theta)$ 单调递增，所以 $l(\theta)$ 一定会收敛。因为 $l(\theta)$ 不是凹函数，所以 EM 算法会**收敛到局部最大值**。

定义：
$$J(Q, \theta) = \sum_i \sum_{z^{(i)}} Q_i(z^{(i)}) \log \frac{p(x^{(i)}, z^{(i)}; \theta)}{Q_i(z^{(i)})}$$
则 EM 算法可以看作是对于下界做 coordinate ascent，先固定 $\theta$ 并优化 $Q$，然后固定 $Q$ 并优化 $\theta$。

# Mixture of Gaussians Revisited
可以证明，高斯混合模型的参数更新规则可以由 EM 算法的参数更新规则得到。

注 1：$\nabla_{A^T} tr ABA^T = B^TA^T + BA^T$
注 2：$\nabla_B tr ABA^T = A^TA$
注 3：$\nabla_A |A| = |A|(A^{-1})^T$

对于 $w_j^{(i)}$：
$$w_j^{(i)} = Q_i(z^{(i)} = j) = p(z^{(i)} = j | x^{(i)}; \phi, \mu, \Sigma)$$

则优化目标 $J(\phi, \mu, \Sigma)$ 可以写成：
$$\begin{align}
& \sum_{i = 1}^m \sum_{z^{(i)}} Q_i(z^{(i)}) \log \frac{p(x^{(i)}, z^{(i)}; \phi, \mu, \Sigma)}{Q_i(z^{(i)})}\\
= & \sum_{i = 1}^m \sum_{j = 1}^k Q_i(z^{(i)} = j) \log \frac{p(x^{(i)} | z^{(i)} = j; \mu, \Sigma) p(z^{(i)} = j; \phi)}{Q_i(z^{(i)} = j)}\\
= & \sum_{i = 1}^m \sum_{j = 1}^k w_j^{(i)} \log \frac{\frac{1}{(2\pi)^{n/2} |\Sigma_j|^{1/2}} \exp(- \frac{1}{2} (x^{(i)} - \mu_j)^T \Sigma_j^{-1} (x^{(i)} - \mu_j)) \phi_j}{w_j^{(i)}}
\end{align}$$

对于 $\phi_j$，与 $\phi_j$ 有关的项为：
$$\sum_{i = 1}^m \sum_{j = 1}^k w_j^{(i)} \log \phi_j$$
约束为：
$$\sum_{j = 1}^k \phi_j = 1$$
构造拉格朗日函数：
$$\mathcal{L}(\phi) = \sum_{i = 1}^m \sum_{j = 1}^k w_j^{(i)} \log \phi_j + \beta (\sum_{j = 1}^k \phi_j - 1)$$
令其导数为 0：
$$\frac{\partial}{\partial \phi_j} \mathcal{L}(\phi) = \sum_{i = 1}^m \frac{w^{(i)}_j}{\phi_j} + \beta$$
则：
$$\phi_j = \frac{\sum_{i = 1}^m w_j^{(i)}}{- \beta}$$
为了满足约束：
$$-\beta = \sum_{j = 1}^k \sum_{i = 1}^m w_j^{(i)} = m$$
即：
$$\phi_j = \frac{1}{m} \sum_{i = 1}^m w_j^{(i)}$$

对于 $\mu_j$，与 $\mu_j$ 有关的项为：
$$\sum_{i = 1}^m \sum_{j = 1}^k w_j^{(i)} (- \frac{1}{2}(x^{(i)} - \mu_j)^T \Sigma_j^{-1} (x^{(i)} - \mu_j))$$
令其导数为 0：
$$\begin{align}
& \nabla_{\mu_j} \sum_{i = 1}^m \sum_{j = 1}^k w_j^{(i)} (- \frac{1}{2}(x^{(i)} - \mu_j)^T \Sigma_j^{-1} (x^{(i)} - \mu_j)) \\
= & \frac{1}{2} \sum_{i = 1}^m w_j^{(i)} \nabla_{\mu_j} (2 \mu_j^T \Sigma_j^{-1} x^{(i)} - \mu_j^T \Sigma_j^{-1} \mu_j)\\
= & \sum_{i = 1}^m w_j^{(i)} (\Sigma_j^{-1} x^{(i)} - \Sigma_j^{-1} \mu_j)\\
= & 0
\end{align}$$
则：
$$\mu_j = \frac{\sum_{i = 1}^m w^{(i)}_j x^{(i)}}{\sum_{i = 1}^m w^{(i)}_j}$$

对于 $\Sigma_j$，与 $\Sigma_j$ 有关的项为：
$$\sum_{i = 1}^m \sum_{j = 1}^k w_j^{(i)} (- \frac{1}{2} \log |\Sigma_j|) + w_j^{(i)} (- \frac{1}{2}(x^{(i)} - \mu_j)^T \Sigma_j^{-1} (x^{(i)} - \mu_j))$$
令其导数为 0：
$$\begin{align}
& \nabla_{\Sigma_j^{-1}} \sum_{i = 1}^m \sum_{j = 1}^k - w_j^{(i)} \log |\Sigma_j^{-1}| + w_j^{(i)} (x^{(i)} - \mu_j)^T \Sigma_j^{-1} (x^{(i)} - \mu_j)\\
= & \sum_{i = 1}^m - w_j^{(i)} \Sigma_j + w_j^{(i)} (x^{(i)} - \mu_j) (x^{(i)} - \mu_j)^T\\
= & 0
\end{align}$$
则：
$$\Sigma_j = \frac{\sum_{i = 1}^m w_j^{(i)} (x^{(i)} - \mu_j) (x^{(i)} - \mu_j)^T}{\sum_{i = 1}^m w_j^{(i)}}$$

# Variational Inference and Variational Auto-Encoder
VAE 将 EM 算法推广到更复杂的神经网络模型。

在 EM 算法中，我们构建了 $l(\theta)$ 的一个下界。这个下界称为**证据下界**（evidence lower bound, ELBO）：
$$p(x; \theta) \geq \text{ELBO}(x; Q, \theta) = \sum_{z} Q(z) \log \frac{p(x, z; \theta)}{Q(z)}$$
**上述不等式对于任意的 $Q(z)$ 都成立**。为了取等号，从而保证收敛性，EM 算法选择了后验分布 $p(z | x; \theta)$。

然而，对于复杂的模型，如下列模型：
$$z \sim \mathcal{N}(0, I)$$
$$x | z \sim \mathcal{N}(g(z; \theta), \sigma^2 I)$$
后验分布 $p(z | x; \theta)$ 难以计算。其中，$g(z; \theta)$ 是神经网络，称为 decoder。

因此，VAE 使用神经网络模型对后验分布 $p(z | x; \theta)$ 做近似。

VAE 基于以下假设：
$$Q = \mathcal{N}(q(x; \phi), \text{diag}(v(x; \psi)^2))$$
即**假设后验分布为高斯分布，且隐变量 $z$ 的各个维度独立**。其中，$q(x; \phi)$ 和 $v(x; \psi)$ 是神经网络，称为 encoder，用于拟合 $Q_i$ 的均值和方差，从而使 $Q_i$ 近似实际的后验分布。

注：在很多情况下，让后验分布为高斯分布并不是一个好的假设，但是这个假设对于 VAE 的优化算法是至关重要的。

ELBO 可以写成：
$$\begin{align}
\text{ELBO}(x; Q, \theta) & = E_{z \sim Q} \left[ \log \frac{p(x, z; \theta)}{Q(z)} \right]\\
& = E_{z \sim Q} [\log p(x | z; \theta)] - D_{KL}(Q || p_z)
\end{align}$$
其中：
- $- E_{z \sim Q}[\log p(x | z; \theta)]$ 为重构损失。重构损失可以通过对 $z$ 进行采样，然后用 decoder 重构 $x$，并取均方误差估计。重构损失用于保证模型能够重构输入数据，使生成的数据与原始数据尽可能相似。
- $D_{KL}(Q||p_z)$ 为 KL 散度损失，$p_z$ 一般取 $\mathcal{N}(0, I)$。对于高斯分布，KL 散度损失可以直接根据均值与方差计算得到。KL 散度损失用于使生成的样本更加多样和真实，防止过拟合，增加稳定性。

VAE 使用随机梯度上升算法优化 ELBO。对于 $\theta$，可以直接求梯度。对于 $\phi$ 和 $\psi$，由于重构损失涉及对 $z$ 进行采样，需要使用 re-parameterization trick。采样 $\xi \sim \mathcal{N}(0, I)$，则 $z = q(x; \phi) + v(x; \psi)^2 \xi$，这样就可以对 $\phi$ 和 $\psi$ 求梯度了。







  









