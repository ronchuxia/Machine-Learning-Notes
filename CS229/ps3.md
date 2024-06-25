# A Simple Neural Network
1. 反向传播
2. 可以，隐藏层的每个神经元都是一个线性分类器，共有三个神经元，每个神经元作为一条边，就可以划分出三角形区域
3. 不可以，如果没有非线性的激活函数，整个神经网络会退化为一个线性分类器

TODO：如果使用 sigmoid 激活函数呢？

# KL Divergence and Maximum Likelihood
分布 $P(X)$ 与分布 $Q(X)$ 的 KL 散度定义为：
$$D_{KL}(P||Q) = \sum_{x \in \mathcal{X}} P(x) \log \frac{P(x)}{Q(x)}$$
**KL 散度描述了两个分布之间的差距**。

在信息论中，分布 $P(X)$ 的**熵**定义为：
$$H(P) = - \sum_{x \in \mathcal{X}} P(x) \log P(x)$$
**熵描述了一个分布的分散程度**。分布越分散，熵越大。分布越集中，熵越小。
在通信理论中，假设符号服从分布 $P(x)$，则熵等于符号的最短平均编码长度。

在信息论中，分布 $P(X)$ 与分布 $Q(X)$ 的**交叉熵**定义为：
$$H(P, Q) = - \sum_{x \in \mathcal{X}} P(x) \log Q(x)$$
在通信理论中，假设符号 A 服从分布 $P(X)$，符号 B 服从分布 $Q(X)$，则交叉熵等于对符号 A 使用符号 B 的最佳编码方式进行编码时，符号 A 的平均编码长度。

$$\begin{align}
D_{KL}(P||Q) & = \sum_{x \in \mathcal{X}} P(x) \log \frac{P(x)}{Q(x)}\\
& = \sum_{x \in \mathcal{X}} P(x) \log P(x) - \sum_{x \in \mathcal{X}} P(x) \log Q(x)\\
& = H(P, Q) - H(P)
\end{align}$$
据此，KL 散度等于对符号 A 使用符号 B 的最佳编码方式进行编码时，符号 A 的平均编码长度比符号 A 的最短平局编码长度长了多少。

在机器学习中，通常需要寻找分布 Q，使它足够接近分布 P。为此，我们可以将 KL 散度作为损失函数，最小化 KL 散度。又因为 $H(P)$ 固定，我们可以将交叉熵作为损失函数，最小化交叉熵。

可以证明，对于密度估计问题，**MLE 等价于最小化训练数据（经验分布）与模型的 KL 散度**。

1. 证明 KL 散度非负
由 Jensen 不等式：
$$\begin{align}
D_{KL}(P||Q) & = \sum_{x \in \mathcal{X}} P(x) (- \log \frac{Q(x)}{P(X)})\\
& = E_{x \sim P} \left[ - \log \frac{Q(x)}{P(x)} \right]\\
& \geq - \log E_{x \sim P} \left[ \frac{Q(x)}{P(X)} \right]\\
& = - \log \sum_{x \in \mathcal{X}} Q(x)\\
& = 0
\end{align}$$
且：
$$D_{KL}(P||Q) = 0 \ \text{if and only if} \ P = Q$$

2. 条件分布 $P(X|Y)$ 与条件分布 $Q(X|Y)$ 的 KL 散度定义为： 
$$D_{KL}(P(X|Y) || Q(X|Y)) = \sum_y P(y) \left( \sum_x P(x|y) \log \frac{P(x|y)}{Q(x|y)} \right)$$
证明 KL 散度的链式法则：
$$\begin{align}
D_{KL}(P(X, Y) || Q(X, Y)) & = \sum_x \sum_y P(x, y) \log \frac{P(x, y)}{Q(x, y)}\\
& = \sum_x \sum_y P(y | x) P(x) \log \frac{P(y | x) P(x)}{Q(y | x) Q(x)}\\
& = \sum_x \sum_y P(y | x) P(x) \log \frac{P(x)}{Q(x)} + \sum_x \sum_y P(y | x) P(x) \log \frac{P(y | x)}{Q(y | x)}\\
& = \sum_x P(x) \log \frac{P(x)}{Q(x)} + \sum_x P(x) \sum_y P(y | x) \log \frac{P(y | x)}{Q(y | x)}\\
& = D_{KL}(P(X) || Q(X)) + D_{KL}(P(Y|X) || Q(Y|X))
\end{align}$$

3.  对于密度估计问题，定义经验分布：
$$\hat P(x) = \frac{1}{m} \sum_{i = 1}^m 1\{x^{(i)} = x\}$$
即经验分布就是训练数据集上的均匀分布。从经验分布中采样相当于从训练数据集中随机选择一个样本。

证明 MLE 等价于最小化训练数据（经验分布）与模型的 KL 散度：
$$\begin{align}
\arg\min_\theta D_{KL}(\hat P || P_\theta) & = \arg\min_\theta \sum_{i = 1}^m \hat P(x^{(i)}) \log \frac{\hat P(x^{(i)})}{P_\theta(x^{(i)})}\\
& = \arg\min_\theta \sum_{i = 1}^m \hat P(x^{(i)}) \log \hat P(x^{(i)}) - \sum_{i = 1}^m \hat P(x^{(i)}) \log P(x^{(i)})\\
& = \arg\min_\theta \log \frac{1}{m} - \frac{1}{m} \sum_{i = 1}^m \log P_\theta(x^{(i)})\\
& = \arg\max_\theta \sum_{i = 1}^m \log P_\theta(x^{(i)})
\end{align}$$

# KL Divergence, Fisher Information, and the Natural Gradient
在对 KL 散度进行梯度下降时，梯度的大小与参数有关。对于不同的参数，如果使用相同的步长，则 KL 散度的变化有可能很大，也有可能很小。

**自然梯度**（Natural Gradient）是一种优化算法，它是对传统梯度下降算法的一种改进，特别适用于概率模型参数的更新。在每一步迭代中，自然梯度在当前参数的附近选择一组新的参数，使得**新的分布与当前分布的 KL 散度为一定值**，并最大化对数似然函数（即**最小化经验分布与新的分布的 KL 散度**）。

1. 假设随机变量 Y 服从分布 $p(y; \theta)$。$p(y; \theta)$ 的 **score function** 定义为 $\nabla_\theta \log p(y; \theta)$，即对数似然函数的梯度。Score function 描述了似然函数对参数的敏感度。

证明 score function 的期望为 0：
$$\begin{align}
E_{y \sim p(y; \theta)}[\nabla_{\theta} \log p(y; \theta)] & = \int_{-\infty}^{+\infty} p(y; \theta) \nabla_\theta \log p(y; \theta) dy\\
& = \int_{-\infty}^{+\infty} \nabla_\theta  p(y; \theta) dy\\
& = \nabla_\theta \int_{-\infty}^{+\infty} p(y; \theta) dy\\
& = 0
\end{align}$$
注：用到了 Leibniz 积分法则交换求导和积分的顺序。

2. 定义 **Fisher information** 为 score function 的协方差矩阵：
$$\mathcal{I}(\theta) = Cov_{y \sim p(y; \theta)} [\nabla_\theta \log p(y; \theta)]$$
Fisher information 描述了随机变量 Y 携带的有关参数 $\theta$ 的信息量。

由于 score function 的期望为 0：
$$\mathcal{I}(\theta) = E_{y \sim p(y; \theta)} [\nabla_\theta \log p(y; \theta) \nabla_\theta \log p(y; \theta)^T]$$

3. 证明 Fisher information 也可以写成对数似然函数的海森矩阵的负的期望：
$$\begin{align}
E_{y \sim p(y; \theta)} [- \nabla_\theta^2 \log p(y; \theta)]_{ij} & = E_{y \sim p(y; \theta)} \left[ - \frac{1}{p(y; \theta)} \frac{\partial^2 p(y; \theta)}{\partial \theta_i \partial \theta_j} + \frac{1}{p(y; \theta)^2} \frac{\partial p(y; \theta)}{\partial \theta_i} \frac{\partial p(y; \theta)}{\partial \theta_j} \right]\\
& = - \int_{- \infty}^{+ \infty} \frac{\partial^2 p(y; \theta)}{\partial \theta_i \partial \theta_j} dy + E_{y \sim p(y; \theta)} \left[ \frac{1}{p(y; \theta)^2} \frac{\partial p(y; \theta)}{\partial \theta_i} \frac{\partial p(y; \theta)}{\partial \theta_j} \right]\\
& = - \frac{\partial^2}{\partial \theta_i \partial \theta_j} \int_{- \infty}^{+ \infty} p(y; \theta) dy + E_{y \sim p(y; \theta)} \left[ \frac{1}{p(y; \theta)^2} \frac{\partial p(y; \theta)}{\partial \theta_i} \frac{\partial p(y; \theta)}{\partial \theta_j} \right]\\
& = E_{y \sim p(y; \theta)} \left[ \frac{1}{p(y; \theta)^2} \frac{\partial p(y; \theta)}{\partial \theta_i} \frac{\partial p(y; \theta)}{\partial \theta_j} \right]\\
& = E_{y \sim p(y; \theta)} [\nabla_\theta \log p(y; \theta) \nabla_\theta \log p(y; \theta)^T]_{ij}\\
& = \mathcal{I}(\theta)_{ij}
\end{align}$$
所以：
$$E_{y \sim p(y; \theta)} [- \nabla_\theta^2 \log p(y; \theta)] = \mathcal{I}(\theta)$$

函数的海森矩阵就是这个函数在当前点处的曲率。因此，**Fisher information 就是对数似然函数在当前参数处的曲率**。

4. 可以用 Fisher information 估计 KL 散度：
$$\begin{align}
D_{KL}(p_\theta || p_{\theta + d}) & = \int_{-\infty}^{+\infty} p_\theta(y) \log \frac{p_\theta(y)}{p_{\theta + d}(y)} dy\\
& = \int_{-\infty}^{+\infty} p_\theta(y) \log p_\theta(y) dy - \int_{-\infty}^{+\infty} p_\theta(y) \log p_{\theta + d}(y) dy\\
& \approx \int_{-\infty}^{+\infty} p_\theta(y) \log p_\theta(y) dy - \int_{-\infty}^{+\infty} p_\theta(y) (\log p_\theta(y) + d^T \nabla_\theta \log p_\theta(y) + \frac{1}{2} d^T \nabla^2_\theta \log p_\theta(x) d) dy\\
& = - \int_{-\infty}^{+\infty} p_\theta(y) d^T \nabla_\theta \log p_\theta(y) dy - \int_{-\infty}^{+\infty} p_\theta(y) \frac{1}{2} d^T \nabla^2_\theta \log p_\theta(x) d dy\\
& = - d^T E_{y \sim p_\theta(y)}[\nabla_\theta \log p_\theta(y)] - \frac{1}{2} d^T E_{y \sim p_\theta(x)}[\nabla_\theta^2 \log p_\theta(x)] d\\
& = - \frac{1}{2} d^T \mathcal{I}(\theta) d
\end{align}$$

5. 假设对数似然函数为 $l(\theta) = \log p(y; \theta)$，新的分布与当前分布的 KL 散度为常数 $c$，则自然梯度：
$$d^* = \arg\max_d l(\theta + d)$$
$$D_{KL}(p_\theta || p_{\theta + d}) = c$$
使用泰勒展开近似 $l(\theta + d) \approx l(\theta) + d^T \nabla_\theta l(\theta)$，求自然梯度的解析解。

这是一个约束优化问题，构造拉格朗日函数：
$$\begin{align}
\mathcal{L}(d, \lambda) & = l(\theta + d) - \lambda (D_{KL}(p_\theta || p_{\theta + d}) - c)\\
& \approx l(\theta) + d^T \nabla_\theta l(\theta) - \lambda \left( \frac{1}{2} d^T \mathcal{I}(\theta) d - c \right)
\end{align}$$

让导数等于 0：
$$\nabla_d \mathcal{L}(d, \lambda) = \nabla_\theta l(\theta) - \lambda \mathcal{I}(\theta) d = 0$$
$$\nabla_\lambda \mathcal{L}(d, \lambda) = \frac{1}{2} d^T \mathcal{I}(\theta)^{-1} d - c = 0$$

则：
$$d = \frac{1}{\lambda} \mathcal{I}(\theta)^{-1} \nabla_\theta l(\theta)$$
$$\lambda = \sqrt{\frac{\nabla_\theta l(\theta)^T \mathcal{I}(\theta)^{-1} \nabla_\theta l(\theta)}{2c}}$$

6. GLM 中：
$$H(l(\theta)) = Var[y; \theta] xx^T$$
与 y 的值无关，所以：
$$\mathcal{I}(\theta) = E_{y \sim GLM(y; \theta^T x)}[- H(l(\theta))] = - H(l(\theta))$$
所以，GLM 中，牛顿法与自然梯度等价。

# Semi-supervised EM
对于半监督学习，假设除了 $m$ 个无标签的样本 $\{ x^{(1)}, x^{(2)}, \cdots, x^{(m)}\}$ 外，还有 $\hat m$ 个有标签的样本 $\{ (\hat x^{(1)}, \hat z^{(1)}), (\hat x^{(2)}, \hat z^{(2)}), \cdots, (\hat x^{(\hat m)}, \hat z^{(\hat m)}) \}$。半监督学习最大化无标签样本的边缘似然函数与有标签样本的联合似然函数的加权和：
$$\begin{align}
l_{unsup}(\theta) & = \sum_{i = 1}^m \log p(x^{(i)}; \theta)\\
& = \sum_{i = 1}^m \log \sum_{z^{(i)}} p(x^{(i)})
\end{align}$$
$$l_{sup}(\theta) = \sum_{i = 1}^{\hat m} \log p(\hat x^{(i)}, \hat z^{(i)}; \theta)$$
$$l_{semi-sup}(\theta) = l_{unsup}(\theta) + \alpha l_{sup}(\theta)$$
类似之前 EM 算法的推导过程，可以证明，半监督学习的 EM 算法如下：
1. E-step：对 $i = 1, 2, \cdots, m$，计算条件概率
$$Q_i(z^{(i)}) := p(z^{(i)} | x^{(i)}; \theta)$$
2. M-step：更新参数
$$\theta := \arg\max_\theta \left( \sum_{i = 1}^m \sum_{z^{(i)}} Q_i(z^{(i)}) \log \frac{p(x^{(i)}, z^{(i)}; \theta)}{Q_i(z^{(i)})} + \alpha \sum_{i = 1}^{\hat m} \log p(\hat x^{(i)}, \hat z^{(i)}; \theta) \right)$$

1. 证明半监督学习的 EM 算法的收敛性：
$$l_{semi-sup}(\theta^{(t + 1)}) \geq l'_{semi-sup}(\theta^{(t + 1)}) \geq l'_{semi-sup}(\theta^{(t)}) = l_{semi-sup}(\theta^{(t)})$$

2. 半监督高斯混合模型的 E-step：
$$w_j^{(i)} = Q_i(z^{(i)} = j) = p(z^{(i)} = j | x^{(i)}; \phi, \mu, \Sigma)$$

3. 半监督高斯混合模型的 M-step：
$$\phi_j = \frac{\sum_{i = 1}^m w_j^{(i)} + \alpha \sum_{i = 1}^{\hat m} 1\{\hat z^{(i)} = j\}}{m + \alpha \hat m}$$
$$\mu_j = \frac{\sum_{i = 1}^m w^{(i)}_j x^{(i)} + \alpha \sum_{i = 1}^{\hat m} 1\{\hat z^{(i)} = j\} \hat x^{(i)}}{\sum_{i = 1}^m w^{(i)}_j + \alpha \sum_{i = 1}^{\hat m} 1\{\hat z^{(i)} = j\}}$$
$$\Sigma_j = \frac{\sum_{i = 1}^m w_j^{(i)} (x^{(i)} - \mu_j) (x^{(i)} - \mu_j)^T + \alpha \sum_{i = 1}^{\hat m} 1\{\hat z^{(i)} = j\} (\hat x^{(i)} - \mu_j) (\hat x^{(i)} - \mu_j)^T}{\sum_{i = 1}^m w_j^{(i)} + \alpha \sum_{i = 1}^{\hat m} 1\{\hat z^{(i)} = j\}}$$

4. 实现无监督 GMM 的 EM 算法。
5. 实现半监督 GMM 的 EM 算法。
6. 半监督 EM 算法与无监督 EM 算法相比：
	1. 收敛更快
	2. 更稳定
	3. 质量更高

# K-means for Compression
1. 实现 K-means 算法
2. 6