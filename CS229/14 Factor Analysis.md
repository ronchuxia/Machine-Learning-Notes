当 $n \geq m$ 时，m 个数据点只能张成 n 维空间的一个低维子空间。此时，如果用 MLE 将数据拟合为高斯分布：
$$\mu = \frac{1}{m} \sum_{i = 1}^m x^{(i)}$$
$$\Sigma = \frac{1}{m} \sum_{i = 1}^m (x^{(i)} - \mu) (x^{(i)} - \mu)^T$$
则**协方差矩阵 $\Sigma$ 是奇异矩阵**（不可逆）。此时，无法计算高斯分布的概率密度。换句话说，MLE 拟合的高斯分布的概率密度全部位于由数据点张成的仿射空间内。

# Restrictions of $\Sigma$
当数据量不足以拟合整个协方差矩阵时，可以限制协方差矩阵是对角矩阵，从而使协方差矩阵满秩。可以证明，此时 MLE 拟合的协方差矩阵满足：
$$\Sigma_{jj} = \frac{1}{m} \sum_{i = 1}^m (x^{(i)}_j - \mu_j)^2$$
此时，高斯分布的轮廓为椭球，且椭球的轴与坐标轴平行。

也可以进一步限制协方差矩阵的对角线上的元素相等，即 $\Sigma = \sigma^2 I$。可以证明，此时 MLE 拟合的协方差矩阵满足：
$$\sigma^2 = \frac{1}{mn} \sum_{j = 1}^n \sum_{i = 1}^m (x^{(i)}_j - \mu_j)^2$$
此时，高斯分布的轮廓为球。

上述两种方法中，协方差矩阵都是对角矩阵，因此都假设数据的各个维度之间两两无关且两两独立。而有时我们希望捕获各个维度之间的相关性。

# Marginals and Conditionals of Gaussians
对于向量随机变量：
$$x = \begin{bmatrix}
x_1\\
x_2
\end{bmatrix}$$
假设 $x \sim \mathcal{N}(\mu, \Sigma)$：
$$\mu = \begin{bmatrix}
\mu_1\\
\mu_2
\end{bmatrix}, \ \Sigma = \begin{bmatrix}
\Sigma_{11} & \Sigma_{12}\\
\Sigma_{21} & \Sigma_{22}
\end{bmatrix}$$

则协方差矩阵满足：
$$\begin{align}
\Sigma 
& = E[(x - \mu)(x - \mu)^T]\\
& = E\left[
\begin{pmatrix}
x_1 - \mu_1\\
x_2 - \mu_2
\end{pmatrix}
\begin{pmatrix}
x_1 - \mu_1\\
x_2 - \mu_2
\end{pmatrix}^T
\right]\\
& = E\left[
\begin{matrix}
(x_1 - \mu_1)(x_1 - \mu_1)^T & (x_1 - \mu_1)(x_2 - \mu_2)^T\\
(x_2 - \mu_2)(x_1 - \mu_1)^T & (x_1 - \mu_1)(x_2 - \mu_2)^T
\end{matrix}
\right]
\end{align}$$

由边缘概率公式，求得**边缘分布**：
$$x_1 \sim \mathcal{N}(\mu_1, \Sigma_{11})$$

由条件概率公式，求得**条件分布**：
$$x_1 | x_2 \sim \mathcal{N}(\mu_{1|2}, \Sigma_{1|2})$$
$$\mu_{1|2} = \mu_1 + \Sigma_{12}\Sigma_{22}^{-1}(x_2 - \mu_2)$$
$$\Sigma_{1|2} = \Sigma_{11} - \Sigma_{12}\Sigma_{22}^{-1}$$

# The Factor Analysis Model
因子分析模型基于以下假设：
$$z \sim \mathcal{N}(0, I)$$
$$x | z \sim \mathcal{N}(\mu + \Lambda z, \Psi)$$
其参数为：
- $\mu \in \mathbb{R}^n$
- $\Lambda \in \mathbb{R}^{n \times k}$
- $\Psi \in \mathbb{R}^{n \times n}$ 是对角矩阵
即先生成服从多变量高斯分布的 k 维隐变量 $z$，然后通过计算 $\mu + \Lambda z$ 将 $z$ 映射到 $\mathbb{R}^n$ 的一个 k 维仿射空间中，最后添加协方差矩阵为对角矩阵 $\Psi$ 的高斯噪声，从而生成 $x$。

因子分析模型的假设也可以写成：
$$z \sim \mathcal{N}(0, I)$$
$$\epsilon \sim \mathcal{N}(0, \Psi)$$
$$x = \mu + \Lambda z + \epsilon$$
其中，$\epsilon$ 和 $z$ 独立。

根据假设，**联合分布 $p(x, z)$ 服从多变量高斯分布**：
$$\begin{bmatrix}
z\\
x
\end{bmatrix} \sim \mathcal{N}(\mu_{zx}, \Sigma_{zx})$$
由于：
$$\mu_{zx} = \begin{bmatrix}
\mu_z\\
\mu_x
\end{bmatrix}$$
$$\Sigma_{zx} = E\left[
\begin{matrix}
(z - \mu_z)(z - \mu_z)^T & (z - \mu_z)(x - \mu_x)^T\\
(x - \mu_x)(z - \mu_z)^T & (x - \mu_x)(x_ - \mu_x)^T
\end{matrix}
\right]$$
可以证明：
$$\mu_{zx} = \begin{bmatrix}
\vec 0\\
\mu
\end{bmatrix}$$
$$\Sigma = \begin{bmatrix}
I & \Lambda^T\\
\Lambda & \Lambda \Lambda^T + \Psi
\end{bmatrix}$$
边缘分布：
$$x \sim \mathcal{N}(\mu, \Lambda \Lambda^T + \Psi)$$

# EM for Factor Analysis
MLE 最大化边缘概率的对数似然函数：
$$l(\theta) = \sum_{i = 1}^m \log p(x^{(i)}; \mu, \Lambda, \Psi)$$
十分复杂，无法通过让导数等于 0 求得其闭形解。

**EM 算法**可以优化因子分析模型：
1. E-step：对 $i = 1, 2, \cdots, m$，计算条件概率
$$Q_i(z^{(i)}) := p(z^{(i)}| x^{(i)}; \mu, \Sigma, \Psi)$$
$$z^{(i)} | x^{(i)} \sim \mathcal{N}(\mu_{z^{(i)} | x^{(i)}}, \Sigma_{z^{(i)} | x^{(i)}})$$
$$\mu_{z^{(i)} | x^{(i)}} = \Lambda^T (\Lambda \Lambda^T + \Psi)^{-1} (x^{(i)} - \mu)$$
$$\Sigma_{z^{(i)} | x^{(i)}} = I - \Lambda^T (\Lambda \Lambda^T + \Psi)^{-1} \Lambda$$
2. M-step：更新参数
TODO





