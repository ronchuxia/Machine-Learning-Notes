# Exponential Family
指数族分布是一组分布，其 pdf 可以写成：
$$p(y;\eta) = b(y) \exp(\eta^T T(y) - a(\eta))$$
- $\eta$ : natural parameter，标量或向量，是 canonical parameters 的函数
- $T(y)$ : sufficient statistic，标量或向量，通常 $T(y) = y$
- $a(\eta)$ : log partition function，标量，用于归一化 pdf
- $b(y)$ : base measure，标量

指数族分布的性质：
- $L(\eta)$ （负对数似然函数）是凸函数
- $E[T(y);\eta] = \frac{\partial}{\partial \eta}a(\eta)$
	- 如果 $\eta$ 是向量，$E[T(y);\eta] = \nabla_\eta a(\eta)$
- $Var[T(y);\eta] = \frac{\partial^2}{\partial \eta^2} a(\eta)$
	- 如果 $\eta$ 是向量，$Var[T(y);\eta] = H(a(\eta))$

指数族分布的优点：计算 $E[y;\eta]$ 和 $Var[y;\eta]$ 不需要积分，只需要微分

以下分布都属于指数族分布：
- Binary: Bernoulli, ...
- Real: Gaussian, ...
- Count: Poisson, ...
- Non-negative Real: Gamma, Exponential, ...
- Distribution: Beta, Dirichlet, ...

## Bernoulli Distribution
$$
\begin{align}
p(y;\phi) & = \phi^y (1-\phi)^{(1-y)}\\
& = \exp(\log(\phi^y (1-\phi)^{(1-y)}))\\
& = \exp((\log\frac{\phi}{1-\phi})y + \log(1-\phi))
\end{align}
$$
- $\eta = \log\frac{\phi}{1-\phi} \Rightarrow \phi = \frac{1}{1+e^{-\eta}}$
- $T(y) = y$
- $a(\eta) = -\log(1-\phi) = \log(1+e^\eta)$
- $b(y) = 1$

## Gaussian Distribution
在线性回归中，$\sigma$ 对 $\theta$ 和 $h_\theta(x)$ 没有影响，因此不妨设 $\sigma = 1$：
$$
\begin{align}
p(y;\mu) & = \frac{1}{\sqrt{2\pi}} \exp(-\frac{(y - \mu)^2}{2})\\
& = \frac{1}{\sqrt{2\pi}} \exp(-\frac{y^2}{2}) \exp(\mu y - \frac{\mu^2}{2})
\end{align}
$$
- $\eta = \mu$
- $T(y) = y$
- $a(\eta) = \frac{\mu^2}{2} = \frac{\eta^2}{2}$
- $b(y) = \exp(-\frac{y^2}{2})$

对于一般的高斯分布，也可以将其写成指数族分布的形式。此时 $\eta$ 和 $T(y)$ 都是二维向量。可以验证其期望和方差的性质。

# Generalized Linear Models
广义线性模型基于以下假设：
- 对于任意的输入 $x$，输出 $y$ 服从指数族分布：$y | x ; \theta \sim Exponential \ Family(\eta)$ 
- 对于任意的输入 $x$，$h_\theta(x)$ 等于 $T(y)$ 的期望：$h_\theta(x) = E[T(y)|x;\theta]$
	- 如果 $T(y) = y$，$h_\theta(x) = E[y|x;\theta]$
- $\eta$ 与 $x$ 线性相关：$\eta = \theta^T x$

对于任意的输入 $x$，可以由 $h_\theta(x)$ 得到 $T(y)$，再由 $T(y)$ 得到 $y$ 。

对于线性回归：
$$h_\theta(x) = E[y|x;\theta] = \mu = \eta = \theta^T x$$

对于 logistic 回归：
$$h_\theta(x) = E[y|x;\theta] = \phi = \frac{1}{1 + e^{-\eta}} = \frac{1}{1 + e^{-\theta^Tx}}$$

GLM 的参数更新规则：
$$
\begin{align}
\theta_j & := \theta_j + \alpha \frac{\partial}{\partial \theta_j} l(\theta)\\
& := \theta_j + \alpha \frac{\partial}{\partial \theta_j} \sum_{i = 1}^m \log(p(y^{(i)}|x^{(i)};\theta))\\
& := \theta_j + \alpha \sum_{i = 1}^m \frac{\partial}{\partial \theta_j}  (\log b(y^{(i)}) + \eta^T T(y^{(i)}) - a(\eta))\\
& := \theta_j + \alpha \sum_{i = 1}^m \frac{\partial}{\partial \eta}  (\log b(y^{(i)}) + \eta^T T(y^{(i)}) - a(\eta)) \frac{\partial}{\partial \theta_j}\eta\\
& := \theta_j + \alpha \sum_{i = 1}^m (T(y^{(i)}) - \frac{\partial}{\partial \eta}a(\eta)) x^{(i)}_j\\
& := \theta_j + \alpha \sum_{i = 1}^m (T(y^{(i)}) - E[T(y);\eta]) x^{(i)}_j\\
& := \theta_j + \alpha \sum_{i = 1}^m (T(y^{(i)}) - h_\theta(x^{(i)}))x^{(i)}_j
\end{align}
$$
如果 $T(y) = y$：
$$\theta_j := \theta_j + \alpha \sum_{i = 1}^m (y^{(i)} - h_\theta(x^{(i)}))x^{(i)}_j$$

Notations:
- $g(\eta) = E[T(y)|x;\eta]$ : canonical **response function**, maps natural parameters to canonical parameters 
- $g^{-1}$ : canonical **link function**, maps canonical parameters to natural parameters

可以验证：
- 线性回归和 logistic 回归的 $h_\theta(x) = E[T(y)|x;\theta] = \frac{\partial}{\partial \eta}a(\eta)$
- 线性回归和 logistic 回归的 $\theta_j := \theta_j + \alpha \sum_{i = 1}^m (y^{(i)} - h_\theta(x^{(i)}))x^{(i)}_j$

# Softmax Regression
Softmax 回归基于以下假设：
$$p(y^{(i)}|x^{(i)}, \theta) = \phi_1^{1\{y^{(i)} = 1\}} \phi_2^{1\{y^{(i)} = 2\}} \cdots \phi_k^{1\{y^{(i)} = k\}}$$
即对于给定的点 $x^{(i)}$，$y^{(i)}$ 服从**分类分布** (Categorical Distribution)。

下面我们证明，**分类分布属于指数分布族**。
首先，定义 $T(y)$。与线性回归和 logistic 回归不同，$T(y) \neq y$：
$$T(1) = \begin{bmatrix}1\\0\\0\\ \vdots \\0\end{bmatrix}, \ T(2) = \begin{bmatrix}0\\1\\0\\ \vdots \\0\end{bmatrix}, \cdots, \ T(k - 1) = \begin{bmatrix}0\\0\\0\\ \vdots \\1\end{bmatrix}, \ T(k) = \begin{bmatrix}0\\0\\0\\ \vdots \\0\end{bmatrix}$$
令 $(T(y))_i$ 表示 $T(y)$ 的第 i 个元素，则有：
$$(T(y))_i = 1\{y = i\}$$
$$\mathrm{E}((T(y))_i) = P(y = i) = \phi_i$$
因此：
$$
\begin{align}
p(y^{(i)}|x^{(i)}, \theta) & = \phi_1^{1\{y^{(i)} = 1\}} \phi_2^{1\{y^{(i)} = 2\}} \cdots \phi_k^{1\{y^{(i)} = k\}}\\
& = \phi_1^{1\{y^{(i)} = 1\}} \phi_2^{1\{y^{(i)} = 2\}} \cdots \phi_k^{1 - \sum_{j = 1}^{k - 1}1\{y^{(i)} = j\}}\\
& = \phi_1^{(T(y))_1} \phi_2^{(T(y))_2} \cdots \phi_k^{1 - \sum_{j = 1}^{k - 1}1\{(T(y))_j\}}\\
& = \exp[(T(y))_1 \log \phi_1 + (T(y))_2 \log \phi_2 + \cdots + (1 - \sum_{j = 1}^{k - 1}1\{(T(y))_j\}) \log \phi_k]\\
& = \exp[(T(y))_1 \log(\phi_1/\phi_k) + (T(y))_2 \log(\phi_2/\phi_k) + \cdots + \log \phi_k]
\end{align}
$$
- $\eta = \begin{bmatrix} \log(\phi_1/\phi_k) \\ \log(\phi_{2}/\phi_k) \\ \vdots \\ \log(\phi_{k - 1}/\phi_k) \end{bmatrix}$
- $a(\eta) = -\log\phi_k$
- $b(y) = 1$

Link function :
$$\eta_i = \log \frac{\phi_i}{\phi_k}$$

求 link function 的反函数可以得到 response function :
$$\phi_i = \phi_k \exp(\eta_i)$$
因此：
$$\sum_{i = 1}^k \phi_i = \sum_{i = 1}^k \phi_k \exp(\eta_i) = \phi_k \sum_{i = 1}^k \exp(\eta_i) = 1$$
因此：
$$\phi_k = \frac{1}{\sum_{i = 1}^k \exp(\eta_i)}$$
Response function :
$$\phi_i = \frac{\exp(\eta_i)}{\sum_{j = 1}^k \exp(\eta_j)}$$

假设 $\eta = \theta^T x$，对参数 $\theta$ 做最大似然估计：
$$l(\theta) = \sum_{i = 1}^m \log \prod_{l = 1}^k (\frac{e^{\theta_l^T x^{(i)}}}{\sum_{j = 1}^k e^{\theta_j^T x^{(i)}}})^{1\{y^{(i)} = l\}}$$
可以验证：
$$\frac{\partial l(\theta)}{\partial \theta_{mn}} = (T(y^{(i)})_m - h(x)_m) x_n^{(i)}$$
与 GLM 的参数更新规则一致。



