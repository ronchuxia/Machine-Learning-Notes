# Linear Classifiers (Logistic Regression and GDA)
1. 证明逻辑回归的损失函数为凸函数：
$$\begin{align}
H_{jk} & = \frac{\partial^2}{\partial \theta_j \partial \theta_k} J(\theta) \\ 
& = \frac{\partial}{\partial \theta_k} (- \frac{1}{m} \sum_{i = 1}^m (y^{(i)} - h_\theta(x^{(i)}))x^{(i)}_j)\\
& = \frac{1}{m} \sum_{i = 1}^m h_\theta(x^{(i)}) (1 - h_\theta(x^{(i)})) x_j^{(i)} x_k^{(i)}
\end{align}$$
$$\begin{align}
z^T H z & = \sum_{j = 1}^m \sum_{k = 1}^m z_j H_{jk} z_k\\
& = \frac{1}{m} \sum_{j = 1}^m \sum_{k = 1}^m \sum_{i = 1}^m h_\theta(x^{(i)}) (1 - h_\theta(x^{(i)})) x_j^{(i)} x_k^{(i)} z_j z_k\\
& = \frac{1}{m} \sum_{i = 1}^m h_\theta(x^{(i)}) (1 - h_\theta(x^{(i)})) \sum_{j = 1}^m \sum_{k = 1}^m  x_j^{(i)} x_k^{(i)} z_j z_k\\
& = \frac{1}{m} \sum_{i = 1}^m h_\theta(x^{(i)}) (1 - h_\theta(x^{(i)})) \sum_{j = 1}^m x^{(i)}_j z_j \sum_{k = 1}^m x^{(i)}_k z_k\\
& = \frac{1}{m} \sum_{i = 1}^m h_\theta(x^{(i)}) (1 - h_\theta(x^{(i)})) (\sum_{j = 1}^m x^{(i)}_j z_j)^2\\
& = \frac{1}{m} \sum_{i = 1}^m h_\theta(x^{(i)}) (1 - h_\theta(x^{(i)})) ((x^{(i)})^T z)^2\\
& \geq 0
\end{align}$$
2. 牛顿法解 logistic 回归

3. 证明 GDA 的决策边界为线性：
$$\begin{align}
p(y = 1|x) &= \frac{p(x|y = 1)p(y = 1)}{p(x|y = 1)p(y = 1) + p(x|y = 0)p(y = 0)}\\
& = \frac{1}{1 + \frac{\phi}{1 - \phi} \exp(-(\mu_0 - \mu_1)^T \Sigma^{-1} x + \frac{1}{2}(\mu_0^T \Sigma^{-1} \mu_0 + \mu_1^T \Sigma^{-1} \mu_1))}\\
& = \frac{1}{1 + \exp(-(\mu_0 - \mu_1)^T \Sigma^{-1} x + \frac{1}{2} \mu_0^T \Sigma^{-1} \mu_0 + \frac{1}{2} \mu_1^T \Sigma^{-1} \mu_1 + \log \frac{\phi}{1 - \phi})}\\
& = \frac{1}{1 + \exp(-(\theta^T x + \theta_0))}
\end{align}$$
4. 证明 GDA 的最优解
5. 实现 GDA
注：根据决策边界即可进行预测，不需要计算概率值。

# Incomplete, Positive-Only Labels
在某些场景中，我们只知道部分正样本的 label，即只有部分正样本的 label 为 1，其余样本的 label 都为 0。
- $y$ ：上述场景下样本的 label
- $t$ ：实际场景下样本的 label
在训练集 $\{(x^{(i)}, y^{(i)}): i = 1, 2, \cdots, m\}$ 上训练 logistic 回归，模型的输出为 $p(y = 1 | x)$，这与我们想要的输出 $p(t = 1 | x)$ 存在偏差。

1. 假设 $y^{(i)}$ 与 $x^{(i)}$ 在给定 $t^{(i)}$ 的情况下条件独立：
$$p(y^{(i)} = 1 | t^{(i)} = 1, x) = p(y^{(i)} = 1 | t^{(i)} = 1)$$
这等价于 $y = 1$ 的正样本是从所有正样本中均匀随机选择的。

证明某个样本 $y = 1$ 的概率是该样本 $t = 1$ 的概率的常数倍：
$$\begin{align}
p(y^{(i)} = 1 | x^{(i)}) & = \frac{p(y^{(i)} = 1, x^{(i)})}{p(x^{(i)})}\\
& = \frac{p(y^{(i)} = 1, t^{(i)} = 1, x^{(i)})}{p(x^{(i)})}\\
& = p(y^{(i)} = 1 | t^{(i)} = 1, x^{(i)}) \frac{p(t^{(i)} = 1, x^{(i)})}{p(x^{(i)})}\\
& = p(y^{(i)} = 1 | t^{(i)} = 1) p(t^{(i)} = 1 | x^{(i)})
\end{align}$$
即：
$$\begin{align}
p(t^{(i)} = 1 | x^{(i)}) & = p(y^{(i)} = 1 | x^{(i)}) / p(y^{(i)} = 1 | t^{(i)} = 1)\\
& = p(y^{(i)} = 1 | x^{(i)}) / \alpha
\end{align}$$
只要知道 $\alpha$，我们就能对 $p(y^{(i)} = 1 | x^{(i)})$ 进行修正，从而得到 $p(t^{(i)} = 1 | x^{(i)})$。

2. 为了估计 $\alpha$ 的值，我们使用一个只含有正样本的验证集 $V_+$，假设：
$$h_\theta(x^{(i)}) \approx p(y^{(i)} = 1 | x^{(i)}), \ \forall x^{(i)} \in V_+$$
$$p(t^{(i)} = 1| x^{(i)}) \approx 1, \ \forall x^{(i)} \in V_+$$
则：
$$\alpha = \frac{p(y^{(i)} = 1 | x^{(i)})}{p(t^{(i)} = 1 | x^{(i)})} \approx h_\theta(x^{(i)}), \ \forall x^{(i)} \in V_+$$
因此，$\alpha$ 的估计值为：
$$\alpha \approx \frac{1}{|V_+|} \sum_{x^{(i)} \in V_+} h_\theta(x^{(i)})$$

# Poisson Regression
用 GLM 进行泊松回归。
1. 证明泊松分布属于指数组分布
$$\begin{align}
p(y; \lambda) & = \frac{e^{-\lambda} \lambda^y}{y!}\\
& = \frac{1}{y!} \exp(y \log\lambda - \lambda)\\
\end{align}$$
$$b(y) = \frac{1}{y!}$$
$$\eta = \log\lambda$$
$$T(y) = y$$
$$a(\eta) = e^\eta$$
2. Canonical response function
$$\begin{align}
g(\eta) & = E[T(y)|x;\eta]\\
& = E[y|x;\eta]\\
& = \lambda\\
& = e^\eta
\end{align}$$
3. Gradient ascent update rule
$$\begin{align}
\theta_j & := \theta_j + \alpha \sum_{i = 1}^m (y^{(i)} - h_\theta(x^{(i)}))x^{(i)}_j\\
& := \theta_j + \alpha \sum_{i = 1}^m (y^{(i)} - g(\eta))x^{(i)}_j\\
& := \theta_j + \alpha \sum_{i = 1}^m (y^{(i)} - e^{\theta^Tx^{(i)}})x^{(i)}_j\\
\end{align}$$
4. 实现泊松回归

# Convexity of Generalized Linear Models
1. 证明：
$$E[T(y);\eta] = \frac{\partial}{\partial \eta}a(\eta)$$

因为：
$$\int b(y) \exp(\eta T(y) - a(\eta)) dy = 1$$
所以：
$$a(\eta) = \log \int b(y) \exp(\eta T(y)) dy$$
所以：
$$\begin{align}
\frac{\partial}{\partial \eta}a(\eta) & = \frac{\partial}{\partial \eta} \log \int b(y) \exp(\eta T(y)) dy\\
& = \frac{1}{\int b(y) \exp(\eta T(y)) dy} \frac{\partial}{\partial \eta} \int b(y) \exp(\eta T(y)) dy\\
& = \frac{1}{\exp(a(\eta))} \int \frac{\partial}{\partial \eta} b(y) \exp(\eta T(y)) dy\\
& = \frac{1}{\exp(a(\eta))} \int b(y) \exp(\eta T(y)) T(y) dy\\
& = \int b(y) \exp(\eta T(y) - a(\eta)) T(y) dy\\
& = E[T(y);\eta]
\end{align}$$

2. 证明：
$$Var[T(y);\eta] = \frac{\partial^2}{\partial \eta^2} a(\eta)$$

$$\begin{align}
\frac{\partial^2}{\partial \eta^2} a(\eta) & = \frac{\partial}{\partial \eta} \int b(y) \exp(\eta T(y) - a(\eta)) T(y) dy\\
& = \int \frac{\partial}{\partial \eta} b(y) \exp(\eta T(y) - a(\eta)) T(y) dy\\
& = \int b(y) \exp(\eta T(y) - a(\eta)) T(y) (T(y) - \frac{\partial}{\partial \eta} a(\eta)) dy\\
& = \int b(y) \exp(\eta T(y) - a(\eta)) T(y) (T(y) - E[T(y)]) dy\\
& = \int b(y) \exp(\eta T(y) - a(\eta)) (T(y))^2 dy - \int b(y) \exp(\eta T(y) - a(\eta)) T(y) E[T(y)] dy\\
& = E[(T(y))^2] - (E[T(y)])^2\\
& = Var[T(y); \eta]
\end{align}$$

3. 证明 GLM 的 NLL 损失函数为凸函数：
$$\begin{align}
H_{jk} & = \frac{\partial^2}{\partial \theta_j \partial \theta_k} J(\theta) \\ 
& = \frac{\partial}{\partial \theta_k} (- \sum_{i = 1}^m (T(y^{(i)}) - \frac{\partial}{\partial \eta} a(\eta)) x^{(i)}_j)\\
& = \sum_{i = 1}^m \frac{\partial^2}{\partial \eta^2} a(\eta) x_j^{(i)} x_k^{(i)}\\
& = \sum_{i = 1}^m Var[T(y); \eta] x_j^{(i)} x_k^{(i)}
\end{align}$$
$$\begin{align}
z^T H z & = \sum_{j = 1}^m \sum_{k = 1}^m \sum_{i = 1}^m Var[T(y); \eta] x_j^{(i)} x_k^{(i)}z_j z_k\\
& = Var[T(y); \eta] \sum_{i = 1}^m \sum_{j = 1}^m \sum_{k = 1}^m x_j^{(i)} x_k^{(i)}z_j z_k\\
& = Var[T(y); \eta] \sum_{i = 1}^m (x^{(i)})^T z\\
& \geq 0
\end{align}$$

# Locally Weighted Linear Regression
1. Locally weighted linear regression 的损失函数：
$$J(\theta) = \frac{1}{2} \sum_{i = 1}^m w^{(i)}(h_\theta(x^{(i)}) - y^{(i)})^2$$
可以写成向量形式：
$$J(\theta) = (X\theta - y)^T W (X\theta - y)$$
$$W = \begin{bmatrix} 
\frac{1}{2} w^{(1)} & & &\\
& \frac{1}{2} w^{(1)} & &\\
& & \ddots &\\
& & & \frac{1}{2} w^{(m)}
\end{bmatrix}$$

2. 推导 locally weighted linear regression 的正则表达式：
$$\begin{align}
\nabla_\theta J(\theta) & = \nabla_\theta (X\theta - y)^T W (X\theta - y)\\
& = \nabla_\theta (\theta^T X^T W X \theta - 2 \theta^T X^T W y + y^T W y)\\
& = 2 X^T W X \theta - 2 X^T W y\\
& = 0
\end{align}$$
所以：
$$\theta = (X^T W X)^{-1} X^T W y$$

3. 证明 locally weighted linear regression 相当于根据预测点的位置调整每个输出的正态分布的方差：
$$\begin{align}
L(\theta) & = \sum_{i = 1}^m - \log p(y^{(i)} | x^{(i)}; \theta)\\
& = \sum_{i = 1}^m \log \sqrt{2\pi} \sigma^{(i)} + \sum_{i = 1}^m \frac{(y^{(i)} - \theta^T x^{(i)})^2}{2 (\sigma^{(i)})^2}\\
\end{align}$$
$$J(\theta) = \sum_{i = 1}^m \frac{(y^{(i)} - \theta^T x^{(i)})^2}{2 (\sigma^{(i)})^2}$$
$$w^{(i)} = \frac{1}{(\sigma^{(i)})^2}$$















