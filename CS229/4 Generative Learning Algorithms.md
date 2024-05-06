# Generative Learning Algorithms
- Discriminative Learning Algorithms
	- 学习： $p(y|x)$
	- 最大化**条件概率**： $l(\theta) = \sum_{i = 1}^m \log p(y^{(i)} | x^{(i)})$ 
- Generative Learning Algorithms
	- 学习： $p(x|y)$ 和 $p(y)$ (class prior)
	- 最大化**联合概率**： $l(\theta) = \sum_{i = 1}^m \log p(x^{(i)}, y^{(i)})$
	- 根据贝叶斯公式
	$$p(y|x) = \frac{p(x|y)p(y)}{p(x)}$$
	- 实际测试时
	$$\begin{align}
	\arg\max_y p(y|x) & = \arg\max_y  \frac{p(x|y)p(y)}{p(x)}\\
	& = \arg\max_y p(x|y)p(y)
	\end{align}$$

# Multivariate Normal Distribution
$$p(x; \mu, \Sigma) = \frac{1}{(2\pi)^{1/2} |\Sigma|^{1/2}} \exp(-\frac{1}{2}(x - \mu)^T \Sigma^{-1} (x-\mu))$$

# Gaussian Discriminant Analysis
GDA 用于输入 $x$ 为**连续实数**，输出 $y$ 为 \{0, 1\} 的**二分类**任务。

GDA 基于以下假设：
$$y \sim \mathrm{Bernoulli}(\phi)$$
$$x|y = 0 \sim \mathcal{N}(\mu_0, \Sigma)$$
$$x|y = 1 \sim \mathcal{N}(\mu_1, \Sigma)$$
即对于给定的类别$y^{(i)}$，$x^{(i)}$ 服从高斯分布。
两个高斯分布使用相同的协方差矩阵，是为了使两个高斯的形状和方向相同，从而使**决策边界呈线性**。

最大化联合概率的对数似然函数：
$$
\begin{align}
l(\phi, \mu_0, \mu_1, \Sigma) & = \log \prod_{i = 1}^m p(x^{(i)}, y^{(i)}; \phi, \mu_0, \mu_1, \Sigma)\\
& = \log \prod_{i = 1}^m p(x^{(i)} | y^{(i)}; \mu_0, \mu_1, \Sigma) p(y^{(i)}; \phi)
\end{align}
$$
TODO：为什么用联合概率，而不用条件概率？

可以证明，当满足：
$$\phi = \frac{1}{m}\sum_{i = 1}^m 1\{y^{(i)} = 1\}$$
$$\mu_0 = \frac{\sum_{i = 1}^m 1\{y^{(i)} = 0\} x^{(i)}}{\sum_{i = 1}^m 1\{y^{(i)} = 0\}}$$
$$\mu_1 = \frac{\sum_{i = 1}^m 1\{y^{(i)} = 1\} x^{(i)}}{\sum_{i = 1}^m 1\{y^{(i)} = 1\}}$$
$$\Sigma = \frac{1}{m} \sum_{i = 1}^m (x^{(i)} - \mu_{y^{(i)}}) (x^{(i)} - \mu_{y^{(i)}})^T$$
时，梯度等于 0，联合概率的对数似然函数取最大值。

GDA 的拟合非常易于计算，只需要统计训练数据集的均值和协方差即可。

# GDA and Logistic Regression
将 
$$p(y = 1|x; \phi, \mu_0, \mu_1, \Sigma) = \frac{p(x | y = 1) p(y = 1)}{p(x)}$$
看作自变量为 $x$ 的函数，可以证明：
$$p(y = 1|x; \phi, \mu_0, \mu_1, \Sigma) = \frac{1}{1 + \exp(- \theta^T x)}$$
其中，$\theta$ 是 $\phi, \mu_0, \mu_1, \Sigma$ 的函数。
**决策边界**：$\theta^T x = 0$。

注：若两个高斯分布的协方差矩阵不同，则输出将不满足上述形式，决策边界也不是线性的。

GDA 和 logistic 回归的输出的形式虽然一样（都是 logistic 函数），但两者对数据的假设不一样，因此习得的**决策边界也不一样**。GDA 对数据有更强的假设。
- $p(x|y)$ 服从高斯分布 $\Rightarrow$ $p(y|x)$ 为 logistic 函数
- $p(y|x)$ 为 logistic 函数 $\nRightarrow$ $p(x|y)$ 服从高斯分布（也可能服从其他指数族分布，如泊松分布）

使用场景：
- 数据服从或近似服从高斯分布，或数据量较少时，使用 GDA
- 数据不服从高斯分布，或数据量较大时，使用 logistic 回归
在数据量较小的情况下，使用更多的假设可以增强模型的效果。

注：**Box-Cox 变换**可以改善数据的正态性。

# Naive Bayes
朴素贝叶斯用于输入 $x$ 为**离散值**的**二分类**任务。
## Multivariate Bernoulli Event Model
对于给定的词表（词表长度为 n）和一封邮件，$x_j \in \{0, 1\}$ 表示词表中的词 j 是否在这封邮件中出现，$y \in \{0, 1\}$ 表示这封邮件是否是垃圾邮件：
$$x = \begin{bmatrix}
x_1\\
x_2\\
\vdots\\
x_n
\end{bmatrix}$$
则 $p(x|y)$ 服从分类分布。由于 $x$ 共有 $2^n$ 种取值，该分类分布的参数量为 $2^n - 1$，参数量随输入维度的增加呈指数级增长。

为了减少参数量，朴素贝叶斯假设输入的各个维度**条件独立**。
$$
\begin{align}
p(x|y) & = p(x_1, x_2, x_3\cdots, x_n | y)\\
& = p(x_1|y) p(x_2|y, x_1) p(x_3|y, x_1, x_2) \cdots p(x_n|y, x_1, x_2, x_3, \cdots x_{n-1})\\
& = p(x_1|y) p(x_2|y) p(x_3|y) \cdots p(x_n|y)
\end{align}
$$

朴素贝叶斯假设非常强，但是在很多问题上效果很好。

输入 $x \in \{0, 1\}^n$，输出 $y \in \{0, 1\}$ 的朴素贝叶斯基于以下假设：
$$y \sim \mathrm{Bernoulli(\phi_y)}$$
$$x_i | y = 1 \sim \mathrm{Bernoulli}(\phi_{i|y=1})$$
$$x_i | y = 0 \sim \mathrm{Bernoulli}(\phi_{i|y=0})$$
其参数为：
- $\phi_y = p(y = 1)$
- $\phi_{i|y = 1} = p(x_i = 1 | y = 1)$
- $\phi_{i|y = 0} = p(x_i = 1 | y = 0)$
输入的每个变量都各自服从一个伯努利分布，因此称为 Multivariate Bernoulli Event Model。

最大化联合概率的对数似然函数：
$$
\begin{align}
l(\theta) & = \log \prod_{i = 1}^m p(x^{(i)}, y^{(i)})\\
& = \log \prod_{i = 1}^m p(y^{(i)}) \prod_{j = 1}^n p(x_j^{(i)} | y^{(i)})

\end{align}
$$

可以证明，当参数满足：
$$\phi_y = \frac{\sum_{i = 1}^m 1\{y^{(i)} = 1\}}{m}$$
$$\phi_{i|y = 1} = \frac{\sum_{i = 1}^m 1\{x^{(i)}_j = 1 \land y^{(i)} = 1\}}{\sum_{i = 1}^m 1\{y^{(i)} = 1\}}$$
$$\phi_{i|y = 0} = \frac{\sum_{i = 1}^m 1\{x^{(i)}_j = 1 \land y^{(i)} = 0\}}{\sum_{i = 1}^m 1\{y^{(i)} = 0\}}$$
时，联合概率的对数似然函数取最大值。

可以将每个变量各自服从一个二项分布扩展为每个变量各自服从一个分类分布。
对于输入为连续值，且不服从高斯分布的情况，可以将其**离散化**后使用朴素贝叶斯分类，从而达到比 GDA 更好的分类效果。

**Laplace Smoothing**
如果词表中的词 $j$ 在训练集中没有出现，则：
$$\phi_{j|y = 0} = 0, \ \phi_{j|y = 1} = 0$$
此时，如果收到一封含有词 $j$ 的邮件，朴素贝叶斯计算的条件概率：
$$p(y = 1 | x) = \frac{\prod_{i = 1}^m p(x_i | y = 1) p(y = 1)}{\prod_{i = 1}^m p(x_i | y = 1) p(y = 1) + \prod_{i = 1}^m p(x_i | y = 0) p(y = 0)} = \frac{0}{0}$$
 会发生**数值错误**。

若随机变量 $z \in \{1, 2, \cdots, k\}$ 服从参数为 $\phi_i = p(z = i)$ 分类分布，m 次独立观测的观测值为 $\{z^{(1)}, z^{(2)}, \cdots, z^{(m)}\}$，则根据最大似然估计：
$$\phi_j = \frac{\sum_{i = 1}^m 1\{z^{(i)} = j\}}{m}$$
然而，从统计学角度而言，**将一个在有限的数据集中未发生的事件的概率估计为 0 是不合理的**。因此，对上述估计值做拉普拉斯平滑：
$$\phi_j = \frac{\sum_{i = 1}^m 1\{z^{(i)} = j\} + 1}{m + k}$$
注：平滑后的概率仍满足 $\sum_{i = 1}^k \phi_i = 1$。

对 Multivariate Bernoulli Event Model 的参数做拉普拉斯平滑：
$$\phi_y = \frac{\sum_{i = 1}^m 1\{y^{(i)} = 1\}}{m}$$
$$\phi_{i|y = 1} = \frac{\sum_{i = 1}^m 1\{x^{(i)}_j = 1 \land y^{(i)} = 1\} + 1}{\sum_{i = 1}^m 1\{y^{(i)} = 1\} + 2}$$
$$\phi_{i|y = 0} = \frac{\sum_{i = 1}^m 1\{x^{(i)}_j = 1 \land y^{(i)} = 0\} + 1}{\sum_{i = 1}^m 1\{y^{(i)} = 0\} + 2}$$

## Multinomial Event Model
对于给定的词表（词表长度为 k）和一封邮件（邮件长度为 n），$x_i = j$ 表示邮件中的第 i 个词是词表中的词 j，$y \in \{0, 1\}$ 表示这封邮件是否是垃圾邮件：
$$x = \begin{bmatrix}
x_1\\
x_2\\
\vdots\\
x_n
\end{bmatrix}$$

输入 $x \in \{1, 2, \cdots, |V|\}^n$，输出 $y \in \{0, 1\}$ 的朴素贝叶斯基于以下假设：
$$y \sim \mathrm{Bernoulli}(\phi_y)$$
$$x_i | y = 1 \sim \mathrm{Categorical}(\phi_{1|y=1}, \phi_{2|y=1}, \cdots, \phi_{|V||y=1})$$
$$x_i | y = 0 \sim \mathrm{Categorical}(\phi_{1|y=0}, \phi_{2|y=0}, \cdots, \phi_{|V||y=0})$$
其参数为：
- $\phi_y = p(y = 1)$
- $\phi_{k|y = 1} = p(x_i = k | y = 1), \ i \in \{1, 2, \cdots, n\}$
- $\phi_{k|y = 0} = p(x_i = k | y = 0), \ i \in \{1, 2, \cdots, n\}$
输入服从多项式分布，因此称为 Multinomial Event Model。

可以证明，当参数满足：
$$\phi_y = \frac{\sum_{i = 1}^m 1\{y^{(i)} = 1\}}{m}$$
$$\phi_{k|y = 1} = \frac{\sum_{i = 1}^m \sum_{j = 1}^n 1\{x^{(i)}_j = k \land y^{(i)} = 1\}}{\sum_{i = 1}^m 1\{y^{(i)} = 1\} n_i}$$
$$\phi_{k|y = 0} = \frac{\sum_{i = 1}^m \sum_{j = 1}^n 1\{x^{(i)}_j = k \land y^{(i)} = 0\}}{\sum_{i = 1}^m 1\{y^{(i)} = 0\} n_i}$$
时，联合概率的对数似然函数取最大值。

**Laplace Smoothing**
对 Multinomial Event Model 的参数做拉普拉斯平滑：

$$\phi_y = \frac{\sum_{i = 1}^m 1\{y^{(i)} = 1\}}{m}$$
$$\phi_{k|y = 1} = \frac{\sum_{i = 1}^m \sum_{j = 1}^n 1\{x^{(i)}_j = k \land y^{(i)} = 1\} + 1}{\sum_{i = 1}^m 1\{y^{(i)} = 1\} n_i + |V|}$$
$$\phi_{k|y = 0} = \frac{\sum_{i = 1}^m \sum_{j = 1}^n 1\{x^{(i)}_j = k \land y^{(i)} = 0\} + 1}{\sum_{i = 1}^m 1\{y^{(i)} = 0\} n_i + |V|}$$



















