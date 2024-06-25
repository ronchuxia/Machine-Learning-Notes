# Cross Validation
以验证集经验风险作为对泛化风险的估计。选择使验证集经验风险最小的模型。

根据数据集的大小，可以选择以下几种交叉验证方式：
- cross validation
- k-fold cross validation
- leave-one-out cross validation

在使用交叉验证完成模型选择后，可以再用全部的数据集训练一遍模型。

# Feature Selection
使经验风险趋向于泛化风险的数据集大小与假设空间的 VC 维成正比。特征维度越大，VC 维也越大。因此，当特征维度较大，而数据集较小时，容易发生过拟合。我们可以使用特征选择减少特征维度，以防止过拟合。

如果特征有 n 维，则有 $2^n$ 个可能的特征子集。一一尝试的开销较大。我们可以使用启发式算法寻找一个较好的特征子集。

## Wrapper model feature selection
下面两个算法都属于 wrapper model feature selection。它们的缺点是需要多次调用学习算法，开销较大。

**Forward search**
1. Initialize $F = \emptyset$. 
2. Repeat { 
	1. For $i = 1, \cdots, n$ if $i \not\in F$, let $F_i = F \cup \{i\}$, and use some version of cross validation to evaluate features $F_i$ . (I.e., train your learning algorithm using only the features in $F_i$ , and estimate its generalization error.) 
	2. Set F to be the best feature subset found on step 2.1. 
	} 
3. Select and output the best feature subset that was evaluated during the entire search procedure.

**Backward search**
与 forward search 类似，但 F 初始化为所有特征，每次删除其中一个特征，并验证其效果。

## Filter feature selection
对于特征 $x_i$，可以计算其与输出 $y$ 的相关性 $S(i)$。保留相关性最大的几个特征。

$S(i)$ 可以取：
- $x_i$ 与 $y$ 的相关系数的绝对值
- $x_i$ 与 $y$ 的互信息

**互信息 (mutual information)**：
$$MI(x_i , y) = \sum_{x_i} \sum_y p(x_i , y) \log \frac{p(x_i, y)}{p(x_i)p(y)}$$
互信息也可以写成 KL 散度的形式：
$$MI(x_i, y) = KL(p(x_i, y)||p(x_i)p(y))$$
它描述了概率分布 $p(x_i, y)$ 与概率分布 $p(x_i)p(y)$ 之间的差距。

# Bayesian Statistics and Regularization
贝叶斯统计学将数据和参数都看作随机变量。数据是可观测的随机变量，参数是不可观测的随机变量。因此，参数不再是固定的值，而是服从一个分布。在对数据进行观测前，我们人为地为参数赋予一个**先验分布** $p(\theta)$。在对数据进行观测后，我们根据数据对先验分布进行修正，得到参数的**后验分布** $p(\theta|x, y)$。

贝叶斯推断利用贝叶斯公式对先验分布进行修正，得到后验分布：
$$\begin{align}
p(\theta | S) & = \frac{p(S | \theta) p(\theta)}{p(S)}\\ 
& = \frac{\prod_{i = 1}^m p(x^{(i)}, y^{(i)} | \theta) p(\theta)}{\int_\theta (\prod_{i = 1}^m p(x^{(i)}, y^{(i)} | \theta))p(\theta) d\theta}\\
& = \frac{\prod_{i = 1}^m p(y^{(i)} | x^{(i)}, \theta) p(\theta)}{\int_\theta (\prod_{i = 1}^m p(y^{(i)} | x^{(i)}, \theta))p(\theta) d\theta}
\end{align}$$

预测时，根据参数的后验分布计算输出的后验分布：
$$p(y | x, S) = \int_\theta p(y | x, \theta) p(\theta | S) d\theta$$
输出的预测值为：
$$E[y | x, S] = \int_y y p(y | x, S) dy$$

贝叶斯推断的案例：
[贝叶斯统计 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/38553838)

贝叶斯推断需要保存整个后验分布，并在推理时对后验分布进行积分以计算输出的后验分布。计算积分的开销较大。因此，**简化的贝叶斯推断只保存后验分布的众数，作为对参数值的估计**：
$$\begin{align}
\theta_{\text{MAP}} & = \arg\max_\theta p(\theta|S)\\
& = \arg\max_\theta p(y^{(i)} | x^{(i)}, \theta) p(\theta)
\end{align}$$
这种估计参数的方法称为**最大后验估计** (MAP, maximum a posteriori estimation)。

先验分布的作用就是对参数进行正则化：[[ps2]]
