集成学习通过减小 variance 或减小 bias 提高模型的预测准确率。

**减小 variance**
对于随机变量 $X_i$，$0 \leq i < n$，方差 $Var(X_i) = \sigma^2$，相关系数 $\rho$。均值的方差：
$$\begin{align}
Var(\overline X) & = Var(\frac{1}{n} \sum_{i = 1}^n X_i)\\
& = \frac{1}{n^2} Var(\sum_{i = 1}^n X_i)\\
& = \frac{1}{n^2} (\sum_{i = 1}^n Var(X_i) + \sum_{i \neq j} Cov(X_i, X_j))\\
& = \frac{\sigma^2}{n} + \frac{n(n - 1)}{n^2} \rho \sigma^2\\
& = \rho \sigma^2 + \frac{1 - \rho}{n} \sigma^2
\end{align}$$

如果 $X_i$ 表示模型 i 的损失，$\overline X$ 表示集成模型的损失，则有两种方法减小集成模型的 variance：
- 增加模型的个数 $n$
- 减小模型的相关系数 $\rho$
	- 使用不同的模型
	- 使用不同的训练集
	- Bagging 

**减小 bias**
- boosting

# Bagging
Bagging = bootstrap aggregation
训练集为 $S$，真实的数据分布为 $P$。假设 $S = P$，从 $S$ 中**有放回**地采样多个子数据集 $Z$（$Z \sim S$，$|Z| = |S|$），在每个子数据集上训练一个模型，最终的预测结果由所有模型的预测结果做平均或投票得到。

由于每次训练只使用了部分的数据集，bias 会增加，但 variance 的减小超过了 bias 的增加。

Bagging 可以用于决策树，以减小决策树的 variance。**随机森林**还要求每个决策树都只使用部分的特征，从而进一步减小模型的相关系数。由于只使用了部分的特征，bias 会增加，但 variance 的减小超过了 bias 的增加。

# Boosting
通过顺序地训练多个模型，每个模型在前一个模型的基础上进行改进。每次训练时，重点关注被前一个模型错误分类的样本，从而逐步提高整体模型的性能。在直接构造强学习器较为困难的情况下，为学习算法提供了一种有效的新思路和新方法。
## Forward Stagewise Additive Modeling
在每一轮训练中，通过拟合残差来逐步提升模型的预测能力。

```
Input: Labeled training data (x_1, y_1), (x_2, y_2), ..., (x_N , y_N) 
Output: Ensemble classifer f(x)

Initialize f_0(x) = 0 
for m = 0 to M do
	Compute (β_m, γ_m) = argmin_{β, γ}(sum(L(y_i, f_{m−1}(x_i) + β * G(x_i; γ))))
	Set f_m(x) = f_{m−1}(x) + β_m * G(x; γ_i)
end
f(x) = f_m(x)
```
### AdaBoost
Forward stagewise additive modeling 的一个特例，用于二分类。
用 sign 引入非线性。
TODO
### Gradient Boosting
Forward stagewise additive modeling 的一种实现方式，使用类似梯度下降的方式减小损失，拟合残差。
TODO








