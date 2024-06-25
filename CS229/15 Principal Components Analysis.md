与 factor analysis 类似，PCA 可以找到数据大致所在的 k 维子空间。PCA 可以用于降维、去相关、压缩和降噪等用途。

在 PCA 之前，首先要对数据做归一化。对于均值不为 0 的数据，要将其均值归一化为 0。对于方差不为 1 的数据，要将其方差归一化为 1。

对于 k = 1 的情况，PCA 的做法是，找到单位向量 $u$，使得当数据投影到 $u$ 所在的方向后，投影数据的方差最大。

由于均值归一化为 0，投影数据的方差为：
$$\begin{align}
\frac{1}{m} \sum_{i = 1}^m ({x^{(i)}}^T u)^2 & = \frac{1}{m} \sum_{i = 1}^m u^T x^{(i)} {x^{(i)}}^T u\\
& = u^T \left( \frac{1}{m} \sum_{i = 1}^m x^{(i)} {x^{(i)}}^T \right) u\\
& = u^T \Sigma u
\end{align}$$

因此，PCA 可以写成：
$$\begin{align}
\max_{u} & \ u^T \Sigma u\\
\mathrm{s.t.} & \ ||u|| = 1
\end{align}$$

构造拉格朗日函数：
$$\mathcal{L}(u) = u^T \Sigma u - \lambda (u^T u - 1)$$
让导数等于 0：
$$\frac{\partial \mathcal{L}}{\partial u} = 2 \Sigma u - 2 \lambda u = 0$$
即 $u$ 是 $\Sigma$ 的特征向量：
$$\Sigma u = \lambda u$$
则：
$$u^T \Sigma u = \lambda u^T u = \lambda$$
则**使投影数据的方差最大的单位向量 $u$ 是数据的经验方差矩阵 $\Sigma$ 的主特征向量（特征值最大的特征向量）**。

对于 k > 1 的情况，PCA 的做法是，找到数据的经验方差矩阵 $\Sigma$ 的特征值最大的 k 个特征向量，作为数据的一组新的**正交基**。可以证明，这种做法最大化投影数据的方差。

注：对称矩阵的特征向量正交。
设 $A$ 是一个对称矩阵，$\lambda_1$ 和 $\lambda_2$ 是 $A$ 的两个不同的特征值，$v_1$ 和 $v_2$ 是其特征向量，则：
$$\lambda_1 v_2^T v_1 = v_2^T A v_1 = v_1^T A v_2 = \lambda_2 v_1^T v_2$$
因为 $\lambda_1 \neq \lambda_2$，所以：
$$v_1^T v_2 = 0$$

PCA 也可以看作是最小化投影数据和原始数据的近似误差。










