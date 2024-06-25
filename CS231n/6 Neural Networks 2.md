# Data Preprocessing
- Mean subtraction
- Normalization
	- 除以标准差
	- 将范围限制在 -1 到 1
- PCA and Whitening
PCA 将数据投影到协方差矩阵特征值较大的几个特征向量张成的空间中，实现数据降维。
```python
# X: input data matrix, shape (N, D)
X -= np.mean(X, axis = 0) # zero-center the data (important)
cov = np.dot(X.T, X) / X.shape[0] # get the data covariance matrix

U,S,V = np.linalg.svd(cov) # SVD factorization of the data covariance matrix
# U: eigenvectors, shape (D, D); S: singular values, shape (D, ); 
# U 中的每一列是协方差矩阵的一个特征向量
# U 中的特征向量是一组正交归一化的向量，可以作为一组正交基。U 中的正交基按特征值的大小排序。

Xrot = np.dot(X, U) # decorrelate the data
# 去相关数据，即将数据从原空间投影到特征空间，投影矩阵即为特征空间的正交基矩阵
# Xrot 的协方差矩阵是对角矩阵

Xrot_reduced = np.dot(X, U[:,:100]) # shape (N, 100)
# 保留特征值较大的几个特征向量，将原始数据投影到这些特征向量（正交基）张成的空间中，实现数据降维
```
PCA 的原理：
[协方差矩阵的特征值与特征向量的几何意义 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/645372196?utm_id=0)
将原始数据投影到协方差矩阵特征值最大的特征向量的方向上时，投影数据的方差最大，即数据的散布程度最大。

白化将 PCA 降维后的数据，每一维除以这一维对应的特征值，从而 normalize 数据。如果输入数据服从多变量高斯分布，则白化后的数据服从均值为 0，协方差矩阵为单位阵的高斯分布。
```python
# whiten the data:
# divide by the eigenvalues (which are square roots of the singular values)
Xwhite = Xrot / np.sqrt(S + 1e-5)
```
白化可能导致**噪声增强**（原来散布程度较小的维度被拉伸）。使用较强的平滑可以缓解这一问题（将 `1e-5`改为较大的数）。
![](https://cs231n.github.io/assets/nn2/prepro2.jpeg)

在图像上进行 PCA 和白化的效果：
![](https://cs231n.github.io/assets/nn2/cifar10pca.jpeg)
将投影到特征空间的数据重新投影到像素空间：
```python
reduced_images = np.dot(X_rot_reduced, U.transpose()[:100, :])
```
- 特征值较大的特征向量捕捉低频信号
- 白化增强了噪声（高频信号）
一般不在卷积神经网络中使用 PCA 和白化。

标准化过程中使用的统计量应在训练集上计算，并应用于验证集和测试集。

# Weight Initialization
在数据标准化后，我们期望网络的权重一半为正，一半为负。

**不应使用全零初始化**
所有神经元的权重都一样，所有神经元的输出都一样，这使得反向传播中所有神经元的更新也都一样（神经元对称）。 

**小随机数初始化**
Symmetry breaking，避免了神经元对称。但权重并不是越小越好。如果权重很小，输出对于输入的梯度就很小，在深层网络中容易导致**梯度消失**。
```python
W = 0.01 * np.random.randn(n)
```

**Xavier 初始化**
输出 $s = \sum_i^n w_ix_s$ 的方差为：
$$\begin{align}
\text{Var}(s) & = \text{Var}(\sum_i^n w_ix_i)\\
& = \sum_i^n \text{Var}(w_ix_i)\\
& = \sum_i^n \text{Var}(w_i) \text{Var}(x_i) + [\text{E}(w_i)]^2\text{Var}(x_i) + [\text{E}(x_i)]^2\text{Var}(w_i)\\
& = \sum_i^n \text{Var}(w_i)\text{Var}(x_i)\\
& = n \text{Var}(w) \text{Var}(x)
\end{align}$$
上述过程基于以下假设和结论：
- 假设 $w_i$ 独立同分布
- 假设 $x_i$ 独立同分布
- 假设 $\text{E}(w_i) = 0$
- 假设 $\text{E}(x_i) = 0$，这并不一定成立，比如在使用 ReLU 激活函数时
- 对于独立同分布的变量 $X$ 和 $Y$，其乘积的方差满足：
$$
\begin{align}
\text{Var}(XY) & = \text{E}(X^2Y^2) - [\text{E}(XY)]^2\\
& = \text{E}(X^2)\text{E}(Y^2) - [\text{E}(X)]^2[\text{E}(Y)]^2\\
& = [\text{Var}(X) + [\text{E}(X)]^2] [\text{Var}(Y) + [\text{E}(Y)]^2] - [\text{E}(X)]^2[\text{E}(Y)]^2\\
& = \text{Var}(X)\text{Var}(Y) + [\text{E}(X)]^2\text{Var}(Y) + [\text{E}(Y)]^2\text{Var}(X)
\end{align}
$$

可见，神经元输出 $s$ 的方差和神经元输入 $x$ 的个数（维度）成正比。为了让 $\text{Var}(s) = \text{Var}(x)$，$\text{Var}(w)$ 应等于 1。因此，在使用标准正态分布初始化权重后，还应将权重**除以 $\sqrt{n}$**。
```python
w = np.random.randn(n) / sqrt(n)
```

这使得初始时，网络每一层的输出服从基本相同的分布，有利于网络的收敛。

另一项研究表明，考虑到反向传播，初始化权重应除以 $\sqrt{n_{in} + n_{out}}$。

**Kaiming 初始化（推荐）**
Xavier 初始化假设输入的期望为 0，这对于使用 ReLU 激活函数的神经网络不成立。Kaiming 认为使用ReLU 激活函数的神经网络，初始化权重应**除以 $\sqrt{\frac{2}{n}}$**。
```python
w = np.random.randn(n) / sqrt(2 / n)
```

考虑到反向传播，初始化权重应除以 $\sqrt{\frac{2}{n_{in} + n_{out}}}$。

**稀疏初始化**
大部分权重初始化为 0，小部分权重随机初始化。

由于权重的随机初始化已经提供了 symmetry breaking，**偏移量一般使用全零初始化**。

注：Batch Normalization 显著缓解了权重初始化的难题，因为它强制数据服从均值为 0，方差为 1 的分布（因此近似正态分布）。

# Regularization
正则化可以**防止过拟合**，从而**提高泛化性**。
**L2 正则化** 
$$\frac{1}{2}\lambda w^2$$
[[2 Linear Classification]]
偏好 diffuse 的权重，起到防止过拟合的作用。

**L1 正则化**
$$|w|$$
偏好**稀疏**的权重（有很多 0），起到特征选择的作用。
一般来讲，L2 正则化的效果优于 L1 正则化。
注：$w$ 接近 0 的时候，L1 范数比 L2 范数大，并且 L1 范数的**梯度只与符号有关**，而 L2 范数的梯度还需输入的大小有关，所以 L1 范数比 L2 范数更具稀疏性。

**Max norm constraints**
裁剪权重，起到避免梯度爆炸的作用。

**Dropout**
训练时，以一定的概率 $p$ 将神经元的输出置零。
测试时，保留所有的神经元输出，但是将输出乘以 $p$，从而使得每个神经元输出的期望不变（只在进行了 dropout 的层进行该操作）。

实际实现时，一般采用 **inverted dropout** 的方法，在训练时对神经元的输出进行缩放。这样的好处是不用修改测试代码，从而可以方便地调整 dropout 的使用位置。
```python
""" 
Inverted Dropout: Recommended implementation example.
We drop and scale at train time and don't do anything at test time.
"""

p = 0.5 # probability of keeping a unit active. higher = less dropout

def train_step(X):
  # forward pass for example 3-layer neural network
  H1 = np.maximum(0, np.dot(W1, X) + b1)
  U1 = (np.random.rand(*H1.shape) < p) / p # first dropout mask. Notice /p!
  H1 *= U1 # drop!
  H2 = np.maximum(0, np.dot(W2, H1) + b2)
  U2 = (np.random.rand(*H2.shape) < p) / p # second dropout mask. Notice /p!
  H2 *= U2 # drop!
  out = np.dot(W3, H2) + b3
  
  # backward pass: compute gradients... (not shown)
  # perform parameter update... (not shown)
  
def predict(X):
  # ensembled forward pass
  H1 = np.maximum(0, np.dot(W1, X) + b1) # no scaling necessary
  H2 = np.maximum(0, np.dot(W2, H1) + b2)
  out = np.dot(W3, H2) + b3
```

$p$ 一般取 0.5，也可以通过验证集调整。

Dropout 是一种降低噪声的方法，其他类似的方法还有 DropConnect（将权重随机置零） 等。

由于 bias 不与输入相乘，**一般不对 bias 做正则化**。

**Batch Normalization**
Machine learning methods tend to perform better with input data consisting of **uncorrelated features** with **zero mean** and **unit variance**. When training a neural network, we can **preprocess** the data before feeding it to the network to explicitly decorrelate its features. This will ensure that the first layer of the network sees data that follows a nice distribution. However, even if we preprocess the input data, **the activations at deeper layers of the network will likely no longer be decorrelated and will no longer have zero mean or unit variance**, since they are output from earlier layers in the network. Even worse, during the training process **the distribution of features at each layer of the network will shift as the weights of each layer are updated**.

# Loss Functions
## Classification
**SVM Loss**
[[2 Linear Classification]]

**Cross Entropy Loss**
[[2 Linear Classification]]
当类数很多时，计算 Cross Entropy Loss 比较耗时，可以使用一些近似方法，比如 NLP 中的 Hierarchical Softmax。

## Attribute Classification
一个样本可以有多个 attribute，这些 attribute 之间不互斥。可以对每个 attribute 训练一个二分类器。

**Binary SVM Loss**
$$L_i = \sum_j \max(0, 1 - y_{ij}f_j)$$ 样本 $i$ 的损失 $L_i$ 对属性 $j$ 求和，其中 $y_{ij}$ 满足：
$$y_{ij} = \begin{cases} +1 \quad \text{样本 i 具有属性 j}\\
-1 \quad \text{样本 i不具有属性 j}
\end{cases}$$
$f_j$ 满足 $f_j$：
$$
\begin{cases}
f_j \geq 0 \Rightarrow \text{样本 i 被判断为具有属性 j}\\
f_j < 0 \Rightarrow \text{样本 i 被判断为不具有属性 j}
\end{cases}
$$

当样本 $i$ 具有属性 $j$，而 $f_j < 1$，或样本 $i$ 不具有属性 $j$，而 $f_j > -1$ 时，损失将会累积。

**Binary Cross Entropy Loss / Logistic Regression Loss**
[[2 Logistic Regression]]

## Regression
**L2 Norm**
L2 损失函数与更稳定的损失函数（比如 Cross Entropy Loss）相比更难优化：
- L2 损失**要求每个样本的预测值都非常准确**，而 Cross Entropy Loss 只要求分数的相对大小准确
- L2 损失**对离群值敏感**，离群值的梯度会很大
- L2 损失只输出一个预测值，而 Cross Entropy Loss 可以输出一个分布（即预测的 confidence）
因此，一般**将回归问题离散化为分类问题**，从而使用分类问题的损失函数。 

**L1 Norm**

## Structured Prediction
需要特殊的优化器，以对 structure space 进行约束。











