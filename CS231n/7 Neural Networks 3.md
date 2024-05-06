# Gradient Check
[[3 Optimization]]
**使用 centered difference formula 计算数值梯度**
$$\frac{df(x)}{dx} = \frac{f(x+h) - f(x)}{h}$$
$$\frac{df(x)}{dx} = \frac{f(x+h) - f(x-h)}{2h}$$
使用泰勒展开可得，第一个式子的误差为 $O(h)$，第二个式子的误差为 $O(h^2)$，因此应使用第二个式子计算数值梯度。

**使用相对误差对比数值梯度和解析梯度**
$$\frac{|f'_a - f'_n|}{\max(f'_a, f'_n)}$$
对于单层网络：
- 对于没有 kink 的目标函数，1e-7 的相对误差是合理的
- 对于有 kink 的目标函数，1e-4 的相对误差是合理的（在 kink 处，数值梯度的计算是不准确的）

网络越深，相对误差越大。

**使用双精度浮点数**

**让梯度保持在浮点数的精度范围内**
打印梯度，对梯度进行检查，如果梯度太小，将损失函数乘以一个常数，将梯度拉回到浮点数的精度范围内。

**使用合适的 $h$**
$h$ 并不是越小越好。如果 $h$ 太小，容易导致数值精度问题。建议 $h$ 取 1e-4 或 1e-6。

**在一段时间内进行 gradient check**
可以让网络训练一段时间，在这段时间内进行 gradient check。这样做可以防止梯度计算在初始化的情况下（比如小随机数）是正确的，而在一般的情况下不正确。

**屏蔽正则化损失**
如果正则化损失对梯度的影响过大，由于正则化损失的梯度不容易算错，即使通过了 gradient check，数据损失的梯度也有可能是错的。在进行 gradient check 时，应先屏蔽正则化损失，只检查数据损失的梯度，然后再添加正则化损失，检查完整损失的梯度。

**固定不确定因素**
Dropout 和随机数据增强可能导致 gradient check 不通过。在进行 gradient check 时，应固定随机因素的种子，使随机因素在两次损失计算时保持固定。

**检查部分维度**
检查整个权重参数向量中的一部分。采样时应按照不同参数的占比进行采样，而不应随机采样（比如 bias 参数比 weight 参数少很多，随机采样可能没有采样到 bias 参数）。

# Before Learning: Sanity Checks Tips/Tricks
先屏蔽正则化损失，然后进行以下检查：
- 初始化损失：小随机数初始化时的损失应与预测值相符
- 过拟合小子集：模型在完整数据集的一个小子集上应能达到 0 损失

# Babysitting the Learning Process
训练过程中可以监控以下指标：
- 损失函数
	- 根据**下降速度**调整 learning rate
	- 根据**抖动程度**调整 batch size
- 训练集和测试集准确率
	- 根据训练集和测试集准确的差值判断是否过拟合
- 更新：权重比
	- 根据权重更新值（learning rate * dW）的模长与权重的模长的比值调整 learning rate
	- 更新：权重比应在 1e-3 左右
- 激活函数的输出的分布
	- 激活函数的输出应布满激活函数的值域
- 特征图
	- 特征图应呈现出一定的特征，而非随机噪声

# Parameter Updates
[[优化算法]]
## Learning rate decay
1. 指数衰减
$$\alpha = \alpha_0 e^{-kt}$$
3. 倒数衰减
$$\alpha = \frac{\alpha_0}{1 + kt}$$
## Second order methods
[[2 Logistic Regression]]
由于计算海森矩阵的逆需要大量的内存，深度学习中一般不使用牛顿法。
L-BFGS 算法近似海森矩阵，但需要在整个数据集上计算，因此也很少使用。

# Hyperparameter Optimization
**Hyperparameter ranges**
Because learning rate and regularization strength have **multiplicative** effects on the training dynamics, it is much more natural to consider a range of learning rate multiplied or divided by some value, than a range of learning rate added or subtracted to by some value.
```python
learning_rate = 10 ** uniform(-6, 1)
```

**Prefer random search to grid search**
**随机选取超参数**比等距选取超参数有更大的概率找到最优的超参数。
![](https://cs231n.github.io/assets/nn3/gridsearchbad.jpeg)

**Careful with best values on border**

**Stage your search from coarse to fine**

# Model Ensembles
- 不同的初始化
- 不同的超参数
- 不同的 checkpoint
- 对最后几次迭代的权重求平均
	最后几次迭代的权重一般在最优解附近摆动，其平均值有较大的概率更接近最优解。

