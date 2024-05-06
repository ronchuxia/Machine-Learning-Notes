# Computing the Gradient
## 数值梯度
$$\frac{\partial f}{\partial x} = \lim_{h \rightarrow 0} \frac{f(x + h) - f(x - h)}{2h}$$
优点：容易计算
缺点：需要多次计算函数的值，**复杂度高**（与参数量成正比）
## 解析梯度
优点：复杂度低
缺点：容易算错
可以做 **gradient check**，用数值梯度验证解析梯度的正确性。

# Gradient Descent
一般用 stochastic gradient descent 指代 batch gradient descent。 