# PixelRNN/PixelCNN
从图像的左上角开始，往右下方逐个生成像素。
显式密度模型。显式地描述了每个像素基于其先前像素的条件概率。最大化似然函数：
$$p_\theta(x) = \prod_{i = 1}^n p_\theta(x_i | x_{i - 1} x_{i-2} \cdots x_1)$$
缺点：逐个生成像素，慢。

# AutoEncoder
自监督的数据降维和特征提取模型。
不是密度模型。

# Variational AutoEncoder
[从零推导：变分自编码器（VAE） - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/249296925)
显式密度模型。
## 隐变量模型
隐变量模型假设样本由以下随机过程生成：
1. 根据先验分布 $p_\theta(z)$ 生成隐变量 $z$
2. 根据隐变量 $z$ 和条件分布 $p_\theta(x|z)$ 生成样本 $x$
似然函数为：
$$p_\theta(x) = \int p_\theta(x|z) p(z) dz$$
其中，先验分布 $p(z)$ 固定为标准正态分布，条件分布 $p_\theta(x|z)$ 用神经网络拟合。基于标准正态分布可以被映射为任意分布的思想，隐变量模型将对 $p_\theta(x)$ 的概率密度估计问题转化为了对 $p_\theta(x|z)$ 的函数拟合问题。
缺点：$p_\theta(x)$ 涉及积分，难以计算。
## 变分推断
VAE 试图从后验分布 $p(z|x)$ 中采样 $z$。由于后验分布 $p(z|x)$ 未知，VAE 使用神经网络拟合后验分布 $q_\theta(z|x)$ 。VAE 同时优化后验分布和条件分布：
$$\ln p(x) - KL(q_\theta(z|x) || p(z|x)) = \mathrm{E}_{z\sim q_\theta(z|x)}[\ln p(x|z)] - KL(q_\theta(z|x)||p(z))$$
RHS 的第一项为重构损失，第二项为 KL 散度。
缺点：
- 无法控制生成样本的类别
- 生成的样本可能比较模糊（没有类似 GAN discriminator 的结构对生成的样本进行评价）

# Conditional Variational AutoEncoder
将每个数据映射到一个标准的正态分布（所有数据整体也构成一个标准的正态分布），除此之外，在解码时还添加了条件，从而控制生成样本的类别

# GAN
隐式密度模型。
Generator G + Discriminator D
$$\min_G \max_D \mathrm{E}_{x \sim p_{data}}[\log D(x)] + \mathrm{E}_{z \sim p(z)}[\log (1 - D(G(z)))]$$
1. 更新 generator，最大化 discriminator 做出错误判断的概率
$$\max_G \mathrm{E}_{z \sim p(z)}[\log D(G(z))]$$
2. 更新 discriminator，最大化 discriminator 做出正确判断的概率
$$\max_D \mathrm{E}_{x \sim p_{data}}[\log D(x)] + \mathrm{E}_{z \sim p(z)}[\log (1 - D(G(z)))]$$
3. 重复 1 和 2
在第 1 步中，我们最大化 discriminator 做出错误判断的概率，而不是最小化 discriminator 做出正确判断的概率。这样做使得在 generator 效果较差时梯度较大，在 generator 效果较好时梯度较小，有利于网络收敛。




