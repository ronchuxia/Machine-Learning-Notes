# AutoEncoder
一种无监督的数据降维和特征提取模型

# Variational AutoEncoder
将每个数据映射到一个标准的正态分布（所有数据整体也构成一个标准的正态分布）
缺点：
- 无法控制生成样本的类别
- 生成的样本可能比较模糊（没有类似 GAN discriminator 的结构对生成的样本进行评价）

# Conditional Variational AutoEncoder
将每个数据映射到一个标准的正态分布（所有数据整体也构成一个标准的正态分布），除此之外，在解码时还添加了条件，从而控制生成样本的类别