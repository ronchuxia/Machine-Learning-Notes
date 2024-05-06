可以将词向量作为可学习的参数，在训练语言模型的过程中学得词向量（词向量是神经网络语言模型训练后得到的副产品）。

# Word2Vec
[深入浅出Word2Vec原理解析 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/114538417)
训练策略：
- **CBOW (Continuous Bag of Words)**
	- 根据词 w 的前 c 个词和后 c 个词预测词 w
- **Skip-gram**
	- 根据词 w 预测词 w 的前 c 个词和后 c 个词

基础模型：
TODO

进阶模型：
Word2Vec 提出了两种进阶模型，比基础模型的训练速度更快。
- **Hierarchical Softmax**
	- 如果由神经网络一次性输出所有词的分数，然后计算 softmax，则当词表很大时计算 softmax 会非常耗时。因此，word2vec 使用 hierarchical softmax 结构计算近似 softmax 概率。
	- Hierarchical softmax 结构使用一个**哈夫曼树**。目标词的概率近似为从根节点到（目标词所在的）叶子节点的路径上的概率的乘积。
	- 路径上的每个内部节点都做二分类，都有一个参数向量 $\theta$ 用于计算二分类的概率。
- **Negative Sampling**
	- 对于目标词 w（正样本），使用加权采样抽取一些负样本，最小化负样本概率的同时最大化正样本概率

# GloVe
[（十五）通俗易懂理解——Glove算法原理 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/42073620
GloVe 结合了基于统计的词嵌入方法（充分利用统计信息、训练迅速，如：**基于窗口的共现矩阵 + SVD 分解**）和基于预测的词嵌入方法（适用于各种任务、可以捕捉复杂的模式，如：word2vec）的优点。

基本思想：将词向量作为可学习的参数，最大化词向量与统计信息的匹配度。
TODO



