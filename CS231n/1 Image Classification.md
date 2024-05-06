# Nearest Neighbor Classifier
选择最近邻的 label 作为预测的 label。
## L1 dist vs. L2 dist
L1 距离与坐标轴的选取有关，L2 距离与坐标轴的选取无关。

# k - Nearest Neighbor Classifier
选择 k 个最近邻中出现次数最多的 label 作为预测的 label。

NN 和 kNN 的 decision boundary
![](https://cs231n.github.io/assets/knn.jpeg)

优点：
- 与 NN 相比更加平滑，对离群点不那么敏感，从而提高了泛化性

缺点：
- 测试非常耗时
	- 提出了 Approximate Nearest Neighbor (ANN) 算法，使用 kdtree 或 k-means 算法进行预处理
- 内存占用大
	- 需要保存所有训练数据
- 不适用于高维数据（比如图像）
	- 逐像素的 L2 距离并不能很好的衡量图像在语义层面的相似度
		- 比如将图像进行平移，则 L2 距离很大，但两张图像在语义上是相似的
		- 比如同一个物体的两张图像的背景色不一样，则 L2 距离很大，但两张图像在语义上是相似的
	- 随着维度的增加，需要的训练数据成指数级增长
		- 若特征维度为 n，则为了填充特征空间（保持样本间距基本不变），需要的样本数为 $k^n$
	- 可以使用 PCA，NCA 或 Random Projection 对高维数据进行降维

# Validation Sets for Hyperparameter Tuning
不能使用测试集进行调参，否则模型会**在测试集上过拟合**。
应将原始训练集拆分成新训练集和验证集，**在验证集上调参**。选择使模型在验证集上效果最好的超参数作为最终的超参数，然后再**在测试集上评价模型效果**。
**对于原始数据集较小的情况，可以使用交叉验证 (cross-validation)**：将原始数据集 n 等分，每一份依次作为验证集进行 n 次训练和验证，对验证集上的准确度求平均。

