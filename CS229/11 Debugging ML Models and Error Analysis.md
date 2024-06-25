# Debugging ML Models
可能导致算法效果较差的原因有：
- high bias：训练损失和测试损失都很大
- high variance：训练损失和测试损的差距较大
- poor optimization algorithm：最优参数的损失小于输出参数的损失
- poor objective function：最优参数的损失大于输出参数的损失

# Error Analysis
如果模型由很多部件组成，我们需要找到那个提升空间最大的部件，并对它进行改进。有两种方法：
- 依次用 ground truth 替换每个部件的输出，替换后测试准确率提升最大的部件就是提升空间最大的部件
- 依次用 baseline 替换每个部件，替换后测试准确率下降最大的部件就是贡献最大的部件