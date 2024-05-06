# GPT-1
有效地从无标签文本中利用超单词级信息有两个主要的难点：  
- 无法确定什么样的优化目标能学到最有效的文本表征，使之更好的用于迁移目的
- 对于学习到的文本表征，采用何种方式将其迁移到目标任务上，目前尚无共识

本文提出了一种半监督的学习方式：
- 无监督 Pre-training
- 有监督 Fine-tuning

目的是学习一种通用的表示方法，针对不同种类的任务只需略作修改便能适应。

## Pre-training
使用改造的 transformer decoder 作为网络结构。去除了 encoder-decoder attention 部分。

![[Pasted image 20240221204031.png|250]]

将输入序列的最后一个 token 的 transformer 输出（结合了整个输入序列的特征）作为下一个 token 的预测依据。

训练时，可以并行完成。推理时，需要一个词一个词地生成。

GPT-1 使用 BPE 算法对文本进行编码。

## Fine-tuning

## Task-specific input transformations
针对简单任务如文本预测、文本分类等，只需要 Fine-tuning。
针对其他任务如文本蕴含、文本相似度、问答、常识推理等，需要对模型略作修改。

注：
文本蕴含（Text entailment）：给定一个前提文本（Premise）和假说文本（Hypothesis），推断前提文本与假说文本的关系，一般分为蕴含关系（Entailment）和矛盾关系（Contradiction）。因此文本蕴含即分类，其输出就是这几个关系的概率值。

![[Pasted image 20240221202821.png]]

## Implementations
- [karpathy/minGPT: A minimal PyTorch re-implementation of the OpenAI GPT (Generative Pretrained Transformer) training (github.com)](https://github.com/karpathy/minGPT)
- [huggingface/pytorch-openai-transformer-lm: 🐥A PyTorch implementation of OpenAI's finetuned transformer language model with a script to import the weights pre-trained by OpenAI (github.com)](https://github.com/huggingface/pytorch-openai-transformer-lm)

# T5 (Text-To-Text Transfer Transformer)
[T5 模型：NLP Text-to-Text 预训练模型超大规模探索 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/88438851)
- Transformer Encoder-Decoder 模型
- BERT-style 的破坏方法
- Replace Span 的破坏策略（将破坏的部分替换成单个 \[MASK\]）
- 15% 的破坏比
- 3 的破坏小段长度

Text-to-Text：将所有任务都转换为文本到文本任务（类似 GPT-2）。

# Llama