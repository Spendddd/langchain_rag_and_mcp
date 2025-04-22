# embedding

State: 学习中

https://zhuanlan.zhihu.com/p/664867771

### **如何生成和存储 Embedding**

生成Embedding的方法有很多。这里列举几个比较经典的方法和库：

- Word2Vec：是一种基于神经网络的模型，用于将单词映射到向量空间中。Word2Vec包括两种架构：CBOW (Continuous Bag-of-Words) 和 Skip-gram。**CBOW 通过上下文预测中心单词，而 Skip-gram 通过中心单词预测上下文单词**。这些预测任务训练出来的神经网络权重可以用作单词的嵌入。
- GloVe：全称为 Global Vectors for Word Representation，是一种基于共现矩阵的模型。该模型**使用统计方法来计算单词之间的关联性，然后通过奇异值分解（SVD）来生成嵌入**。GloVe 的特点是在计算上比 Word2Vec 更快，并且可以扩展到更大的数据集。
- FastText：是由 Facebook AI Research 开发的一种模型，它在 Word2Vec 的基础上**添加了一个字符级别的 n-gram 特征**。这使得 FastText **可以将未知单词的嵌入表示为已知字符级别 n-gram 特征的平均值**。FastText 在处理不规则单词和罕见单词时表现出色。
- 大模型的 Embeddings：如OpenAI官方发布的 第二代模型：text-embedding-ada-002。它最长的输入是8191个tokens，输出的维度是1536。