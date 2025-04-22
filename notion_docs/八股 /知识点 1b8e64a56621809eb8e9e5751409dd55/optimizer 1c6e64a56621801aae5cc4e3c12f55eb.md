# optimizer

State: 已完成

# **常用的优化器**

https://zhuanlan.zhihu.com/p/6023762003：**【机器学习】优化器optimizer**

https://zhuanlan.zhihu.com/p/6023762003：**一文搞懂--深度学习中的优化器**

https://blog.csdn.net/qq_35353673/article/details/145813042

https://zhuanlan.zhihu.com/p/4536946192：**算法面经系列：L1和L2正则化**

- 前置知识：L1、L2回归
    
    1. L1正则化（[Lasso回归](https://zhida.zhihu.com/search?content_id=249929658&content_type=Article&match_order=1&q=Lasso%E5%9B%9E%E5%BD%92&zhida_source=entity)）:
    
    正则化项是模型参数的绝对值之和。公式可以表示为：
    
    ![image.png](%E6%AD%A3%E5%88%99%E5%8C%96%201b8e64a5662180a28b56feb48f302875/image.png)
    
    因此，L1正则化的损失函数为：
    
    ![image.png](%E6%AD%A3%E5%88%99%E5%8C%96%201b8e64a5662180a28b56feb48f302875/image%201.png)
    
    **L1正则化会导致模型的参数趋向稀疏化**，即有些参数被压缩到零，从而有**特征选择**的效果。
    
    2. L2正则化（[Ridge回归](https://zhida.zhihu.com/search?content_id=249929658&content_type=Article&match_order=1&q=Ridge%E5%9B%9E%E5%BD%92&zhida_source=entity)、岭回归）:
    
    正则化项是模型参数的平方和。公式可以表示为：
    
    ![image.png](%E6%AD%A3%E5%88%99%E5%8C%96%201b8e64a5662180a28b56feb48f302875/image%202.png)
    
    因此，L2正则化的损失函数为：
    
    ![image.png](%E6%AD%A3%E5%88%99%E5%8C%96%201b8e64a5662180a28b56feb48f302875/image%203.png)
    
    **L2正则化会将模型参数压缩到接近零，但一般不为零，因此不会进行特征选择**。
    

一阶优化器：[SGD](https://link.zhihu.com/?target=https%3A//so.csdn.net/so/search%3Fq%3DSGD%26spm%3D1001.2101.3001.7020)、SDGwith Momentum、NAG（牛顿动量法）、AdaGrad（自适应梯度）、[RMSProp](https://zhida.zhihu.com/search?content_id=250227161&content_type=Article&match_order=1&q=RMSProp&zhida_source=entity)（均方差传播）、[Adam](https://zhida.zhihu.com/search?content_id=250227161&content_type=Article&match_order=1&q=Adam&zhida_source=entity)、Nadam、共轭梯度法

二阶优化器：牛顿法、拟牛顿法、[BFGS](https://zhida.zhihu.com/search?content_id=250227161&content_type=Article&match_order=1&q=BFGS&zhida_source=entity)、[L-BFGS](https://zhida.zhihu.com/search?content_id=250227161&content_type=Article&match_order=1&q=L-BFGS&zhida_source=entity)

其中：AdaGrad算法，RMSProp算法，Adam算法以及AdaDelta算法是自适应学习率算法

![image.png](optimizer%201c6e64a56621801aae5cc4e3c12f55eb/image.png)

## 1、SGD、[BGD](https://zhida.zhihu.com/search?content_id=250227161&content_type=Article&match_order=1&q=BGD&zhida_source=entity)、[MBGD](https://zhida.zhihu.com/search?content_id=250227161&content_type=Article&match_order=1&q=MBGD&zhida_source=entity)

### SGD：随机梯度下降

SGD由于**每次参数更新仅仅需要计算一个样本的梯度**，训练速度很快，即使在样本量很大的情况下，可能只需要其中一部分样本就能迭代到最优解，由于每次迭代并不是都向着整体最优化方向，导致**梯度下降的波动非常大，更容易从一个局部最优跳到另一个局部最优，准确度下降**。通常，SGD收敛速度要比BGD快。

**SGD缺点：**

- 选择合适的learning rate比较困难 ，学习率太低会收敛缓慢，学习率过高会使收敛时的波动过大
- 所有参数都是用同样的learning rate
- SGD容易收敛到局部最优，并且在某些情况下可能被困在鞍点

### BGD：批量梯度下降

每进行一次参数更新，**需要计算整个数据样本集**，因此导致批量梯度下降法的速度会比较慢，尤其是数据集非常大的情况下，收敛速度就会非常慢，**但是由于每次的下降方向为总体平均梯度，它得到的会是一个全局最优解。**

### MBGD：小批量梯度下降

小批量梯度下降，对SGD和BGD的折中，**每次使用batch_size个样本对参数进行更新**。小批量梯度下降法即保证了训练的速度，又能保证最后收敛的准确率，**目前的SGD默认是小批量梯度下降算法**。

# 梯度优化算法

## 动量优化法

动量优化方法引入物理学中的动量思想，**加速梯度下降**，有**Momentum和[Nesterov](https://zhida.zhihu.com/search?content_id=250227161&content_type=Article&match_order=1&q=Nesterov&zhida_source=entity)**两种算法。当我们将一个小球从山上滚下来，没有阻力时，它的动量会越来越大，但是如果遇到了阻力，速度就会变小，动量优化法就是借鉴此思想，使得**梯度方向在不变的维度上，参数更新变快，梯度有所改变时，更新参数变慢，这样就能够加快收敛并且减少动荡**。

### 2、SGD with [Momentum](https://link.zhihu.com/?target=https%3A//so.csdn.net/so/search%3Fq%3DMomentum%26spm%3D1001.2101.3001.7020)：考虑之前的梯度

**参数更新时在一定程度上保留之前更新的方向，同时又利用batch的梯度微调最终的更新方向**，SGD只使用了当步参数的梯度，随机性较大，如果将历次迭代的梯度按比例融合，可能更稳定。

![](https://pic1.zhimg.com/v2-c4e94ab0166662bb2ac68b75a92ba11a_1440w.jpg)

在梯度方向改变时，momentum能够降低参数更新速度，从而减少震荡；在梯度方向相同时，momentum可以加速参数更新， 从而加速收敛。总而言之，momentum能够加速SGD收敛，抑制震荡

### 3、NAG：考虑动量一起计算梯度

![image.png](optimizer%201c6e64a56621801aae5cc4e3c12f55eb/image%201.png)

在梯度更新时将之前的动量加进来一起计算梯度

# 自适应学习率优化算法

在机器学习中，学习率是一个非常重要的超参数，但是学习率是非常难确定的，虽然可以通过多次训练来确定合适的学习率，但是一般也不太确定多少次训练能够得到最优的学习率，玄学事件，对人为的经验要求比较高，所以是否存在一些策略自适应地调节学习率的大小，从而提高训练速度。 目前的自适应学习率优化算法主要有：AdaGrad算法，RMSProp算法，Adam算法以及AdaDelta算法。

### 4、AdaGard：自适应梯度，梯度平方累计

针对SGD中始终使用一个学习率的问题，AdaGard在每次进行参数更新的时候，对于每个参数，初始化一个 Gt = 0，然后每次将该参数的梯度平方求和累加到这个变量 Gt 上，而在更新这个参数的时候，学习率就变成：

![image.png](optimizer%201c6e64a56621801aae5cc4e3c12f55eb/image%202.png)

但是这种方法会存在二阶动量一直累加的问题，学习率迟早要很小。

**缺点：**

- 仍需要手工设置一个全局学习率 *η* , 如果 *η* 设置过大的话，会使regularizer过于敏感，对梯度的调节太大（通常设置为 0.01 或 0.001）
- 中后期，分母上梯度累加的平方和会越来越大，使得参数更新量趋近于0，使得训练提前结束，无法学习

### 5、RMSprop算法：在AdaGrad基础上进行梯度平方的指数平移

![image.png](optimizer%201c6e64a56621801aae5cc4e3c12f55eb/image%203.png)

### 6、Adadelta：用参数变化量平方的指数平移代替显式学习率

![image.png](optimizer%201c6e64a56621801aae5cc4e3c12f55eb/image%204.png)

### 7、Adam 适应性矩估计：在AdaGrad的基础上用梯度指数平移代替显式学习率，还引入修正因子对平移后的梯度和梯度方进行偏差修正

![image.png](optimizer%201c6e64a56621801aae5cc4e3c12f55eb/image%205.png)

![image.png](optimizer%201c6e64a56621801aae5cc4e3c12f55eb/image%206.png)

**Adam**：采用 **L2 正则化**，通过**在梯度更新时手动添加 `weight_decay` 项**：

![image.png](optimizer%201c6e64a56621801aae5cc4e3c12f55eb/image%207.png)

其中，λ 是[权重衰减](https://so.csdn.net/so/search?q=%E6%9D%83%E9%87%8D%E8%A1%B0%E5%87%8F&spm=1001.2101.3001.7020)系数。

Adam中**动量直接并入了梯度一阶矩（指数加权）的估计**。其次，相比于缺少修正因子导致二阶矩估计可能在训练初期具有很高偏置的RMSProp，**Adam包括偏置修正，修正从原点初始化的一阶矩（动量项）和（非中心的）二阶矩估计**。

**特点：**

- Adam梯度**经过偏置校正后，每一次迭代学习率都有一个固定范围，使得参数比较平稳**。
- **结合了Adagrad善于处理稀疏梯度和RMSprop善于处理非平稳目标的优点**
- 为不同的参数计算不同的自适应学习率
- 也**适用于大多非凸优化问题**——适用于大数据集和高维空间。

**一阶矩和二阶矩的作用**

1. 一阶矩估计的作用
    - 加速收敛：通过动量机制保留历史梯度方向，减少震荡，使参数更新更稳定.
    - 捕捉梯度趋势：在非凸优化问题中，帮助模型避开局部极小值，向全局最优方向移动。
2. 二阶矩估计的作用
    - 自适应学习率：根据梯度方差调整步长。梯度变化大时，学习率自动减小（因 Gt 较大），防止震荡；梯度变化小时，学习率增大，加快收敛。
    - 处理稀疏梯度：对稀疏数据(如自然语言处理任务)中的低频参数分配更大更新步长，提升训练效率。

![image.png](optimizer%201c6e64a56621801aae5cc4e3c12f55eb/image%208.png)

![image.png](optimizer%201c6e64a56621801aae5cc4e3c12f55eb/image%209.png)

### 8、Adamw

![image.png](optimizer%201c6e64a56621801aae5cc4e3c12f55eb/image%2010.png)

**AdamW**：直接在**参数更新时**执行权重衰减，而不是在[梯度计算](https://so.csdn.net/so/search?q=%E6%A2%AF%E5%BA%A6%E8%AE%A1%E7%AE%97&spm=1001.2101.3001.7020)时添加 L2 正则：

![image.png](optimizer%201c6e64a56621801aae5cc4e3c12f55eb/image%2011.png)

这里权重衰减项独立于梯度计算，在更新参数时进行。

### AdamW相比于Adam做出的修改有何作用

- **Adam 的 L2 正则化** 会影响梯度估计值，导致优化器在 **自适应学习率的调节** 过程中对权重衰减的影响不稳定。
- **AdamW 的独立权重衰减** 让权重衰减成为一个**真正的正则化项**，而**不会干扰梯度估计**，使得模型的 **泛化能力更强**。

# 各个优化器特点以及演化对比

GD（梯度下降）/BGD（批量梯度下降）：使用全部训练集，沿着梯度反方向以固定学习率更新参数。
缺点：计算全局梯度耗时长；对所有参数使用相同学习率；只根据当前时刻梯度，容易陷入局部最优解。

SGD（随机梯度下降）：相较于GD，只使用1个样本的来计算梯度。

优点：对于大数据集可以更快进行更新。

缺点：学习率固定；使用一个样本的梯度，优化过程震荡，需要长时间才能收敛。

MBGD（小批量梯度下降）：相较于SGD，使用一个batch的样本的梯度。收敛更稳定。

Momenum（动量优化）：在SGD/MBGD基础上引入动量，每次的更新量不是梯度*学习率，而是动量。 M t = β ⋅ M t − 1 + η ⋅ g t M_t = \beta \cdot M_{t-1}+\eta \cdot g_t Mt=β⋅Mt−1+η⋅gt

优点：加速收敛，抑制震荡，跳出局部最优解。

缺点：需要人工设置动量参数，学习率固定。

AdaGrad（自适应梯度优化器）：自适应学习率，与梯度平方和有关.

![image.png](optimizer%201c6e64a56621801aae5cc4e3c12f55eb/c9a4f552-97e3-43d5-b5b0-7e0909cbbbdd.png)

RMSProp：用指数平移解决AdaGrad学习率一直变小的问题，但是模型还是会对初始学习率敏感

![image.png](optimizer%201c6e64a56621801aae5cc4e3c12f55eb/72c9ec62-5bd8-4575-b5ac-68943cc09db4.png)

Adadelta：用梯度更新量的平方指数平移代替全局标量学习率

![image.png](optimizer%201c6e64a56621801aae5cc4e3c12f55eb/4ead1d79-fa51-4522-b094-fc26f2db0945.png)

Adam(自适应矩估计)：结合了Momentum和RMSProp的优点，既保留了动量，又有自适应学习率。和SGD with Momentum相比，自适应学习率可以很大程度上避免学习率选择不当带来的训练震荡或收敛速度慢的问题。