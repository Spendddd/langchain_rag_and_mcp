# MLA、MHA、GQA、MQA

State: 已完成

![image.png](MLA%E3%80%81MHA%E3%80%81GQA%E3%80%81MQA%201bae64a5662180b4b8f7e6419d7f225a/image.png)

# MLA

https://zhuanlan.zhihu.com/p/16730036197

https://spaces.ac.cn/archives/10091

![image.png](MLA%E3%80%81MHA%E3%80%81GQA%E3%80%81MQA%201bae64a5662180b4b8f7e6419d7f225a/image%201.png)

## **MLA原理解读**

### 完整计算过程

![image.png](MLA%E3%80%81MHA%E3%80%81GQA%E3%80%81MQA%201bae64a5662180b4b8f7e6419d7f225a/image%202.png)

![image.png](MLA%E3%80%81MHA%E3%80%81GQA%E3%80%81MQA%201bae64a5662180b4b8f7e6419d7f225a/image%203.png)

### **KV的计算过程**

下面我们参照图8的公式看看MHA的计算过程，首先对图中公式的变量做如下解释说明：

![image.png](MLA%E3%80%81MHA%E3%80%81GQA%E3%80%81MQA%201bae64a5662180b4b8f7e6419d7f225a/image%204.png)

![image.png](MLA%E3%80%81MHA%E3%80%81GQA%E3%80%81MQA%201bae64a5662180b4b8f7e6419d7f225a/image%205.png)

> 注：经过上述的变换，非常类似[LoRA](https://zhida.zhihu.com/search?content_id=252368029&content_type=Article&match_order=1&q=LoRA&zhida_source=entity)做低参数微调的逻辑。通过两个低秩矩阵先做压缩、再做扩展，最终能降低参数的数量。但MLA本质是要做到减少KV-cache的存储。LoRA强调的是参数量的减少，类似MLA这操作确实也减少了参数量，按DeepSeek-V3的参数配置，两个低秩矩阵参数量：2×dc×d=2×512×7168*，而正常MHA的参数矩阵参数量：*d×d=7168×7168。但MLA强调的是KV-cache的减少，也就是KV的激活值减少。当前我们还看不出来怎么减少激活值的数量的，因为单从KV的数量和维度上看跟MHA是一个量级，比GQA和MQA都要多，同时计算又多了一步。当前是比较迷糊的...我们再往下继续看...
> 

### **Q的计算过程**

![image.png](MLA%E3%80%81MHA%E3%80%81GQA%E3%80%81MQA%201bae64a5662180b4b8f7e6419d7f225a/image%206.png)

注：ctQ的维度是dq = 1536 = 3dc

### q，k增加RoPE位置编码

![image.png](MLA%E3%80%81MHA%E3%80%81GQA%E3%80%81MQA%201bae64a5662180b4b8f7e6419d7f225a/image%207.png)

![image.png](MLA%E3%80%81MHA%E3%80%81GQA%E3%80%81MQA%201bae64a5662180b4b8f7e6419d7f225a/image%208.png)

q的RoPE向量是用ctQ计算得到的，**WQR尺寸是dhR x dq**

**k的RoPE向量是用输入ht计算得到的**，**WKR尺寸是dhR x d**，其中**dhR = 1/2dh = 1/8dc = 1/24dq**

![image.png](MLA%E3%80%81MHA%E3%80%81GQA%E3%80%81MQA%201bae64a5662180b4b8f7e6419d7f225a/image%209.png)

![image.png](MLA%E3%80%81MHA%E3%80%81GQA%E3%80%81MQA%201bae64a5662180b4b8f7e6419d7f225a/image%2010.png)

所以到目前为止，我们得到的 q,k包括两部分拼接而成：一部分是做了低秩压缩得到的 q,k 向量，一部分是增加了RoPE位置编码的 q,k 向量。（后面这部分向量是基于MQA方式计算得到的，所有Head共享1个 k ）。

如何理解上述的操作过程？**这也是MLA方法的核心。**

**我们先来看看DeepSeek-V2论文中有一段原文解释（中文翻译）：**

![image.png](MLA%E3%80%81MHA%E3%80%81GQA%E3%80%81MQA%201bae64a5662180b4b8f7e6419d7f225a/image%2011.png)

「矩阵吸收计算」

![image.png](MLA%E3%80%81MHA%E3%80%81GQA%E3%80%81MQA%201bae64a5662180b4b8f7e6419d7f225a/image%2012.png)

通过上面的例子我们可以看到，两种方法计算出的结果是一样的，但第二种方法是先做了矩阵乘法，**相当于把** x1 **的变换矩阵** P **吸收到了** x2 **的变换矩阵** Q **里。**

理解了上面的例子，我们再来看看原文说的「**RoPE与低秩KV不兼容，没法做矩阵吸收计算**」的问题。

**a) 不加RoPE**

我们先假设当前不增加RoPE，那么 q,k 乘积计算如下，其中(i) 表示变换矩阵第 i 个Head的切片：

![image.png](MLA%E3%80%81MHA%E3%80%81GQA%E3%80%81MQA%201bae64a5662180b4b8f7e6419d7f225a/image%2013.png)

![image.png](MLA%E3%80%81MHA%E3%80%81GQA%E3%80%81MQA%201bae64a5662180b4b8f7e6419d7f225a/image%2014.png)

![image.png](MLA%E3%80%81MHA%E3%80%81GQA%E3%80%81MQA%201bae64a5662180b4b8f7e6419d7f225a/image%2015.png)

![image.png](MLA%E3%80%81MHA%E3%80%81GQA%E3%80%81MQA%201bae64a5662180b4b8f7e6419d7f225a/image%2016.png)

**【改进方法】c）通过增加一个很小 q,k 分量，引入RoPE**

**为了引入位置编码，作者在一个很小维度下，用MQA方式计算了 q,k** ，也就是在每层网络中，所有Head只计算一个 k （如论文中公式43所示）。引入位置编码的向量维度取的比较小为：
 **dhR = dh/2 = 128/2 = 64** 。

所以**最终 q,k 向量通过两部分拼接而成**，**计算权重时，由前后两部分分别相乘再相加得到**，如下公式（8）所示：

![image.png](MLA%E3%80%81MHA%E3%80%81GQA%E3%80%81MQA%201bae64a5662180b4b8f7e6419d7f225a/image%2017.png)

qti的维度是dh+dhR，kji的维度也是dh+dhR，向量内积结果是一个标量

![image.png](MLA%E3%80%81MHA%E3%80%81GQA%E3%80%81MQA%201bae64a5662180b4b8f7e6419d7f225a/image%2018.png)

![image.png](MLA%E3%80%81MHA%E3%80%81GQA%E3%80%81MQA%201bae64a5662180b4b8f7e6419d7f225a/image%2019.png)

![image.png](MLA%E3%80%81MHA%E3%80%81GQA%E3%80%81MQA%201bae64a5662180b4b8f7e6419d7f225a/image%2020.png)

上面我们就完整介绍了MLA做KV Cache压缩的原理。我们再来回顾下，MLA实际缓存的向量（如图8蓝色框），维度如下：

![image.png](MLA%E3%80%81MHA%E3%80%81GQA%E3%80%81MQA%201bae64a5662180b4b8f7e6419d7f225a/image%2021.png)

cKV是用来计算k和v的公共隐向量，分别用WUK和WUV还原成k和v

### **MLA与MQA、GQA对比**

最后我们再简单看看几种方法的对比，直接截取DeepSeeku-V2论文的图，如下：

![](https://pica.zhimg.com/v2-91ce40059dd409364fbc145b302fa8ca_1440w.jpg)

图9、MLA，MHA，GQA，MQA对比图

从上图我们可以看到，虽然MLA缓存的Latent KV比较短（相当于2.25个MQA的缓存量），但MLA有恢复全 k,v 的能力，特征表达能力显著比GQA、MQA要强。所以MLA能做到又快又省又强。论文中也给出了下图的数据

![image.png](MLA%E3%80%81MHA%E3%80%81GQA%E3%80%81MQA%201bae64a5662180b4b8f7e6419d7f225a/image%2022.png)