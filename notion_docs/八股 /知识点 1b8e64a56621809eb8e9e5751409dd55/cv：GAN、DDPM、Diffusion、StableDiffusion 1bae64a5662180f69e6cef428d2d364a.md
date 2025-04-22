# cv：GAN、DDPM、Diffusion、StableDiffusion

State: 已完成

# GAN

输入噪音，输出高度模拟

https://zhuanlan.zhihu.com/p/580137376：**AIGC爆火的背后——对抗生成网络GAN浅析**

# DDPM

https://zhuanlan.zhihu.com/p/590840909：**AIGC爆火的背后——扩散模型DDPM浅析**

和GAN相比，DDPM拟合的是加噪图片，并通过反向过程（去噪）生成原始图片。而GAN通过判别器拟合原始图片，和DDPM有本质的区别。

![image.png](cv%EF%BC%9AGAN%E3%80%81DDPM%E3%80%81Diffusion%E3%80%81StableDiffusion%201bae64a5662180f69e6cef428d2d364a/image.png)

DDPM是拟合整个从真实图片x0到随机高斯噪声z的过程，再通过反向过程生成新的图片

## 模型原理

DDPM包括两个过程：**前向过程（forward process）**和**反向过程（reverse process）**，其中前向过程又称为**扩散过程（diffusion process）**，如下图所示。无论是前向过程还是反向过程都是一个**参数化的[马尔可夫链](https://zhida.zhihu.com/search?content_id=219431759&content_type=Article&match_order=1&q=%E9%A9%AC%E5%B0%94%E5%8F%AF%E5%A4%AB%E9%93%BE&zhida_source=entity)（Markov chain）**，其中反向过程可以用来生成图片。

- 马尔可夫链
    
    马尔可夫链是指**一个随机过程**，其中**未来状态的概率分布仅依赖于当前状态，而与过去的任何状态无关**。在原文中，马尔可夫链被用来描述DDPM模型中的前向和反向过程，这两个过程都是参数化的马尔可夫链。具体来说：
    
    - **马尔可夫链的具体含义**：
        - **无记忆性**：马尔可夫链的核心特性是**“无记忆性”**，意味着系统在下一时刻的状态只取决于当前状态，而不受之前状态的影响。这使得马尔可夫链在处理时间序列数据时非常有用，因为它**可以简化模型，避免考虑所有历史信息**。
        - **状态转移**：**在一个马尔可夫链中，系统从一个状态转移到另一个状态，转移的概率由转移概率矩阵决定**。这个矩阵中的每个元素表示从一个状态转移到另一个状态的概率。
        - **转移概率矩阵**：转移概率矩阵是一个描述状态之间转移概率的矩阵。例如，如果系统有三个状态A、B和C，转移概率矩阵可能如下所示：
            - A到A的概率：0.3
            - A到B的概率：0.7
            - B到A的概率：0.9
            - B到B的概率：0.1
            - C到A的概率：0.5
            - C到B的概率：0.5
        - **时间齐次性**：**在时间齐次的马尔可夫链中，转移概率不随时间变化**，这意味着无论何时从一个状态转移到另一个状态，转移概率都是相同的。
    - **马尔可夫链在DDPM中的应用**：
        - 在DDPM模型中，前向过程是一个扩散过程，它将原始图像逐渐转换为随机噪声，这个过程可以看作是一个马尔可夫链，其中每个时间步的状态都是当前状态的噪声版本。
        - 反向过程则是从随机噪声生成图像的过程，同样可以视为一个马尔可夫链，其中每个时间步的状态都是基于当前噪声状态生成的图像。
        - 通过参数化的马尔可夫链，DDPM模型能够在生成图像时逐步恢复细节，从而实现高质量的图像生成。
    - **马尔可夫链的其他应用**：
        - **强化学习**：在强化学习中，马尔可夫决策过程（MDP）是马尔可夫链的一个扩展，用于描述智能体在环境中的决策过程。
        - **自然语言处理**：隐马尔可夫模型（HMM）用于处理语音识别、词性标注等问题，其中状态序列代表隐藏的语音或词性，观察序列代表实际的语音信号或文本。
        - **金融建模**：马尔可夫链用于预测股市趋势，其中状态可以代表不同的市场条件，转移概率表示市场条件的变化。
    
    通过这些应用，可以看出马尔可夫链在处理复杂系统和时间序列数据时具有广泛的应用价值。
    

![image.png](cv%EF%BC%9AGAN%E3%80%81DDPM%E3%80%81Diffusion%E3%80%81StableDiffusion%201bae64a5662180f69e6cef428d2d364a/image%201.png)

## **DDPM前向过程（扩散过程）**

一句话概括，前向过程就是**对原始图片 x0 不断加高斯噪声最后生成随机噪声 xT** 的过程，如下图所示（[图片来自于网络](https://link.zhihu.com/?target=https%3A//drive.google.com/file/d/1DYHDbt1tSl9oqm3O333biRYzSCOtdtmn/view)）。

![image.png](cv%EF%BC%9AGAN%E3%80%81DDPM%E3%80%81Diffusion%E3%80%81StableDiffusion%201bae64a5662180f69e6cef428d2d364a/image%202.png)

![image.png](cv%EF%BC%9AGAN%E3%80%81DDPM%E3%80%81Diffusion%E3%80%81StableDiffusion%201bae64a5662180f69e6cef428d2d364a/image%203.png)

![image.png](cv%EF%BC%9AGAN%E3%80%81DDPM%E3%80%81Diffusion%E3%80%81StableDiffusion%201bae64a5662180f69e6cef428d2d364a/image%204.png)

## **DDPM反向过程（去噪过程）**

前向过程是将原始图片变成随机噪声，而反向过程就是**通过预测噪声 ϵ ，将随机噪声 xT 逐步还原为原始图片 x0** ，如下图所示。

![image.png](cv%EF%BC%9AGAN%E3%80%81DDPM%E3%80%81Diffusion%E3%80%81StableDiffusion%201bae64a5662180f69e6cef428d2d364a/image%205.png)

![image.png](cv%EF%BC%9AGAN%E3%80%81DDPM%E3%80%81Diffusion%E3%80%81StableDiffusion%201bae64a5662180f69e6cef428d2d364a/image%206.png)

![image.png](cv%EF%BC%9AGAN%E3%80%81DDPM%E3%80%81Diffusion%E3%80%81StableDiffusion%201bae64a5662180f69e6cef428d2d364a/image%207.png)

**DDMP的关键是训练噪声估计模型以估计真实的噪声**

## **DDPM如何训练**

### 完整损失函数

DDPM的损失函数基于变分下界（VLB），其推导过程涉及对数似然和KL散度的计算。最终的损失函数可以表示为：

![image.png](cv%EF%BC%9AGAN%E3%80%81DDPM%E3%80%81Diffusion%E3%80%81StableDiffusion%201bae64a5662180f69e6cef428d2d364a/image%208.png)

前面提到DDPM的关键是训练噪声估计模型 ϵθ(xt,t) ，用于估计 ϵ ，那么**损失函数可以使用[MSE误差](https://zhida.zhihu.com/search?content_id=219431759&content_type=Article&match_order=1&q=MSE%E8%AF%AF%E5%B7%AE&zhida_source=entity)，**表示如下：

![image.png](cv%EF%BC%9AGAN%E3%80%81DDPM%E3%80%81Diffusion%E3%80%81StableDiffusion%201bae64a5662180f69e6cef428d2d364a/image%209.png)

![image.png](cv%EF%BC%9AGAN%E3%80%81DDPM%E3%80%81Diffusion%E3%80%81StableDiffusion%201bae64a5662180f69e6cef428d2d364a/image%2010.png)

![image.png](cv%EF%BC%9AGAN%E3%80%81DDPM%E3%80%81Diffusion%E3%80%81StableDiffusion%201bae64a5662180f69e6cef428d2d364a/image%2011.png)

![image.png](cv%EF%BC%9AGAN%E3%80%81DDPM%E3%80%81Diffusion%E3%80%81StableDiffusion%201bae64a5662180f69e6cef428d2d364a/image%2012.png)

整个训练过程可以表示如下：

![image.png](cv%EF%BC%9AGAN%E3%80%81DDPM%E3%80%81Diffusion%E3%80%81StableDiffusion%201bae64a5662180f69e6cef428d2d364a/image%2013.png)

论文中的DDPM训练过程如下所示：

![](https://pic4.zhimg.com/v2-a3297eca7fadadb9c2cfe91ba28447ab_1440w.jpg)

## **DDPM如何生成图片**

在得到预估噪声 ϵθ(xt,t) 后，就可以按公式（3）逐步得到最终的图片 x0 ，整个过程表示如下：

![](https://pic1.zhimg.com/v2-7a0073eb21ab840c0ab580a4d9ad3164_1440w.jpg)

## **DDPM代码实现**

网上有很多DDPM的实现，包括[论文中基于tensorflow的实现](https://link.zhihu.com/?target=https%3A//github.com/hojonathanho/diffusion)，还有[基于pytorch的实现](https://link.zhihu.com/?target=https%3A//github.com/xiaohu2015/nngen/blob/main/models/diffusion_models/ddpm_mnist.ipynb)，但是由于代码结构复杂，很难上手。为了便于理解以及快速运行，我们将代码合并在一个文件里面，基于tf2.5实现，直接copy过去就能运行。代码主要分为3个部分：DDPM前向和反向过程（都在GaussianDiffusion一个类里面实现）、模型训练过程、新图片生成过程。

- 前向反向过程代码（时间步为500）
    
    ```python
    import pandas as pd
    import numpy as np
    import os
    import numpy as np
    import sys
    import pandas as pd
    from numpy import arange
    import math
    import pyecharts
    import sys,base64,urllib,re
    import multiprocessing
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import ndcg_score
    import warnings 
    from optparse import OptionParser
    import logging
    import logging.config
    import time
    import tensorflow as tf
    from sklearn.preprocessing import normalize
    from tensorflow.keras.datasets import mnist
    from tensorflow.keras.layers import Dense, Dropout, Input
    from tensorflow.keras.models import Model, Sequential
    from tensorflow.keras.layers import LeakyReLU, Conv2D
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras import datasets
    from tensorflow import keras
    from tqdm import tqdm
    import numpy as np
    import matplotlib.pyplot as plt
    %matplotlib inline
    
    # beta schedule
    def linear_beta_schedule(timesteps):
        scale = 1000 / timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(beta_start, beta_end, timesteps, dtype=np.float64)
    
    def cosine_beta_schedule(timesteps, s=0.008):
        """
        cosine schedule
        as proposed in https://arxiv.org/abs/2102.09672
        """
        steps = timesteps + 1
        x = np.linspace(0, timesteps, steps, dtype=np.float64)
        alphas_cumprod = np.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return np.clip(betas, 0, 0.999)
    
    class GaussianDiffusion:
        def __init__(
            self,
            timesteps=1000,
            beta_schedule='linear'
        ):
            self.timesteps = timesteps
            
            if beta_schedule == 'linear':
                betas = linear_beta_schedule(timesteps)
            elif beta_schedule == 'cosine':
                betas = cosine_beta_schedule(timesteps)
            else:
                raise ValueError(f'unknown beta schedule {beta_schedule}')
                
            alphas = 1. - betas
            alphas_cumprod = np.cumprod(alphas, axis=0)
            alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
            
            self.betas = tf.constant(betas, dtype=tf.float32)
            self.alphas_cumprod = tf.constant(alphas_cumprod, dtype=tf.float32)
            self.alphas_cumprod_prev = tf.constant(alphas_cumprod_prev, dtype=tf.float32)
            
            # calculations for diffusion q(x_t | x_{t-1}) and others
            self.sqrt_alphas_cumprod = tf.constant(np.sqrt(self.alphas_cumprod), dtype=tf.float32)
            self.sqrt_one_minus_alphas_cumprod = tf.constant(np.sqrt(1.0 - self.alphas_cumprod), dtype=tf.float32)
            self.log_one_minus_alphas_cumprod = tf.constant(np.log(1. - alphas_cumprod), dtype=tf.float32)
            self.sqrt_recip_alphas_cumprod = tf.constant(np.sqrt(1. / alphas_cumprod), dtype=tf.float32)
            self.sqrt_recipm1_alphas_cumprod = tf.constant(np.sqrt(1. / alphas_cumprod - 1), dtype=tf.float32)
            
            # calculations for posterior q(x_{t-1} | x_t, x_0)
            self.posterior_variance = (
                betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
            )
            # below: log calculation clipped because the posterior variance is 0 at the beginning
            # of the diffusion chain
            self.posterior_log_variance_clipped = tf.constant(
                np.log(np.maximum(self.posterior_variance, 1e-20)), dtype=tf.float32)
            
            self.posterior_mean_coef1 = tf.constant(
                betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod), dtype=tf.float32)
            
            self.posterior_mean_coef2 = tf.constant(
                (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod), dtype=tf.float32)
        
        @staticmethod
        def _extract(a, t, x_shape):
            """
            Extract some coefficients at specified timesteps,
            then reshape to [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
            """
            bs, = t.shape
            assert x_shape[0] == bs
            out = tf.gather(a, t)
            assert out.shape == [bs]
            return tf.reshape(out, [bs] + ((len(x_shape) - 1) * [1]))
        
        # forward diffusion (using the nice property): q(x_t | x_0)
        def q_sample(self, x_start, t, noise=None):
            if noise is None:
                noise = tf.random.normal(shape=x_start.shape)
    
            sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
            sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
    
            return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
        
        # Get the mean and variance of q(x_t | x_0).
        def q_mean_variance(self, x_start, t):
            mean = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            variance = self._extract(1.0 - self.alphas_cumprod, t, x_start.shape)
            log_variance = self._extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
            return mean, variance, log_variance
        
        # Compute the mean and variance of the diffusion posterior: q(x_{t-1} | x_t, x_0)
        def q_posterior_mean_variance(self, x_start, x_t, t):
            posterior_mean = (
                self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
                + self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
            )
            posterior_variance = self._extract(self.posterior_variance, t, x_t.shape)
            posterior_log_variance_clipped = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)
            return posterior_mean, posterior_variance, posterior_log_variance_clipped
        
        # compute x_0 from x_t and pred noise: the reverse of `q_sample`
        def predict_start_from_noise(self, x_t, t, noise):
            return (
                self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        
        # compute predicted mean and variance of p(x_{t-1} | x_t)
        def p_mean_variance(self, model, x_t, t, clip_denoised=True):
            # predict noise using model
            pred_noise = model([x_t, t])
            # get the predicted x_0: different from the algorithm2 in the paper
            x_recon = self.predict_start_from_noise(x_t, t, pred_noise)
            if clip_denoised:
                x_recon = tf.clip_by_value(x_recon, -1., 1.)
            model_mean, posterior_variance, posterior_log_variance = \
                        self.q_posterior_mean_variance(x_recon, x_t, t)
            return model_mean, posterior_variance, posterior_log_variance
        
        def p_sample(self, model, x_t, t, clip_denoised=True):
            # predict mean and variance
            model_mean, _, model_log_variance = self.p_mean_variance(model, x_t, t,
                                                        clip_denoised=clip_denoised)
            noise = tf.random.normal(shape=x_t.shape)
            # no noise when t == 0
            nonzero_mask = tf.reshape(1 - tf.cast(tf.equal(t, 0), tf.float32), [x_t.shape[0]] + [1] * (len(x_t.shape) - 1))
            # compute x_{t-1}
            pred_img = model_mean + nonzero_mask * tf.exp(0.5 * model_log_variance) * noise
            return pred_img
        
        def p_sample_loop(self, model, shape):
            batch_size = shape[0]
            # start from pure noise (for each example in the batch)
            img = tf.random.normal(shape=shape)
            imgs = []
            for i in tqdm(reversed(range(0, self.timesteps)), desc='sampling loop time step', total=self.timesteps):
                img = self.p_sample(model, img, tf.fill([batch_size], i))
                imgs.append(img.numpy())
            return imgs
        
        def sample(self, model, image_size, batch_size=8, channels=3):
            return self.p_sample_loop(model, shape=[batch_size, image_size, image_size, channels])
        
        # compute train losses
        def train_losses(self, model, x_start, t):
            # generate random noise
            noise = tf.random.normal(shape=x_start.shape)
            # get x_t
            x_noisy = self.q_sample(x_start, t, noise=noise)
            model.train_on_batch([x_noisy, t], noise)
            predicted_noise = model([x_noisy, t])
            loss = model.loss(noise, predicted_noise)
            return loss
    
    # Load the dataset
    def load_data():
        (x_train, y_train), (_, _) = mnist.load_data()
        x_train = (x_train.astype(np.float32) - 127.5)/127.5
        return (x_train, y_train)
    
    print("forward diffusion: q(x_t | x_0)")
    timesteps = 500
    X_train, y_train = load_data()
    gaussian_diffusion = GaussianDiffusion(timesteps)
    plt.figure(figsize=(16, 8))
    x_start = X_train[7:8]
    for idx, t in enumerate([0, 50, 100, 200, 499]):
        x_noisy = gaussian_diffusion.q_sample(x_start, t=tf.convert_to_tensor([t]))
        x_noisy = x_noisy.numpy()
        x_noisy = x_noisy.reshape(28, 28)
        plt.subplot(1, 5, 1 + idx)
        plt.imshow(x_noisy, cmap="gray")
        plt.axis("off")
        plt.title(f"t={t}")
    ```
    
- 模型训练过程（时间步要和前向过程一致，为500步）（残差网络）
    
    ```python
    # ResNet model为例
    class ResNet(keras.layers.Layer):
        
        def __init__(self, in_channels, out_channels, name='ResNet', **kwargs):
            super(ResNet, self).__init__(name=name, **kwargs)
            self.in_channels = in_channels
            self.out_channels = out_channels
        
        def get_config(self):
            config = super(ResNet, self).get_config()
            config.update({'in_channels': self.in_channels, 'out_channels': self.out_channels})
            return config
        
        @classmethod
        def from_config(cls, config, custom_objects=None):
            return cls(**config)
        
        def build(self, input_shape):
            self.conv1 = Sequential([
                keras.layers.LeakyReLU(),
                keras.layers.Conv2D(filters=self.out_channels, kernel_size=3, padding='same')
            ])
            self.conv2 = Sequential([
                keras.layers.LeakyReLU(),
                keras.layers.Conv2D(filters=self.out_channels, kernel_size=3, padding='same', name='conv2')
            ])
    
        def call(self, inputs_all, dropout=None, **kwargs):
            """
            `x` has shape `[batch_size, height, width, in_dim]`
            """
            x, t = inputs_all
            h = self.conv1(x)
            h = self.conv2(h)
            h += x
            
            return h
    
    def build_DDPM(nn_model):
        nn_model.trainablea = True
        inputs = Input(shape=(28, 28, 1,))
        timesteps=Input(shape=(1,))
        outputs = nn_model([inputs, timesteps])
        ddpm = Model(inputs=[inputs, timesteps], outputs=outputs)
        ddpm.compile(loss=keras.losses.mse, optimizer=Adam(5e-4))
        return ddpm
    
    # train ddpm
    def train_ddpm(ddpm, gaussian_diffusion, epochs=1, batch_size=128, timesteps=500):
        
        #Loading the data
        X_train, y_train = load_data()
        step_cont = len(y_train) // batch_size
        
        step = 1
        for i in range(1, epochs + 1):
            for s in range(step_cont):
                if (s+1)*batch_size > len(y_train):
                    break
                images = X_train[s*batch_size:(s+1)*batch_size]
                images = tf.reshape(images, [-1, 28, 28 ,1])
                t = tf.random.uniform(shape=[batch_size], minval=0, maxval=timesteps, dtype=tf.int32)
                loss = gaussian_diffusion.train_losses(ddpm, images, t)
                if step == 1 or step % 100 == 0:
                    print("[step=%s]\tloss: %s" %(step, str(tf.reduce_mean(loss).numpy())))
                step += 1
    
    print("[ResNet] train ddpm")
    nn_model = ResNet(in_channels=1, out_channels=1)
    ddpm = build_DDPM(nn_model)
    gaussian_diffusion = GaussianDiffusion(timesteps=500)
    train_ddpm(ddpm, gaussian_diffusion, epochs=10, batch_size=64, timesteps=500)
    
    print("[ResNet] generate new images")
    generated_images = gaussian_diffusion.sample(ddpm, 28, batch_size=64, channels=1)
    fig = plt.figure(figsize=(12, 12), constrained_layout=True)
    gs = fig.add_gridspec(8, 8)
    
    imgs = generated_images[-1].reshape(8, 8, 28, 28)
    for n_row in range(8):
        for n_col in range(8):
            f_ax = fig.add_subplot(gs[n_row, n_col])
            f_ax.imshow((imgs[n_row, n_col]+1.0) * 255 / 2, cmap="gray")
            f_ax.axis("off")
    
    print("[ResNet] show the denoise steps")
    fig = plt.figure(figsize=(12, 12), constrained_layout=True)
    gs = fig.add_gridspec(16, 16)
    
    for n_row in range(16):
        for n_col in range(16):
            f_ax = fig.add_subplot(gs[n_row, n_col])
            t_idx = (timesteps // 16) * n_col if n_col < 15 else -1
            img = generated_images[t_idx][n_row].reshape(28, 28)
            f_ax.imshow((img+1.0) * 255 / 2, cmap="gray")
            f_ax.axis("off")
    ```
    

实际应用中一般是基于U-Net模型，模型结构如下：

![image.png](cv%EF%BC%9AGAN%E3%80%81DDPM%E3%80%81Diffusion%E3%80%81StableDiffusion%201bae64a5662180f69e6cef428d2d364a/image%2014.png)

- 模型训练（UNet版）
    
    ```python
    """
    U-Net model
    as proposed in https://arxiv.org/pdf/1505.04597v1.pdf
    """
    
    # use sinusoidal position embedding to encode time step (https://arxiv.org/abs/1706.03762)   
    def timestep_embedding(timesteps, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param timesteps: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an [N x dim] Tensor of positional embeddings.
        """
        half = dim // 2
        freqs = tf.exp(
            -math.log(max_period) * tf.experimental.numpy.arange(start=0, stop=half, step=1, dtype=tf.float32) / half
        )
        args = timesteps[:, ] * freqs
        embedding = tf.concat([tf.cos(args), tf.sin(args)], axis=-1)
        if dim % 2:
            embedding = tf.concat([embedding, tf.zeros_like(embedding[:, :1])], axis=-1)
        return embedding
    
    # upsample
    class Upsample(keras.layers.Layer):
        def __init__(self, channels, use_conv=False, name='Upsample', **kwargs):
            super(Upsample, self).__init__(name=name, **kwargs)
            self.use_conv = use_conv
            self.channels = channels
        
        def get_config(self):
            config = super(Upsample, self).get_config()
            config.update({'channels': self.channels, 'use_conv': self.use_conv})
            return config
        
        @classmethod
        def from_config(cls, config, custom_objects=None):
            return cls(**config)
        
        def build(self, input_shape):
            if self.use_conv:
                self.conv = keras.layers.Conv2D(filters=self.channels, kernel_size=3, padding='same')
    
        def call(self, inputs_all, dropout=None, **kwargs):
            x, t = inputs_all
            x = tf.image.resize_with_pad(x, target_height=x.shape[1]*2, target_width=x.shape[2]*2, method='nearest')
    #         if self.use_conv:
    #             x = self.conv(x)
            return x
    
    # downsample
    class Downsample(keras.layers.Layer):
        def __init__(self, channels, use_conv=True, name='Downsample', **kwargs):
            super(Downsample, self).__init__(name=name, **kwargs)
            self.use_conv = use_conv
            self.channels = channels
        
        def get_config(self):
            config = super(Downsample, self).get_config()
            config.update({'channels': self.channels, 'use_conv': self.use_conv})
            return config
        
        @classmethod
        def from_config(cls, config, custom_objects=None):
            return cls(**config)
        
        def build(self, input_shape):
            if self.use_conv:
                self.op = keras.layers.Conv2D(filters=self.channels, kernel_size=3, strides=2, padding='same')
            else:
                self.op = keras.layers.AveragePooling2D(strides=(2, 2))
    
        def call(self, inputs_all, dropout=None, **kwargs):
            x, t = inputs_all
            return self.op(x)
    
    # Residual block
    class ResidualBlock(keras.layers.Layer):
        
        def __init__(
            self, 
            in_channels, 
            out_channels, 
            time_channels, 
            use_time_emb=True,
            name='residul_block', **kwargs
        ):
            super(ResidualBlock, self).__init__(name=name, **kwargs)
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.time_channels = time_channels
            self.use_time_emb = use_time_emb
        
        def get_config(self):
            config = super(ResidualBlock, self).get_config()
            config.update({
                'time_channels': self.time_channels, 
                'in_channels': self.in_channels, 
                'out_channels': self.out_channels,
                'use_time_emb': self.use_time_emb
            })
            return config
        
        @classmethod
        def from_config(cls, config, custom_objects=None):
            return cls(**config)
        
        def build(self, input_shape):
            self.dense_ = keras.layers.Dense(units=self.out_channels, activation=None)
            self.dense_short = keras.layers.Dense(units=self.out_channels, activation=None)
            
            self.conv1 = [
                keras.layers.LeakyReLU(),
                keras.layers.Conv2D(filters=self.out_channels, kernel_size=3, padding='same')
            ]
            self.conv2 = [
                keras.layers.LeakyReLU(),
                keras.layers.Conv2D(filters=self.out_channels, kernel_size=3, padding='same', name='conv2')
            ]
            self.conv3 = [
                keras.layers.LeakyReLU(),
                keras.layers.Conv2D(filters=self.out_channels, kernel_size=1, name='conv3')
            ]
            
            self.activate = keras.layers.LeakyReLU()
    
        def call(self, inputs_all, dropout=None, **kwargs):
            """
            `x` has shape `[batch_size, height, width, in_dim]`
            `t` has shape `[batch_size, time_dim]`
            """
            x, t = inputs_all
            h = x
            for module in self.conv1:
                h = module(x)
            
            # Add time step embeddings
            if self.use_time_emb:
                time_emb = self.dense_(self.activate(t))[:, None, None, :]
                h += time_emb
            for module in self.conv2:
                h = module(h)
            
            if self.in_channels != self.out_channels:
                for module in self.conv3:
                    x = module(x)
                return h + x
            else:
                return h + x
    
    # Attention block with shortcut
    class AttentionBlock(keras.layers.Layer):
        
        def __init__(self, channels, num_heads=1, name='attention_block', **kwargs):
            super(AttentionBlock, self).__init__(name=name, **kwargs)
            self.channels = channels
            self.num_heads = num_heads
            self.dense_layers = []
            
        def get_config(self):
            config = super(AttentionBlock, self).get_config()
            config.update({'channels': self.channels, 'num_heads': self.num_heads})
            return config
        
        @classmethod
        def from_config(cls, config, custom_objects=None):
            return cls(**config)
        
        def build(self, input_shape):
            for i in range(3):
                dense_ = keras.layers.Conv2D(filters=self.channels, kernel_size=1)
                self.dense_layers.append(dense_)
            self.proj = keras.layers.Conv2D(filters=self.channels, kernel_size=1)
        
        def call(self, inputs_all, dropout=None, **kwargs):
            inputs, t = inputs_all
            H = inputs.shape[1]
            W = inputs.shape[2]
            C = inputs.shape[3]
            qkv = inputs
            q = self.dense_layers[0](qkv)
            k = self.dense_layers[1](qkv)
            v = self.dense_layers[2](qkv)
            attn = tf.einsum("bhwc,bHWc->bhwHW", q, k)* (int(C) ** (-0.5))
            attn = tf.reshape(attn, [-1, H, W, H * W])
            attn = tf.nn.softmax(attn, axis=-1)
            attn = tf.reshape(attn, [-1, H, W, H, W])
            
            h = tf.einsum('bhwHW,bHWc->bhwc', attn, v)
            h = self.proj(h)
            
            return h + inputs
    
    # upsample
    class UNetModel(keras.layers.Layer):
        def __init__(
            self,
            in_channels=3,
            model_channels=128,
            out_channels=3,
            num_res_blocks=2,
            attention_resolutions=(8, 16),
            dropout=0,
            channel_mult=(1, 2, 2, 2),
            conv_resample=True,
            num_heads=4,
            name='UNetModel',
            **kwargs
        ):
            super(UNetModel, self).__init__(name=name, **kwargs)
            self.in_channels = in_channels
            self.model_channels = model_channels
            self.out_channels = out_channels
            self.num_res_blocks = num_res_blocks
            self.attention_resolutions = attention_resolutions
            self.dropout = dropout
            self.channel_mult = channel_mult
            self.conv_resample = conv_resample
            self.num_heads = num_heads
            self.time_embed_dim = self.model_channels * 4
        
        def build(self, input_shape):
            
            # time embedding
            self.time_embed = [
                keras.layers.Dense(self.time_embed_dim, activation=None),
                keras.layers.LeakyReLU(),
                keras.layers.Dense(self.time_embed_dim, activation=None)
            ]
            
            # down blocks
            self.conv = keras.layers.Conv2D(filters=self.model_channels, kernel_size=3, padding='same')
            self.down_blocks = []
            down_block_chans = [self.model_channels]
            ch = self.model_channels
            ds = 1
            index = 0
            for level, mult in enumerate(self.channel_mult):
                for _ in range(self.num_res_blocks):
                    
                    layers = [
                        ResidualBlock(
                            in_channels=ch, 
                            out_channels=mult * self.model_channels, 
                            time_channels=self.time_embed_dim,
                            name='resnet_'+str(index)
                        )
                    ]
                    index += 1
                    ch = mult * self.model_channels
                    if ds in self.attention_resolutions:
                        layers.append(AttentionBlock(ch, num_heads=self.num_heads))
                    self.down_blocks.append(layers)
                    down_block_chans.append(ch)
            
                if level != len(self.channel_mult) - 1: # don't use downsample for the last stage
                    self.down_blocks.append(Downsample(ch, self.conv_resample))
                    down_block_chans.append(ch)
                    ds *= 2
                    
            # middle block
            self.middle_block = [
                ResidualBlock(ch, ch, self.time_embed_dim, name='res1'),
                AttentionBlock(ch, num_heads=self.num_heads),
                ResidualBlock(ch, ch, self.time_embed_dim, name='res2')
            ]
            
            # up blocks
            self.up_blocks = []
            index = 0
            for level, mult in list(enumerate(self.channel_mult))[::-1]:
                for i in range(self.num_res_blocks + 1):
                    layers = []
                    layers.append(
                        ResidualBlock(
                            in_channels=ch + down_block_chans.pop(), 
                            out_channels=self.model_channels * mult, 
                            time_channels=self.time_embed_dim,
                            name='up_resnet_'+str(index)
                        )
                    )
                    
                    layer_num = 1
                    ch = self.model_channels * mult
                    if ds in self.attention_resolutions:
                        layers.append(AttentionBlock(ch, num_heads=self.num_heads))
                    if level and i == self.num_res_blocks:
                        layers.append(Upsample(ch, self.conv_resample))
                        ds //= 2
                    self.up_blocks.append(layers)
                    
                    index += 1
                
            
            self.out = Sequential([
                keras.layers.LeakyReLU(),
                keras.layers.Conv2D(filters=self.out_channels, kernel_size=3, padding='same')
            ])
    
        def call(self, inputs, dropout=None, **kwargs):
            """
            Apply the model to an input batch.
            :param x: an [N x H x W x C] Tensor of inputs. N, H, W, C
            :param timesteps: a 1-D batch of timesteps.
            :return: an [N x C x ...] Tensor of outputs.
            """
            x, timesteps = inputs
            hs = []
            
            # time step embedding
            emb = timestep_embedding(timesteps, self.model_channels)
            for module in self.time_embed:
                emb = module(emb)
            
            # down stage
            h = x
            h = self.conv(h)
            hs = [h]
            for module_list in self.down_blocks:
                if isinstance(module_list, list):
                    for module in module_list:
                        h = module([h, emb])
                else:
                    h = module_list([h, emb])
                hs.append(h)
                
            # middle stage
            for module in self.middle_block:
                h = module([h, emb])
            
            # up stage
            for module_list in self.up_blocks:
                cat_in = tf.concat([h, hs.pop()], axis=-1)
                h = cat_in
                for module in module_list:
                    h = module([h, emb])
            
            return self.out(h)
    
    print("[U-Net] train ddpm")
    nn_model = UNetModel(
        in_channels=1,
        model_channels=96,
        out_channels=1,
        channel_mult=(1, 2, 2),
        attention_resolutions=[]
    )
    ddpm = build_DDPM(nn_model)
    gaussian_diffusion = GaussianDiffusion(timesteps=500)
    train_ddpm(ddpm, gaussian_diffusion, epochs=10, batch_size=64, timesteps=500)
    
    print("[U-Net] generate new images")
    generated_images = gaussian_diffusion.sample(ddpm, 28, batch_size=64, channels=1)
    fig = plt.figure(figsize=(12, 12), constrained_layout=True)
    gs = fig.add_gridspec(8, 8)
    
    imgs = generated_images[-1].reshape(8, 8, 28, 28)
    for n_row in range(8):
        for n_col in range(8):
            f_ax = fig.add_subplot(gs[n_row, n_col])
            f_ax.imshow((imgs[n_row, n_col]+1.0) * 255 / 2, cmap="gray")
            f_ax.axis("off")
    
    print("[U-Net] show the denoise steps")
    fig = plt.figure(figsize=(12, 12), constrained_layout=True)
    gs = fig.add_gridspec(16, 16)
    
    for n_row in range(16):
        for n_col in range(16):
            f_ax = fig.add_subplot(gs[n_row, n_col])
            t_idx = (timesteps // 16) * n_col if n_col < 15 else -1
            img = generated_images[t_idx][n_row].reshape(28, 28)
            f_ax.imshow((img+1.0) * 255 / 2, cmap="gray")
            f_ax.axis("off")
    ```
    

# Diffusion

https://zhuanlan.zhihu.com/p/599887666：**十分钟读懂Diffusion：图解Diffusion扩散模型**

文生图全过程

## **1. Diffusion文字生成图片——整体结构**

### **1.1 整个生成过程**

我们知道在使用 Diffusion 的时候，是通过文字生成图片，但是上一篇文章中讲的Diffusion模型输入只有随机高斯噪声和time step。那么文字是怎么转换成Diffusion的输入的呢？加入文字后 Diffusion 又有哪些改变？下图可以找到答案。

![image.png](cv%EF%BC%9AGAN%E3%80%81DDPM%E3%80%81Diffusion%E3%80%81StableDiffusion%201bae64a5662180f69e6cef428d2d364a/image%2015.png)

实际上 Diffusion 是**使用Text Encoder生成文字对应的（图像）embedding（Text Encoder使用[CLIP模型](https://link.zhihu.com/?target=https%3A//openai.com/blog/clip/)），然后和随机噪声embedding，time step embedding一起作为Diffusion的输入**，最后生成理想的图片。我们看一下完整的图：

![image.png](cv%EF%BC%9AGAN%E3%80%81DDPM%E3%80%81Diffusion%E3%80%81StableDiffusion%201bae64a5662180f69e6cef428d2d364a/image%2016.png)

上图我们看到了Diffusion的输入为token embedding和随机embedding，time embedding没有画出来。中间的Image Information Creator是由多个[UNet](https://zhida.zhihu.com/search?content_id=221441160&content_type=Article&match_order=1&q=UNet&zhida_source=entity)模型组成，更详细的图如下：

![image.png](cv%EF%BC%9AGAN%E3%80%81DDPM%E3%80%81Diffusion%E3%80%81StableDiffusion%201bae64a5662180f69e6cef428d2d364a/image%2017.png)

可以看到中间的**Image Information Creator是由多个UNet组合而成的**，关于UNet的结构我们放在后面来讲。现在我们了解了加入文字embedding后 Diffusion 的结构，那么文字的embedding是如何生成的？接下来我们介绍下如何使用CLIP模型生成文字embedding。

### **1.2 使用CLIP模型生成输入文字embedding**

CLIP 在图像及其描述的数据集上进行训练。想象一个看起来像这样的数据集，包含4 亿张图片及其说明：

![image.png](cv%EF%BC%9AGAN%E3%80%81DDPM%E3%80%81Diffusion%E3%80%81StableDiffusion%201bae64a5662180f69e6cef428d2d364a/image%2018.png)

实际上CLIP是根据从网络上抓取的图像及其文字说明进行训练的。CLIP 是图像编码器和文本编码器的组合，它的**训练过程可以简化为给图片加上文字说明**。首先分别使用图像和文本编码器对它们进行编码。

![image.png](cv%EF%BC%9AGAN%E3%80%81DDPM%E3%80%81Diffusion%E3%80%81StableDiffusion%201bae64a5662180f69e6cef428d2d364a/image%2019.png)

然后使用余弦相似度刻画是否匹配。最开始训练时，相似度会很低。

![image.png](cv%EF%BC%9AGAN%E3%80%81DDPM%E3%80%81Diffusion%E3%80%81StableDiffusion%201bae64a5662180f69e6cef428d2d364a/image%2020.png)

然后计算loss，更新模型参数，得到新的图片embedding和文字embedding

![image.png](cv%EF%BC%9AGAN%E3%80%81DDPM%E3%80%81Diffusion%E3%80%81StableDiffusion%201bae64a5662180f69e6cef428d2d364a/image%2021.png)

通过在训练集上训练模型，最终得到文字的embedding和图片的embedding。有关CLIP模型的细节，[可以参考对应的论文](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/2103.00020.pdf)。

### **1.3 UNet网络中如何使用文字embedding**

前面已经介绍了如何生成输入文字embedding，那么UNet网络又是如何使用的？实际上是**在UNet的每个ResNet之间添加一个Attention，而Attention一端的输入便是文字embedding**。如下图所示。

![image.png](cv%EF%BC%9AGAN%E3%80%81DDPM%E3%80%81Diffusion%E3%80%81StableDiffusion%201bae64a5662180f69e6cef428d2d364a/image%2022.png)

![image.png](cv%EF%BC%9AGAN%E3%80%81DDPM%E3%80%81Diffusion%E3%80%81StableDiffusion%201bae64a5662180f69e6cef428d2d364a/image%2023.png)

## **2. 扩散模型Diffusion**

### **2.1 扩散模型Duffison的训练过程**

![](https://pic1.zhimg.com/v2-494de55e4ea0bb8baa6e4c203bf395b8_1440w.jpg)

Diffusion模型的训练可以分为两个部分：

1. 前向扩散过程（Forward Diffusion Process）→  图片中添加噪声
2. 反向扩散过程（Reverse Diffusion Process）→  去除图片中的噪声

### **2.4 训练过程**

![](https://pic4.zhimg.com/v2-4614a863afc890fc6ba2d31f9d2628c5_1440w.jpg)

在每一轮的训练过程中，包含以下内容：

1. **每一个训练样本选择一个随机时间步长t**
2. 将time step t 对应的高斯噪声应用到图片中
3. **将time step转化为对应embedding**

下面是每一轮详细的训练过程：

![](https://pica.zhimg.com/v2-9f358d9e8916c275ac7110e4818ab62e_1440w.jpg)

### **2.5 从高斯噪声中生成原始图片（反向扩散过程）**

![](https://pic2.zhimg.com/v2-fdd4ec3aff3c629cd826c845406e9abb_1440w.jpg)

上图的Sample a Gaussian表示生成随机高斯噪声，Iteratively denoise the image表示反向扩散过程，如何一步步从高斯噪声变成输出图片。可以看到最终生成的Denoised image非常清晰。

## **补充1：UNet模型结构**

前面已经介绍了Diffusion的整个过程，这里补充以下UNet的模型结构，如下图所示

![](https://pic1.zhimg.com/v2-2da70c77fc6cf287fe17cf89e52d9bc0_1440w.jpg)

这里面Downsampe、Middle block、Upsample中都包含了ResNet残差网络。

- unet
    
    https://cloud.tencent.com/developer/article/2040972
    
    ![image.png](cv%EF%BC%9AGAN%E3%80%81DDPM%E3%80%81Diffusion%E3%80%81StableDiffusion%201bae64a5662180f69e6cef428d2d364a/image%2024.png)
    
- 下采样和上采样
    
    在深度学习中，**下采样（Downsampling）**和**上采样（Upsampling）**是两种常见的操作，用于改变数据（如图像、特征图等）的空间分辨率或数据量。它们在不同的应用场景中发挥着重要作用，如图像分类、目标检测、图像分割等。以下是对这两种操作的详细解释：
    
    ### 1. 下采样（Downsampling）
    
    ### 定义：
    
    下采样是指通过某种方法减少数据在空间维度上的分辨率或尺寸。例如，将一张高分辨率的图像转换为低分辨率的图像。
    
    ### 主要目的：
    
    - **减少计算量**：通过降低数据的空间分辨率，可以减少后续处理所需的计算量。
    - **增加感受野**：下采样操作可以增加模型的感受野，使其能够捕捉更大范围的上下文信息。
    - **特征提取**：通过下采样，可以提取数据中的高层次特征，去除一些冗余信息。
    
    ### 常见的下采样方法：
    
    1. **池化（Pooling）**：
        - **最大池化（Max Pooling）**：在局部区域中选择最大值作为输出。
        - **平均池化（Average Pooling）**：在局部区域中计算平均值作为输出。
        - **全局池化（Global Pooling）**：将整个特征图池化为一个值。
    2. **卷积（Convolution）**：
        - 使用步幅（stride）大于1的卷积操作，可以实现下采样。例如，步幅为2的卷积会将特征图的尺寸缩小一半。
    3. **插值（Interpolation）**：
        - 使用插值方法（如双线性插值）来降低图像的分辨率。
    
    ### 示例：
    
    假设有一个4x4的特征图，使用2x2的最大池化，步幅为2，可以得到一个2x2的特征图。
    
    ### 2. 上采样（Upsampling）
    
    ### 定义：
    
    上采样是指通过某种方法增加数据在空间维度上的分辨率或尺寸。例如，将一张低分辨率的图像转换为高分辨率的图像。
    
    ### 主要目的：
    
    - **恢复分辨率**：在需要高分辨率输出的任务中，如图像分割、图像生成等，上采样可以恢复数据的原始分辨率。
    - **生成细节**：通过上采样，可以生成更多的细节信息，从而提高输出质量。
    
    ### 常见的上采样方法：
    
    1. **插值（Interpolation）**：
        - **最近邻插值（Nearest Neighbor Interpolation）**：复制最近的像素值。
        - **双线性插值（Bilinear Interpolation）**：在水平和垂直方向上分别进行线性插值。
        - **双三次插值（Bicubic Interpolation）**：使用更复杂的插值函数，生成更平滑的结果。
    2. **转置卷积（Transposed Convolution）**：
        - 也称为反卷积（Deconvolution），通过学习卷积核来增加特征图的尺寸。
        - 转置卷积可以学习到更复杂的上采样模式，适用于需要学习参数的上采样任务。
    3. **亚像素卷积（Sub-pixel Convolution）**：
        - 通过重新排列像素来实现上采样。例如，将一个低分辨率的特征图通过重新排列像素转换为高分辨率的特征图。
    4. **上池化（Unpooling）**：
        - 与池化相反，上池化通过某种方式恢复池化前的空间分辨率。例如，记录池化时的最大值的索引，在上池化时将这些值放回原位，其他位置填充0。
    
    ### 示例：
    
    假设有一个2x2的特征图，使用双线性插值，可以将其上采样为4x4的特征图。
    
    ### 3. 下采样和上采样的应用
    
    ### a. 图像分割
    
    在图像分割任务中，通常会使用下采样来提取高层次特征，然后使用上采样来恢复原始分辨率，从而实现像素级的分类。
    
    ### b. 图像生成
    
    在图像生成任务中，如生成对抗网络（GAN），上采样用于将低维的潜在向量转换为高分辨率的图像。
    
    ### c. 目标检测
    
    在目标检测任务中，下采样用于减少计算量并提取特征，而上采样可以用于恢复目标的精确位置。
    
    ### 4. 总结
    
    - **下采样**：减少数据的空间分辨率或尺寸，常用于特征提取和减少计算量。
    - **上采样**：增加数据的空间分辨率或尺寸，常用于恢复分辨率和生成细节。
    - **方法**：下采样常用的方法包括池化、步幅卷积等；上采样常用的方法包括插值、转置卷积、亚像素卷积等。
    
    通过合理地使用下采样和上采样，可以有效地平衡计算效率和模型性能，从而实现更好的深度学习模型。
    

## **补充2：Diffusion模型的缺点及改进版——[Stable Diffusion](https://zhida.zhihu.com/search?content_id=221441160&content_type=Article&match_order=1&q=Stable+Diffusion&zhida_source=entity)**

前面我们在介绍整个**文字生成图片**的架构中，图里面用的都是**Stable Diffusion**，后面介绍又主要介绍的是Diffusion。其实Stable Diffusion是Diffusion的改进版。

Diffusion的**缺点**是在反向扩散过程中需要把完整尺寸的图片输入到U-Net，这使得当图片尺寸以及time step t足够大时，Diffusion会非常的慢。Stable Diffusion就是为了解决这一问题而提出的。后面有时间再介绍下Stable Diffusion是如何改进的。

## **补充4：DDPM为什么要引入时间步长*t***

**引入时间步长 t 是为了模拟一个随时间逐渐增强的扰动过程**。**每个时间步长 t 代表一个扰动过程**，从初始状态开始，通过多次应用噪声来逐渐改变图像的分布。因此，**较小的 t 代表较弱的噪声扰动，而较大的 t 代表更强的噪声扰动。**

这里还有一个原因，**DDPM 中的 UNet 都是共享参数的，那如何根据不同的输入生成不同的输出，最后从一个完全的一个随机噪声变成一个有意义的图片，这还是一个非常难的问题**。我们希望这个 UNet 模型在刚开始的反向过程之中，它可以先生成一些物体的大体轮廓，随着扩散模型一点一点往前走，然后到最后快生成逼真图像的时候，这时候希望它学习到高频的一些特征信息。**由于 UNet 都是共享参数，这时候就需要 time embedding 去提醒这个模型，我们现在走到哪一步了，现在输出是想要粗糙一点的，还是细致一点的。**

所以加入时间步长 t 对生成和采样过程都有帮助。

## **补充5：为什么训练过程中每一次引入的是随机时间步长 *t***

我们知道模型**在训练过程中 loss 会逐渐降低，越到后面 loss 的变化幅度越小。如果时间步长 t 是递增的，那么必然会使得模型过多的关注较早的时间步长（因为早期 loss 大），而忽略了较晚的时间步长信息。**

# Stable diffusion

https://zhuanlan.zhihu.com/p/600251419：**十分钟读懂Stable Diffusion**

由于Diffusion在反向扩散过程中需要把完整尺寸的图片输入到[U-Net](https://zhida.zhihu.com/search?content_id=221521797&content_type=Article&match_order=1&q=U-Net&zhida_source=entity)，使得速度非常慢，因此目前应用最广的并不是Diffusion，而实其改进版Stable Diffusion。

接下来我们介绍Stable Diffusion是如何根据文字生成图像的，相比Diffusion做了哪些优化。

**写在最前面：**

由于Stable Diffusion里面有关扩散过程的描述，描述方法有很多版本，比如前向过程也可以叫加噪过程，为了便于理解，这里把各种描述统一说明一下。

## **1. Stable Diffusion文字生成图片过程**

Stable Diffusion其实是Diffusion的改进版本，**主要是为了解决Diffusion的速度问题**。那么Stable Diffusion是如何根据文字得出图片的呢？下图是Stable Diffusion生成图片的具体过程：

![image.png](cv%EF%BC%9AGAN%E3%80%81DDPM%E3%80%81Diffusion%E3%80%81StableDiffusion%201bae64a5662180f69e6cef428d2d364a/image%2025.png)

可以看到，对于输入的文字（图中的“An astronout riding a horse”）会经过一个[CLIP模型](https://zhida.zhihu.com/search?content_id=221521797&content_type=Article&match_order=1&q=CLIP%E6%A8%A1%E5%9E%8B&zhida_source=entity)转化为text embedding，然后和初始图像（初始化使用随机高斯噪声Gaussian Noise）一起输入去噪模块（也就是图中Text conditioned latent U-Net），最后输出 512×512 大小的图片。在文章（[绝密伏击：十分钟读懂Diffusion：图解Diffusion扩散模型](https://zhuanlan.zhihu.com/p/599887666)）中，我们已经知道了CLIP模型和U-Net模型的大致原理，这里面关键是**Text conditioned latent U-net**，翻译过来就是**文本条件隐U-net网络**，其实是**通过对U-Net引入[多头Attention](https://zhida.zhihu.com/search?content_id=221521797&content_type=Article&match_order=1&q=%E5%A4%9A%E5%A4%B4Attention&zhida_source=entity)机制，使得输入文本和图像相关联**，后面我们重点讲讲这一块是怎么做的。

## **2. Stable Diffusion的改进一：图像压缩**

Stable Diffusion原来的名字叫“**Latent Diffusion Model**”（**LDM**），很明显就是**扩散过程发生隐空间中（latent space），**其实就是对图片做了压缩，这也是Stable Diffusion比Diffusion速度快的原因。

![image.png](cv%EF%BC%9AGAN%E3%80%81DDPM%E3%80%81Diffusion%E3%80%81StableDiffusion%201bae64a5662180f69e6cef428d2d364a/image%2026.png)

Stable Diffusion会**先训练一个自编码器，来学习将图像压缩成低维表示**。

- 通过训练好的编码器 E，可以将原始大小的图像压缩成低维的latent data（图像压缩）
- 通过训练好的解码器 D，可以将latent data还原为原始大小的图像

**在将图像压缩成latent data后，便可以在latent space中完成扩散过程**，对比下和Diffusion扩散过程的区别，如下图所示：

![image.png](cv%EF%BC%9AGAN%E3%80%81DDPM%E3%80%81Diffusion%E3%80%81StableDiffusion%201bae64a5662180f69e6cef428d2d364a/image%2027.png)

可以看到Diffusion扩散模型就是在原图 x 上进行的操作，而Stale Diffusion是在压缩后的图像 z 上进行操作。

Stable Diffusion的前向扩散过程和Diffusion扩散模型基本没啥区别，只是多了一个图像压缩，只是反向扩散过程二者之前还是有区别。

## **3. Stable Diffusion的改进二：反向扩散过程**

在第一节我们已经简单介绍过Stable Diffusion文字生成图片的过程，这里我们扩展下，看一下里面的细节，如下图所示：

![image.png](cv%EF%BC%9AGAN%E3%80%81DDPM%E3%80%81Diffusion%E3%80%81StableDiffusion%201bae64a5662180f69e6cef428d2d364a/image%2028.png)

![image.png](cv%EF%BC%9AGAN%E3%80%81DDPM%E3%80%81Diffusion%E3%80%81StableDiffusion%201bae64a5662180f69e6cef428d2d364a/image%2029.png)

**Stable Diffusion在反向扩散过程中其实谈不上改进，只是支持了文本的输入**，对U-Net的结构做了修改，**使得每一轮去噪过程中文本和图像相关联**。在上一篇文章（[绝密伏击：十分钟读懂Diffusion：图解Diffusion扩散模型](https://zhuanlan.zhihu.com/p/599887666)）中，我们在介绍使用Diffusion扩散模型生成图像时，一开始就已经介绍了在扩散过程中如何支持文本输入，以及如何修改U-Net结构，只是介绍U-Net结构改进的时候，讲的比较粗，感兴趣的可以去看看里面的第一节。下面我们就补充介绍下Stable Diffusion是如何对U-Net结构做修改，从而更好的处理输入文本。

### **3.1 反向扩散细节：单轮去噪U-Net引入多头Attention（改进U-Net结构）**

我们先看一下反向扩散的整体结构，如下图所示：

![image.png](cv%EF%BC%9AGAN%E3%80%81DDPM%E3%80%81Diffusion%E3%80%81StableDiffusion%201bae64a5662180f69e6cef428d2d364a/image%2030.png)

![image.png](cv%EF%BC%9AGAN%E3%80%81DDPM%E3%80%81Diffusion%E3%80%81StableDiffusion%201bae64a5662180f69e6cef428d2d364a/image%2031.png)

![image.png](cv%EF%BC%9AGAN%E3%80%81DDPM%E3%80%81Diffusion%E3%80%81StableDiffusion%201bae64a5662180f69e6cef428d2d364a/image%2032.png)

上图的最左边里面的Semantic Map、Text、Representations、Images稍微不好理解，这是Stable Diffusion处理不同任务的通用框架：

- Semantic Map：表示处理的是通过语义生成图像的任务
- **Text：表示的就是文字生成图像的任务**
- Representations：表示的是通过语言描述生成图像
- Images：表示的是根据图像生成图像

这里我们只考虑输入是Text，因此首先会通过模型CLIP模型生成文本向量，然后输入到U-Net网络中的多头Attention(Q, K, V)。

这里补充一下多头Attention(Q, K, V)是怎么工作的，我们就以右边的第一个Attention(Q, K, V)为例。

![image.png](cv%EF%BC%9AGAN%E3%80%81DDPM%E3%80%81Diffusion%E3%80%81StableDiffusion%201bae64a5662180f69e6cef428d2d364a/image%2033.png)

【注意，是图像（随机隐噪声）作为Query】

## **Stable Diffusion完整结构**

最后我们来看一下Stable Diffusion完整结构，包含文本向量表示、初始图像（随机高斯噪声）、时间embedding，如下图所示：

![image.png](cv%EF%BC%9AGAN%E3%80%81DDPM%E3%80%81Diffusion%E3%80%81StableDiffusion%201bae64a5662180f69e6cef428d2d364a/image%2034.png)

上图详细的展示了Stable Diffusion前向扩散和反向扩散的整个过程，我们再看一下不处理输入文字，只是单纯生成任意图像的Diffusion结构。

![](https://pic1.zhimg.com/v2-cb239c4e92c6c6ebaf336ef48606af86_1440w.jpg)

不输入文字，单纯生成任意图像的Diffusion结构

可以看到，不处理文字输入，生成任意图像的Diffusion模型，和Stable Diffusion相比，主要有两个地方不一样：

- 少了对输入文字的embedding过程（少了编码器 E、解码器 D ）
- U-Net网络少了多头Attention结构

https://blog.csdn.net/zzz777qqq/article/details/144930229

Stable Diffusion是一种扩散模型（diffusion model）的变体，叫做“潜在扩散模型”（latent diffusion model;
LDM）。扩散模型是在2015年推出的，其目的是消除对训练图像的连续应用高斯噪声，可以将其视为一系列去噪自编码器。Stable
Diffusion由3个部分组成：变分自编码器（VAE）、U-Net和一个文本编码器。与其学习去噪图像数据（在“像素空间”中），而是训练VAE将图像转换为低维潜在空间。添加和去除高斯噪声的过程被应用于这个潜在表示，然后将最终的去噪输出解码到像素空间中。在前向扩散过程中，高斯噪声被迭代地应用于压缩的潜在表征。每个去噪步骤都由一个包含ResNet骨干的U-
Net架构完成，通过从前向扩散往反方向去噪而获得潜在表征。最后，VAE解码器通过将表征转换回像素空间来生成输出图像。研究人员指出，降低训练和生成的计算要求是LDM的一个优势。

去噪步骤可以以文本串、图像或一些其他数据为条件。调节数据的编码通过交叉注意机制（cross-attention mechanism）暴露给去噪U-Net的架构。为了对文本进行调节，一个预训练的固定CLIP ViT-L/14文本编码器被用来将提示词转化为嵌入空间。

![image.png](cv%EF%BC%9AGAN%E3%80%81DDPM%E3%80%81Diffusion%E3%80%81StableDiffusion%201bae64a5662180f69e6cef428d2d364a/image%2035.png)

### **文本引导图像：attention机制如何起作用**

https://zhuanlan.zhihu.com/p/696562930：**Stable Diffusion 的 UNet 和 Attention**

![image.png](cv%EF%BC%9AGAN%E3%80%81DDPM%E3%80%81Diffusion%E3%80%81StableDiffusion%201bae64a5662180f69e6cef428d2d364a/image%2036.png)

方便起见，我们叫左侧为编码阶段，右侧为解码阶段，中间为中间阶段。

![image.png](cv%EF%BC%9AGAN%E3%80%81DDPM%E3%80%81Diffusion%E3%80%81StableDiffusion%201bae64a5662180f69e6cef428d2d364a/image%2037.png)

在 UNet 的每一个编码块、解码块、中间阶段，都有 context 参与运算。每个块都是`TimestepEmbedSequential` 实例。 那么 context 怎么参与运算呢。

对于每个块，都是由 `ResBlock`，`Downsample`，`Upsample` 或者 `SpatialTransformer` 组成的，这四个模块和 context 的运算遵从不同的方式：

- `ResBlock`：`TimestepBlock` 的继承类， 将 **context 通过线性层映射到输入特征相同的维度，然后和输入特征相加**，重点在相加；
- `Upsample`，`Downsample`：直接忽略context
- `SpatialTransformer`：**和 context 做交叉注意力**，重点在交叉注意力

我们主要关注交叉注意力的实现。

![image.png](cv%EF%BC%9AGAN%E3%80%81DDPM%E3%80%81Diffusion%E3%80%81StableDiffusion%201bae64a5662180f69e6cef428d2d364a/image%2038.png)

其中对 `x` 的自注意力计算如下图：

![](https://pic4.zhimg.com/v2-393ccfbfc2f87d88f3fe9a12c69098d3_1440w.jpg)

SpatialTransformer 的自注意力

`x` 和 `context` 的交叉注意力计算如下图：

![](https://pic4.zhimg.com/v2-11904d19fb2c8c2bb9446e6fcc1c3513_1440w.jpg)

对于 C×H×W 的特征图，和 L×C′ 的 context，得到的交叉注意力 map 维度为 HW×L 的，**每一行表示一个特征像素对 context 中每个单词的注意力**，或者说，每一列代表 context 中的一个单词对所有的特征像素的注意力。（C，C’都是channel，通道数）

此前的研究发现，在 SD 生成图像的过程中，context 中的特定单词对特征图的交叉注意力 map 和生成的图像有密切的关系。