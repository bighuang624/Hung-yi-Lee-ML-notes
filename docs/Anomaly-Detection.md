本节中我们将学习**异常检测（Anomaly Detection）**。

## 问题定义

给定一个训练数据的集合$\left\\{x^{1}, x^{2}, \cdots, x^{N}\right\\}$，我们希望找到一个函数，能够判断新的输入和已有的训练数据是否相似。如果相似，这个函数（我们现在可以将其称为异常检测器）将其判断为正常数据，否则将其判断为异常。

虽然我们用异常（anomaly）这个词来表示我们的检测目标，但实际上异常检测不一定是检测不好的东西，只是与训练数据不相似。因此，有时候我们也会使用 outlier，novelty 或者 exceptions 等词来替代 anomaly。

## 异常检测问题的分类

怎么做异常检测？一个符合直觉的想法是，我们有一组正常数据和一组异常数据，我们只需要训练一个二元分类器即可。但是问题没有这么简单，原因在于异常数据是与我们的正常数据不同的数据，因此它包含的范围太广，以至于我们无法收集齐异常数据中的所有种类。另外，正常的数据比较容易搜集，而异常的数据比较难搜集。根据上述原因，异常检测不是一个单纯的二元分类的问题，而是一个独立的研究主题。

在这里，我们将异常检测问题简单分为两类。第一类中，我们的训练数据同时包含标签（这些标签涵盖正常数据中的不同类别，不包含 "unknown"），我们就可以用这些数据集训练一个分类器，希望这个分类器能够将不包含在训练数据中的数据标为 "unknown"。这种异常检测问题又叫做 Open-set Recognition；另一种是，训练数据不包含标签。我们又可以将这种情况分为两类，第一类是所有的训练数据都是正常的，而更常见的第二类是训练数据中的一小部分是异常数据。

## 有监督的异常检测

我们先来看训练数据包含不同类别标签的情况。在使用训练数据训练出一个分类器的同时，我们让分类器在输入 $x$，输出标签 $y$ 的同时，输出一个置信度 $c$，指模型对这次分类结果是正确的信心分数。置信度 $c$ 可以是 softmax 层输出的分布结果中的最大值，也可以用其他方法。我们可以设立一个阈值 $\lambda$，如果 $c > \lambda$，我们认为输入数据是正常的；而如果 如果 $c \leq \lambda$，我们认为输入数据是异常的。

实验表明，当输入数据是正常的时，置信度 $c$ 一般都接近于 1；而当输入数据是异常时，只有少数的置信度 $c$ 能接近 1，其他可能为 (0, 1) 范围中的任意值。尽管某些异常数据的置信度非常高，绝大多数正常数据的置信度都会超过异常数据，因此不必过于担心。

这种方法看似简单，但是在实际情况下表现通常不错，可以作为首先尝试的 baseline。当然有更好的方法，举例来说，在训练神经网络时，可以教它输出置信度。详情了解请看论文：Terrance DeVries, Graham W. Taylor, "[Learning Confidence for Out-of-Distribution Detection in Neural Networks](https://arxiv.org/pdf/1802.04865.pdf)", arXiv, 2018.

### 验证集

在异常检测的任务中，我们的验证集需要和测试集相同，标签只需要分为正常和异常，不需要分为正常数据中的具体类别。然后，我们通过验证集来评估模型表现，并以此调整阈值 $\lambda$ 和其他超参数。

### 模型表现评估

在异常检测问题中，正确率不是一个好的评估指标，因为异常检测的任务中正负样本数量比例通常比较悬殊，只要阈值调的足够低，检测器会将所有样本都判断为正常，并且这样做的正确率还会非常可观。

cost 加权、F1、Area under ROC

### 用分类器做异常检测的问题

对分类器来说，某些异常数据会存在正常数据中含有的代表性特征，使得分类器判断错误并给出比较高的置信度。有一些方法可以解决这个问题，一种是假设我们搜集到一些异常数据，我们可以教模型在学习分类的同时，给正常数据高的置信度，给异常数据低的置信度。这种方法的一篇参考论文是 Kimin Lee, Honglak Lee, Kibok Lee, Jinwoo Shin, "[Training Confidence-calibrated Classifiers for Detecting Out-of-Distribution Samples](https://arxiv.org/pdf/1711.09325.pdf)", ICLR 2018.

上述方法存在的问题还是我们很难找到异常数据。因此，我们考虑用生成模型来生成异常数据，并且要让生成的数据与正常数据没有那么像。一篇参考论文是 Mark Kliger, Shachar Fleishman, "[Novelty Detection with GAN](https://arxiv.org/pdf/1802.10560.pdf)", arXiv, 2018.

## 无监督的异常检测

在没有标签的异常检测问题中，我们可以用一个数据的概率分布模型来估计新输入样本的出现概率有多大，如果超过我们事先设置的阈值，我们认为这是一个正常样本，否则认为是异常。

### 最大似然

假设数据点都从一个概率密度函数 $f\_{\theta}(x)$ 采样得到，参数 $\theta$ 决定 $f\_{\theta}(x)$ 的形状，但是未知，必须从数据中得到。

**似然（likelihood）**指根据现有概率密度函数 $f\_{\theta}(x)$，产生的数据分布与我们现有的数据分布一致的概率有多大。似然的公式为$L(\theta)=f\_{\theta}\left(x^{1}\right) f\_{\theta}\left(x^{2}\right) \cdots f\_{\theta}\left(x^{N}\right)$。这里，我们不知道 $\theta$，因此我们要找一个能够使似然最大的 $\theta$，即 $\theta^{'}=\arg \max \_{\theta} L(\theta)$。

一个常用的概率密度函数就是高斯分布：

$$f\_{\mu, \Sigma}(x)=\frac{1}{(2 \pi)^{D / 2}} \frac{1}{|\Sigma|^{1 / 2}} \exp \left\\{-\frac{1}{2}(x-\mu)^{T} \Sigma^{-1}(x-\mu)\right\\}$$

其中，$D$ 是 $x$ 的维数。这个函数输入空间里的一个向量，输出这个向量被采样到的概率。$\mu$ 是均值（mean），$\Sigma$ 是协方差矩阵（covariance matrix），它们其实就是 $\theta$。

选用高斯分布作为概率密度函数，似然的公式就变为

$$L(\mu, \Sigma)=f\_{\mu, \Sigma}\left(x^{1}\right) f\_{\mu, \Sigma}\left(x^{2}\right) \cdots f\_{\mu, \Sigma}\left(x^{N}\right)$$

假如 $\mu$ 落在数据密集的地方，数据集中每个数据被采样到的概率就比较大，因此似然会比较大；而如果 $\mu$ 落在低密度的地方，似然就会比较低。因此，我们要穷举 $\mu$ 和 $\Sigma$ 各组可能的值，来看哪一组带入会算出最大的似然，即 $\mu^{'}, \Sigma^{'}=\arg \max \_{\mu, \Sigma} L(\mu, \Sigma)$。

$\mu^{'}$ 和 $\Sigma^{'}$ 的计算公式如下：

$$\mu^{'}=\frac{1}{N} \sum\_{n=1}^{N} x^{n}$$

$$\Sigma^{'}=\frac{1}{N} \sum\_{n=1}^{N}\left(x-\mu^{'}\right)\left(x-\mu^{'}\right)^{T}$$

因此，对于新输入的数据 $x$，如果 $f\_{\mu^{'}, \Sigma^{'}}(x)>\lambda$，就判断是正常数据；如果 $f\_{\mu^{'}, \Sigma^{'}}(x) \leq \lambda$，就判断是异常数据。

当然，选用高斯分布作为概率密度函数是基于数据分布接近高斯分布的假设而做的决策。一般使用高斯分布能够达到的效果都会好于其他选择，但是如果 $f\_{\theta}(x)$ 是一个神经网络，而 $\theta$ 是这个神经网络的参数，由于 $\theta$ 的参数量很大，因此可以不采用高斯分布。

### 其他方法

#### 自动编码器
 
![](https://raw.githubusercontent.com/bighuang624/pic-repo/master/Hung-yi-Lee-Anomaly-Detection-Autoencoder.png)
 
这里就不详细介绍自动编码器（Autoencoder）的运作原理了。用自动编码器做异常检测的原理是，用正常数据训练得到的自动编码器，在测试阶段时，输入的如果是正常数据，自动编码器重构得到的输出会与输入相似；而如果输入的是异常数据，自动编码器的重构损失会比较高。

#### 更多

![](https://raw.githubusercontent.com/bighuang624/pic-repo/master/Hung-yi-Lee-Anomaly-Detection-More.png)

## 参考资料

* 本节内容对应的 [PPT](http://speech.ee.ntu.edu.tw/~tlkagk/courses/ML\_2019/Lecture/Detection%20(v9).pdf)
* [异常检测（anomaly/ outlier detection）领域还有那些值得研究的问题？ - 知乎](https://www.zhihu.com/question/324999831/answer/716118043)


<script type="text/x-mathjax-config">
MathJax.Hub.Config({
  tex2jax: {inlineMath: [ ['$', '$'] ],
        displayMath: [ ['$$', '$$']]}
});
</script>

<script type="text/javascript" src="https://cdn.bootcss.com/mathjax/2.7.2/MathJax.js?config=default"></script>