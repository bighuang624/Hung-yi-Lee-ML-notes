为什么需要做**模型压缩（Network Compression）**？因为我们希望把我们的深度网络模型放在移动设备，但这些移动设备用于存储和计算的资源可能都是有限的。因此，不能使用层数太深或者参数量太大的深度网络模型。

从硬件角度来看，一种解决方法是为深度网络模型重新设计硬件架构。不过这门课程不会涉及这方面的内容。相应的，本节课程介绍了以下内容：

* **网络剪枝（Network Pruning）**
* **知识蒸馏（Knowledge Distillation）**
* **参数量化（Parameter Quantization）**
* **架构设计（Architecture Design）**
* **动态计算（Dynamic Computation）**

## 网络剪枝

**网络剪枝（Network Pruning）**指把一个大的深度网络模型的一些权重或者神经元剪枝掉，让模型变得小一点。网络剪枝有效的原因在于我们对于指定任务训练出的模型中，有很多参数是没有用的，即这些模型是过参化（over-parameterized）的，会造成神经网络计算开销（时间，存储空间硬件要求）过大。

网络剪枝的做法：

1. 有一个已经训练好的过大的深度网络模型；
2. 评估模型中每一个权重或者神经元的重要程度。评估权重重要性的方法包括直接看权重的数值，接近 0 说明不重要。评估神经元重要性的方法包括给定一个数据集，某个神经元的输出都接近于 0，可以认为这个神经元不重要；
3. 将所有的权重或者神经元按照重要性做排序，通过设置一个阈值来移除不重要的权重或者神经元；
4. 移除以后，任务评估指标一般会下降，但是因为移除的是不重要的部分，因此理想情况下不会下降太多。我们再将得到的新模型在训练数据上进行微调，来恢复模型的表现；
5. 通常我们不会一次移除太多权重或者神经元，因为这样容易导致模型表现无法恢复。通常的做法是，每次移除一小部分，再微调，反复这个过程直至模型的大小符合要求。

一个容易产生的质疑是，为什么我们不直接训练一个小一点的网络，而是要先训练一个大的网络模型，再进行剪枝？原因在于，较大的深度网络模型被认为更容易通过调整得到最好的结果。很多文献已经证明，只要网络够大，可以用梯度下降方法直接找到全局最小点。

ICLR 2019 上获得 best paper 的论文 "[The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks](https://arxiv.org/pdf/1803.03635.pdf)" 提出“彩票假设（Lottery Ticket Hypothesis）”来试图解释为什么需要权重剪枝。

![](https://raw.githubusercontent.com/bighuang624/pic-repo/master/Hung-yi-Lee-Lottery-Ticket-Hypothesis.png)

如上图所示，红色的权重代表随机初始化得到的，模型得到训练后得到紫色的权重。在经过剪枝后，如果将剩余的已训练权重（紫色）换成随机初始化的权重（绿色），整个网络无法训练得到好的结果；而如果是将剩余的已训练权重（紫色）换成对应位置的红色的随机初始化权重（这个操作称为“原始随机初始化（original random initialization）”），那么网络能够训练得到好的表现。文章的解释是，训练网络就像是买彩票，彩票买得越多就越有可能中奖。而一个巨大的神经网络可以认为是由很多的小的网络组成，神经网络越大，子网络越多。每一个小的子网络在随机初始化后，有的能够训练，有的不能训练得到好的表现。如果大的神经网络能够很容易训练得到好的表现，是因为子网络能够训练。然后做剪枝就是把这个子网络提取出来，并且表明最初的随机初始化的权重是好的，因此“原始随机初始化”能够训练得到好的表现。

ICLR 2019 上还有另外一篇论文 "[Rethinking the Value of Network Pruning](https://arxiv.org/pdf/1810.05270.pdf)" 却提出，直接从随机初始化开始训练小的网络，也能够训练出好的表现。这里的“随机初始化”是真正的随机初始化，而非像“彩票假设”中的“原始随机初始化（original random initialization）”。两篇文章的结论有一定的矛盾，请自行看 openreview 中评审的提问和作者的回答来辨证学习。

进行网络剪枝时，我们会考虑是移除权重好还是移除神经元好。在实际操作中，我们会发现移除权重会使得整个网络不规则，从而使实现难度较高，并且 GPU 难以通过矩阵运算进行加速。

![Hung-yi-Lee-weight-pruning](https://raw.githubusercontent.com/bighuang624/pic-repo/master/Hung-yi-Lee-weight-pruning.png)

因此实现时，将要移除的权重设为 0（而不是真正移除），这样做的问题是网络的大小实际上没有改变。NIPS 2016 上的论文 "[Learning Structured Sparsity in Deep Neural Networks](https://arxiv.org/pdf/1608.03665.pdf)" 发现用不同 GPU 对经过权重剪枝后比原来稀疏很多的 AlexNet 并没有大幅度加速。

![](https://raw.githubusercontent.com/bighuang624/pic-repo/master/Hung-yi-Lee-weight-pruning-performance.png)

因此，在实践中，更推荐进行神经元剪枝，实现起来更简单（对应的层少几个神经元），也更容易加速运算。

![Hung-yi-Lee-neuron-pruning](https://raw.githubusercontent.com/bighuang624/pic-repo/master/Hung-yi-Lee-neuron-pruning.png)

## 知识蒸馏

**知识蒸馏（Knowledge Distillation）**指先训练一个大的神经网络（我们称为**教师模型**），再训练一个小的网络模型（我们称为**学生模型**）来学习大的神经网络的行为。也就是说，如果是一个分类任务，学生模型的训练标签不是样本对应的真实类别，而是教师模型的分类判断（概率分布）。

![Hung-yi-Lee-Knowledge-Distillation](https://raw.githubusercontent.com/bighuang624/pic-repo/master/Hung-yi-Lee-Knowledge-Distillation.png)

以下是两篇相关论文：

* "[Distilling the Knowledge in a Neural Network](https://arxiv.org/pdf/1503.02531.pdf)", NIPS 2014 Deep Learning Workshop
* "[Do Deep Nets Really Need to be Deep?](https://arxiv.org/pdf/1312.6184.pdf)", NIPS 2014

为什么让学生模型学习教师模型会有比较好的结果？因为教师模型提供了比真实标签更为丰富的信息。在上图所示的例子中，通过教师模型输出的概率分布，学生模型不但能学习到输入的样本对应的类别是 1，还能学习到 1 和 7 很像。这样，你甚至可以将类别为 7 的所有样本从训练集中取出，在测试时学生模型也可能能够正确分类类别为 7 的样本。

在一些竞赛中，将很多模型集成起来是一个提高指标的常用方法。但在实际使用上，集成很多的模型来换取微不足道的提升是没有效率的。知识蒸馏可以帮助将集成的大量模型变成一个模型。

知识蒸馏的一个技巧是在 softmax 层将计算公式

$$ 
y\_{i}=\frac{\exp \left(x\_{i}\right)}{\sum\_{j} \exp \left(x\_{j}\right)}
 $$

换成

$$ 
y\_{i}=\frac{\exp \left(x\_{i} / T\right)}{\sum\_{j} \exp \left(x\_{j} / T\right)}
 $$

其中，$T$ 是一个被称为“温度（Temperature）”的超参数。这样做的原因是我们希望让学生模型学习到哪些类别更为相近，因此通过温度超参数，在不改变最大值对应的类别的同时来拉近教师模型输出的概率分布的值。不过李老师说根据助教的实验结果来看这种技巧的帮助有限。

## 参数量化

网络剪枝的目标是去掉不重要的权值与神经元，而**参数量化（Parameter Quantization）**着眼于对权值本身的大小进行压缩。

![Hung-yi-Lee-Trained-Quantization-and-Weight-Sharing](https://raw.githubusercontent.com/bighuang624/pic-repo/master/Hung-yi-Lee-Trained-Quantization-and-Weight-Sharing.png)

最容易想到的方法自然是使用更少的位数（bits）来表示一个值。一种更新颖的方法是如上图所示，首先对权重进行聚类。这样，同一类别共享一个权重值（即该类所有权重的均值），权值矩阵就只需要存储每个位置对应的类别来替代权重的实际值，通过 key-value 式的方法查找使用。在进行权值更新时，将同一类别的梯度值相加来更新每个类别共享的权重值即可。注意跨层的权重不进行权值的共享。

对于上述方法，我们还可以借用哈夫曼编码（Huffman Coding）的思想，对于常出现的类别用较少的位数表示，对于较少出现的类别用更多的位数表示，从而减少总体需要的编码长度。

参数量化方法的极致是只用 +1 和 -1 来表示权重。NIPS 2015 上的论文 "[BinaryConnect: Training Deep Neural Networks with binary weights during propagations](https://arxiv.org/pdf/1511.00363.pdf)" 提出了 Binary Connect。

![Hung-yi-Lee-Binary-Connect](https://raw.githubusercontent.com/bighuang624/pic-repo/master/Hung-yi-Lee-Binary-Connect.png)

上图中，我们认为每个位置代表一组参数，而每个灰色的点代表这组参数中的每个数值都是二元化的，都是 +1 或 -1。Binary Connect 的思想是，我们在找二元化参数的同时，保留一组值为实数的参数。Binary Connect 的训练和一般的训练过程很像，先随机初始化权重，这里的权重值可以是实数。但在 Binary Connect 中，不是用随机初始化得到的权重来计算梯度，而是找到与这组权重最接近的二元化参数组来计算梯度，将随机初始化权重根据最接近的二元化参数组计算得到的梯度方向进行更新。反复这个步骤直到权重不再更新，找到与当前权重最接近的二元化参数组作为最终的取值。

Binary Connect 原论文的实验结果显示，在 MNIST 数据集上，通过 Binary Connect 得到的结果比没有经过正则化的网络得到的结果还要略好。这是因为 Binary Connect 可以被看作一种正则化，网络权重被限制为只能是 +1 和 -1。不过，使用 Dropout 进行训练的网络的表现会比通过 Binary Connect 得到的表现要好。

以下还有两篇论文可供参考：

* "[Binarized Neural Networks: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1](https://arxiv.org/pdf/1602.02830.pdf)", 2016
* "[XNOR-Net: ImageNet Classification Using Binary Convolutional Neural Networks](https://arxiv.org/pdf/1603.05279.pdf)", ECCV 2016

## 架构设计

**架构设计（Architecture Design）**方法指调整网络的架构设计，让它变得只需要比较少的参数。

![](https://raw.githubusercontent.com/bighuang624/pic-repo/master/Hung-yi-Lee-Low-rank-Approximation.png)

对于全连接层，一种减少参数量的方法是参考矩阵分解，将一个大的权重矩阵分解为两个小的权重矩阵相乘。这种方法被称为**低秩近似（Low-rank Approximation）**。

![](https://raw.githubusercontent.com/bighuang624/pic-repo/master/Hung-yi-Lee-Standard-Convolution.png)

对于普通的卷积层，我们有很多组滤波器（filter），每组滤波器的数量等同于输入数据的 channel 数。

![](https://raw.githubusercontent.com/bighuang624/pic-repo/master/Hung-yi-Lee-Depthwise-Separable-Convolution.png)

而在 Depthwise Separable Convolution 中，卷积的过程分为两步。第一步叫做 Depthwise Convolution，滤波器的数量等同于输入数据的 channel 数，即每个滤波器只考虑一个 channel。这样，在这一步中 channel 相互之间没有影响。第二步叫做 Pointwise Convolution，使用固定大小为 1x1 的滤波器组做普通的卷积操作即可。由于我们降低了 size 较大的滤波器组中的滤波器数量，因此在输入输出不变的前提下，参数量得以降低。

![](https://raw.githubusercontent.com/bighuang624/pic-repo/master/Hung-yi-Lee-DSConv-Analysis.png)

从上图可以看到，Depthwise Separable Convolution 能够降低参数量的原因在于，普通的滤波器被拆解为两层，第一层的参数是共用的，第二层才使用独立的参数。

![](https://raw.githubusercontent.com/bighuang624/pic-repo/master/Hung-yi-Lee-Convs-Parameters.png)

上图计算了普通卷积和 Depthwise Separable Convolution 具体需要的参数数量。Depthwise Separable Convolution 被广泛用于各种将体积小作为宣传点的模型中，包括：

* SqueezeNet："[SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size](https://arxiv.org/pdf/1602.07360.pdf)", 2016
* MobileNet："[MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/pdf/1704.04861.pdf)", 2017
* ShuffleNet："[ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices](https://arxiv.org/pdf/1707.01083.pdf)", CVPR 2018
* Xception："[Xception: Deep Learning with Depthwise Separable Convolutions](https://arxiv.org/pdf/1610.02357.pdf)", CVPR 2017

## 动态计算

我们通过举例来说明**动态计算（Dynamic Computation）**：当我们的手机的电量严重不足时，我们希望能够动态调整运算量，能够在有限的能耗下得到一个尽量好的结果；而如果电量充足，就让模型做到最好。

最简单的做法是，训练很多参数量不等的模型，根据当前的电量来选择要使用的模型。这样做的缺点是要预先在储存空间较小的移动设备上储存很多预训练好的模型，显然不太满足实际情况。另一种做法是，以分类任务维例，我们训练很多以不同中间层的隐藏状态作为输入的分类器。这种做法的问题有两点：一是前几个中间层抽取的特征比较底层，学习到的表示语义较弱，分类效果不会特别好；二是这些分类器会强迫前几层就抽取高级特征，从而破坏原有模型的布局。

![](https://raw.githubusercontent.com/bighuang624/pic-repo/master/Hung-yi-Lee-Dynamic-Computation-Solutions.png)

对于以上问题，ICLR 2018 上的论文 "[Multi-Scale Dense Networks for Resource Efficient Image Classification](https://arxiv.org/pdf/1703.09844.pdf)" 提出了一种解决方案。这里不再详述。

![](https://raw.githubusercontent.com/bighuang624/pic-repo/master/Hung-yi-Lee-Multi-Scale-Dense-Networks.png)

## 参考资料

* 本节内容对应的 [PPT](http://speech.ee.ntu.edu.tw/~tlkagk/courses/ML_2019/Lecture/Small%20(v6).pdf)
* [Deep Learning Theory 2-4: Geometry of Loss Surfaces (Conjecture)](https://www.youtube.com/watch?v=_VuWvQUMQVk)：李老师之前课程的录影，讲为什么较大的神经网络比较容易得到好的表现
* "[Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman coding](https://arxiv.org/pdf/1510.00149.pdf)", ICLR 2016 (best paper)
* [【知识蒸馏相关文献列表】knowledge-distillation-papers by lhyfst GitHub](https://github.com/lhyfst/knowledge-distillation-papers)


<script type="text/x-mathjax-config">
MathJax.Hub.Config({
  tex2jax: {inlineMath: [ ['$', '$'] ],
        displayMath: [ ['$$', '$$']]}
});
</script>

<script type="text/javascript" src="https://cdn.bootcss.com/mathjax/2.7.2/MathJax.js?config=default"></script>