## Metric-based 的小样本学习

<!--我们在这节会有很多<del>天下第一<\del>三玖。-->

回到小样本学习。小样本学习的模型大致可以分为三类：Model-based、Metric-based 和 Optimization-based。其中，**Model-based 方法**旨在通过模型结构的设计快速在少量样本上更新参数，直接建立输入和预测结果的映射函数；**Metric-based 方法**通过度量支撑集（support set）中的样本和查询集（query set）中样本的距离，借助最近邻的思想完成分类；**Optimization-based 方法**认为普通的梯度下降方法难以在小样本的场景下拟合，因此通过调整优化方法来完成小样本分类的任务。

在这一节中，我们主要对 Metric-based 的方法进行介绍。

---

现在我们回过去看之前那个疯狂的想法：能否直接学习一个函数，这个函数既做了训练，又做了测试。给这个函数训练数据，它黑箱式地训练一个模型，然后给它测试数据，它就输出结果。

实际上，人脸认证（Face Verification）就是这样的一个任务。人脸认证任务要求给一张人脸，判断是不是某个人。而我们在开始使用手机时转头收集的几张照片就是训练数据。注意，人脸认证和人脸识别（Face Identifiction）是不同的任务，人脸识别任务要求判断某一张人脸是一组人中的哪一个。

人脸认证任务也是一个元学习的任务。其原理如下图所示。

![](https://raw.githubusercontent.com/bighuang624/pic-repo/master/Hung-yi-Lee-Face-Verification.png)

### 孪生网络

**孪生网络（Siamese Network）**构造了一个双路的神经网络。一般来说，孪生网络内部的 CNN 的参数是共享的，即两个 CNN 的参数是一样的（也可以不一样）。

![](https://raw.githubusercontent.com/bighuang624/pic-repo/master/Hung-yi-Lee-Siamese-Network.png)

我们从元学习的角度来切入，孪生网络可以看作是一种元学习的方法，直接输入训练和测试数据就得到输出，但是其内部架构设计得非常简洁，完全可以直观解释为什么这个网络要这样做。

#### 孪生网络的直观解释

我们可以不把孪生网络当作元学习问题看待，而认为它是一个单纯的二分类问题。这个二分类问题的输入是两张图片，输出是否是同一个人的判断。

孪生网络内部的 CNN 将所有人脸图片投影到一个空间上，在这个空间上，同一个人的人脸图片比较接近，不同人的人脸图片相隔较远。我们不使用 PCA 或者 Autoencoder 的原因是，因为 Autoencoder 不知道它要解决的任务是什么，它会保留大多数的信息，但它不知道什么信息是重要的，什么信息不重要。而孪生网络内部的 CNN 可以根据任务忽略不重要的信息，因为我们要求将同样的人脸投影后距离较近，不同的人的脸投影后距离较远，因此它可能会学到照片的背景不重要，而照片里人的头发颜色比较重要。

如何计算在这个空间中两个点的距离？有以下参考文献可以作为参考：

* W Liu, Y Wen, Z Yu, M Li, B Raj, et al., "[Sphereface: Deep Hypersphere Embedding for Face Recognition](http://openaccess.thecvf.com/content_cvpr_2017/papers/Liu_SphereFace_Deep_Hypersphere_CVPR_2017_paper.pdf)", CVPR, 2017
* F Wang, J Cheng, W Liu, H Liu, "[Additive Margin Softmax for Face Verification](https://arxiv.org/pdf/1801.05599.pdf)", 2018
* J Deng, J Guo, N Xue, et al., "[ArcFace: Additive Angular Margin Loss for Deep Face Recognition](http://openaccess.thecvf.com/content_CVPR_2019/papers/Deng_ArcFace_Additive_Angular_Margin_Loss_for_Deep_Face_Recognition_CVPR_2019_paper.pdf)", CVPR, 2019


上述例子做的都是 one-shot learning，我们每次只给一张人脸。我们也可以做 triplet loss，每次给一张和目标一致的人脸和一张和目标不是同一个人的人脸，这样的效果会更好。参考文献包括：

* E Hoffer, N Ailon, "[Deep Metric Learning Using Triplet Network](https://arxiv.org/pdf/1412.6622.pdf)", ICLR, 2015
* F Schroff, D Kalenichenko, J Philbin, "[Facenet: A Unified Embedding for Face Recognition and Clustering](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Schroff_FaceNet_A_Unified_2015_CVPR_paper.pdf)", CVPR, 2015

### 其他方法

如果现在做人脸识别任务，从二分类变成多分类。例如，我们要做 5-ways 1-shot，每次输入五类，每类一张图片，然后测试时输入一张图片，要求判断是五类中的哪一个。

![](https://raw.githubusercontent.com/bighuang624/pic-repo/master/Hung-yi-Lee-5-ways-1-shot.png)

网络的架构有很多设计方法，其中一种经典的方法就是论文 "[Prototypical Networks for Few-shot Learning](https://arxiv.org/pdf/1703.05175.pdf)" 提出的**原型网络（Prototypical Network）**。在将训练数据和测试数据都变为 embedding 后，计算测试的 embedding 和训练数据中每个 embedding 的相似度，之后通过 softmax 层，通过交叉熵来做一个传统的分类即可。如果做 few-shot，就把训练数据中每一类的 embedding 求平均即可。这样，分类问题变成在 embedding 空间中的最近邻。

![](https://raw.githubusercontent.com/bighuang624/pic-repo/master/Hung-yi-Lee-Prototypical-Network.png)

还有一种做法是论文 "[Matching Networks for One Shot Learning](https://arxiv.org/pdf/1606.04080.pdf)" 中提出的**匹配网络（Matching Network）**，它与原型网络最大的区别是，原型网络中每张图片有一个单独的 CNN，分开处理；而 匹配网络会认为训练数据中每张图片之间也是有关系的，因此使用一个 Bidirectional LSTM 来将每张图片处理为 embedding。当然，比较容易想到的是我们调换输入数据的顺序时，会影响输出结果，与我们的直觉不太相符。原型网络也确实比匹配网络更晚提出，因此对这个地方进行了改善。

匹配网络还有一个不一样的地方是，在计算出相似度后，还通过一个 multiple hop process，这个和 memory network 中的 process 相似。这里不继续展开了。

![](https://raw.githubusercontent.com/bighuang624/pic-repo/master/Hung-yi-Lee-Matching-Network.png)

论文 "[Learning to Compare: Relation Network for Few-Shot Learning](https://arxiv.org/pdf/1711.06025.pdf)" 提出的 **Relation Network** 和上述的网络的原理也很相似，区别在于上述网络在举例度量上使用了固定的度量方式，而 Relation Network 认为距离度量也是重要一环，需要对其进行建模。因此，该网络将测试数据得到的 embedding 拼接到训练数据的每一个 embedding 的后面，然后将这些拼接的结果输入到一个新的网络中，并更多的关注 relation score，认为更像是回归问题而非分类，所以用 MSE 取代交叉熵作为损失函数。这样，网络自身可以训练 embedding 距离的度量方式。

![](https://raw.githubusercontent.com/bighuang624/pic-repo/master/Hung-yi-Lee-Relation-Network.png)

在 few-shot learning 中常遇到的问题是训练数据很少，我们有一张人平静的脸，希望机器能想象他开心、生气、悲伤的样子。论文 "[Low-Shot Learning from Imaginary Data](https://arxiv.org/abs/1801.05401)" 提出 Few-shot Learning for Imaginary Data，训练数据中每一个类有一个生成器（generator），生成相关的图片。这个生成器与用于分类的神经网络一块训练。

![](https://raw.githubusercontent.com/bighuang624/pic-repo/master/Hung-yi-Lee-Few-shot-Learning-for-Imaginary-Data.png)

### Train+Test as RNN

我们想要找到一个函数，同时能够做学习和测试。我们将训练数据（及标签）和测试数据输入，这个函数就能直接输出对测试数据的预测标签。我们之前使用的是孪生网络以及其变形，可以认为是特别设计了网络的架构来实现上述目标。现在，我们想用一个 general 网络来实现上述目的。

我们可以将其想成是一个 RNN 可以解决的问题，因为输入是一个序列，先把训练数据的每一个样本一个一个输入，然后输入测试数据，就能输出结果。例如人脸认证，我们可以将图片用 CNN 编码，然后类别标签作为 one-hot 编码，两者拼接后作为 RNN 每个时间步的输入。

![](https://raw.githubusercontent.com/bighuang624/pic-repo/master/Hung-yi-Lee-MANN-and-SNAIL.png)

在前人的尝试下，这样做使用一般的 LSTM 无法训练。因此，修改 LSTM 的结构，比较知名的两个例子是：

* MANN，由论文 "[One-shot Learning with Memory-Augmented Neural Networks](https://arxiv.org/pdf/1605.06065.pdf)" 提出
* SNAIL，由论文 "[A Simple Neural Attentive Meta-Learner](https://arxiv.org/pdf/1707.03141.pdf)" 提出

其中，SNAIL 在 LSTM 的基础上加入了 Attention。当输入测试数据时，SNAIL 会对之前输入的训练数据做 Attention，因此还是可以看作将训练数据和测试数据进行了比对，其实和孪生网络有异曲同工之妙。

而 MANN 实际上是一种 Model-based 的方法。

## 参考资料

* 本节内容对应的 [PPT](http://speech.ee.ntu.edu.tw/~tlkagk/courses/ML_2019/Lecture/Meta2%20(v4).pdf)
* [小样本学习（Few-shot Learning）综述 - 知乎](https://zhuanlan.zhihu.com/p/61215293)

<script type="text/x-mathjax-config">
MathJax.Hub.Config({
  tex2jax: {inlineMath: [ ['$', '$'] ],
        displayMath: [ ['$$', '$$']]}
});
</script>

<script type="text/javascript" src="https://cdn.bootcss.com/mathjax/2.7.2/MathJax.js?config=default"></script>