## 元学习简介

**元学习（Meta Learning）**指 learn to learn，我们希望机器根据过去任务上汲取的经验变成更厉害的学习者，学习到学习的技巧（而不只是学习如何执行特定任务）。这样，当新的任务来临时，机器可以学习得更快更好。换言之，我们希望能从以往的经验中明确地学习到提高下游学习效率的先验。

元学习和**终身学习（Life-long Learning）**的异同点：

* 共同点：都是先让机器看过很多任务，希望机器能够在新的任务上仍然做的好。
* 区别：终身学习用同一个模型学习，希望同一个模型能同时学会很多技能；元学习中，不同的任务有不同模型，希望机器可以从过去的学习经验中学到一些东西，在训练新模型时可以训练得又快又好。

在标准的监督学习过程中，我们通过**学习算法（Learning Algorithm）**得到模型，这个模型可以被看作是一个映射函数，输入样本数据，输出预测标签；在元学习中，学习算法也被看作一个函数，这个函数的输入是训练数据，输出另一个函数，这个输出的函数就是模型。

![](https://raw.githubusercontent.com/bighuang624/pic-repo/master/Hung-yi-Lee-Meta-Learning.png)

在模型的常规训练过程中，很多的步骤是人为手工设计的，例如网络架构、初始化方法、参数更新方法等，选择不同的设计导致了不同的学习算法。元学习希望不要人为定义这些选项，而是让机器学习出最好的选项。

![](https://raw.githubusercontent.com/bighuang624/pic-repo/master/Hung-yi-Lee-Define-a-set-of-learning-algorithm.png)

机器学习通常使用损失函数（Loss Function）来评估模型的好坏，元学习也可以定义一个类似的指标来评估学习算法。如下图所示，我们找能使 L(F) 最小的学习算法 F，之后将学习算法 F 用在测试任务上，将测试任务的训练集输入到学习算法 F 中，得到模型 f，然后将测试任务的测试集输入到模型 f 得到结果，通过此结果评估学习算法的好坏。

![](https://raw.githubusercontent.com/bighuang624/pic-repo/master/Hung-yi-Lee-Defining-the-goodness-of-a-learning-algorithm.png)

### 数据集的划分

监督学习和元学习需要的数据集不同。对于监督学习，以猫狗图片分类为例，需要训练集和测试集，每个数据集中有很多猫狗图片；而元学习，需要准备的不是训练集和测试集，而是训练任务和测试任务（有时候也需要用于验证的任务），每个任务中各自有训练集和测试集。这样，我们通过训练任务让机器在类别发生变化时保持模型的泛化能力，在测试任务上，面对全新的类别时，不需要变动已有的模型，就可以完成分类。

![](https://raw.githubusercontent.com/bighuang624/pic-repo/master/Hung-yi-Lee-data-set-differences.png)

元学习常常与**小样本学习（Few-shot Learning）**一起讨论。小样本学习是元学习在监督学习领域的应用，要解决的问题是希望机器学习模型在学习了一定类别的大量数据后，对于新的类别，只需要非常少量的样本就能快速学习。

元学习常用的一个数据集是 [Omniglot](https://github.com/brendenlake/omniglot)，拥有 1623 个不同的符号种类，每类符号有 20 个不同的样本。Omniglot 在应用于小样本学习的分类任务上时，常常遵循 N-ways K-shot classification 的方法，即在训练阶段，会在训练集中随机抽取`N`个类别，每个类别`K`个样本，总共`N*K`个样本作为一个训练任务的**支撑集（support set）**；再从这`N`个类别剩余的数据中抽取一批样本，作为该训练任务的**查询集（query set）**，来训练模型从`N*K`个数据中学会如何区分这`N`个类别。测试任务的构建同理。

![](https://raw.githubusercontent.com/bighuang624/pic-repo/master/Hung-yi-Lee-Omniglot.png)

## 元学习技术

* MAML
  * Chelsea Finn, Pieter Abbeel, Sergey Levine, "[Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks](https://arxiv.org/pdf/1703.03400.pdf)", ICML, 2017

* Reptile
  * Alex Nichol, Joshua Achiam, John Schulman, "[On First-Order Meta-Learning Algorithms](https://arxiv.org/pdf/1803.02999.pdf)", arXiv, 2018

### MAML

过去的神经网络的初始化参数都是从某一个分布中采样得到，**MAML（Model-Agnostic Meta-Learning）**要做的事情是学习出一个用于初始化的最好的参数 $\phi$。因此，MAML 要求所有的模型的结构必须一致。

![](https://raw.githubusercontent.com/bighuang624/pic-repo/master/Hung-yi-Lee-MAML.png)

在将 $L(\phi)$ 最小化时，我们使用梯度下降，$\phi \leftarrow \phi-\eta \nabla\_{\phi} L(\phi)$ 。

注意，我们不在意 $\phi$ 在训练任务上的直接表现，而在意用 $\phi$ 训练出来的 $\widehat{\theta}^{n}$ 的表现。模型预训练（Model Pre-training）看起来与 MAML 有相似之处，区别在于模型预训练试图找到在所有任务上都最好的 $\phi$，但并不保证拿 $\phi$ 去训练以后会得到好的 $\widehat{\theta}^{n}$。

假设 MAML 中训练时参数只更新一次，则有 $\hat{\theta}=\phi-\varepsilon \nabla\_{\phi} l(\phi)$。只做一次参数更新的原因有：

* 元学习的训练的计算量很大，为了速度快，只更新一次参数；
* 假设 MAML 能学习出一个非常好的初始化参数，我们希望能够只进行一次参数更新就得到最好的模型参数，因此将其作为目标来看能否实现；
* 在实际测试时，如果只更新一次时效果不好，可以多更新几次；
* 小样本学习的数据很少，多次更新参数容易导致过拟合。

#### 数学推导

重复一遍 MAML 的训练方法：

$$\begin{array}{l}{\phi \leftarrow \phi-\eta \nabla\_{\phi} L(\phi)} \\\ {L(\phi)=\sum\_{n=1}^{N} l^{n}\left(\hat{\theta}^{n}\right)} \\\ {\hat{\theta}=\phi-\varepsilon \nabla\_{\phi} l(\phi)}\end{array}$$

其中，$\eta$ 和 $\varepsilon$ 是两个不同的学习率（值可以相同，但更多情况下不同）。

这里，我们来推导一下梯度项 $\nabla\_{\phi} L(\phi)$ 是什么样子。

将 $L(\phi)$ 替换，并将求和提取出来，则有：

$$\nabla\_{\phi} L(\phi)=\nabla\_{\phi} \sum\_{n=1}^{N} l^{n}\left(\hat{\theta}^{n}\right)=\sum\_{n=1}^{N} \nabla\_{\phi} l^{n}\left(\hat{\theta}^{n}\right)$$

梯度是一个向量，其每一个维度代表了某一个参数对损失函数的偏微分的结果，即

$$\nabla\_{\phi} l(\hat{\theta})=\left[\begin{array}{c}{\partial l(\hat{\theta}) / \partial \phi\_{1}} \\\ {\partial l(\hat{\theta}) / \partial \phi\_{2}} \\\ {\vdots} \\\ {\partial l(\hat{\theta}) / \partial \phi\_{i}}\end{array}\right]$$

其中，

$$\frac{\partial l(\hat{\theta})}{\partial \phi\_{i}}=\sum\_{j} \frac{\partial l(\hat{\theta})}{\partial \hat{\theta}\_{j}} \frac{\partial \hat{\theta}\_{j}}{\partial \phi\_{i}}$$

$\phi\_{i}$ 是学习到的初始参数，它通过影响 $\hat{\theta}\_{1}, \hat{\theta}\_{2}, \dots, \hat{\theta}\_{j}$ 来最终影响 $l(\hat{\theta})$。

$\frac{\partial l(\hat{\theta})}{\partial \hat{\theta}\_j}$ 与损失函数的形式，以及训练任务中的测试集有关，可以算出。

现在来看 $\frac{\partial \hat{\theta}\_{j}}{\partial \phi\_{i}}$ 。从式子

$$\widehat{\theta}=\phi-\varepsilon \nabla\_{\phi} l(\phi)$$

中选择一个维度，则有，

$$\hat{\theta}\_{j}=\phi\_{j}-\varepsilon \frac{\partial l(\phi)}{\partial \phi\_{j}}$$

当 $i \neq j$ 时，

$$\frac{\partial \hat{\theta}\_{j}}{\partial \phi\_{i}}=-\varepsilon \frac{\partial l(\phi)}{\partial \phi\_{i} \partial \phi\_{j}}$$

而如果 $i=j$，则有

$$\frac{\partial \hat{\theta}\_{j}}{\partial \phi\_{i}}=1-\varepsilon \frac{\partial l(\phi)}{\partial \phi\_{i} \partial \phi\_{j}}$$

这样，我们就可以计算来求 $\frac{\partial l(\hat{\theta})}{\partial \phi\_{i}}$ 了。但是，在这个过程中，我们需要进行二次微分来计算 $\frac{\partial l(\phi)}{\partial \phi\_{i} \partial \phi\_{j}}$，非常花时间。因此，提出 MAML 的原论文考虑将其忽略（文中写作 using a first-order approximation），即 $i \neq j$ 时，$\frac{\partial \hat{\theta}\_{j}}{\partial \phi\_{i}} \approx 0$；$i=j$ 时，$\frac{\partial \hat{\theta}\_{j}}{\partial \phi\_{i}} \approx 1$。因此，只需要考虑 $i=j$ 的情况，即

$$\frac{\partial l(\hat{\theta})}{\partial \phi\_{i}}=\sum\_{j} \frac{\partial l(\hat{\theta})}{\partial \hat{\theta}\_{j}} \frac{\partial \hat{\theta}\_{j}}{\partial \phi\_{i}} \approx \frac{\partial l(\hat{\theta})}{\partial \hat{\theta}\_{i}}$$

因此。就变成损失函数对 $\hat{\theta}$ 做偏微分：

$$\nabla\_{\phi} l(\hat{\theta})=\left[\begin{array}{c}{\partial l(\hat{\theta}) / \partial \phi\_{1}} \\\ {\partial l(\hat{\theta}) / \partial \phi\_{2}} \\\ {\vdots} \\\ {\partial l(\hat{\theta}) / \partial \phi\_{i}}\end{array}\right]=\left[\begin{array}{c}{\partial l(\hat{\theta}) / \partial \hat{\theta}\_{1}} \\\ {\partial l(\hat{\theta}) / \partial \hat{\theta}\_{2}} \\\ {\vdots} \\\ {\partial l(\hat{\theta}) / \partial \hat{\theta}\_{i}}\end{array}\right]=\nabla\_{\hat{\theta}} l(\hat{\theta})$$

计算变得简单很多。

#### MAML 的实际实现

初始参数 $\phi$ 有一个初始值 $\phi^{0}$，每一个任务就是一个训练数据集，有一个 task batch，假设做随机梯度下降，那么实际实现的过程如下：

1. 采样得到一个训练任务 m；
2. 通过训练从 $\phi^{0}$ 更新一次参数，得到 $\hat{\theta}^m$；
3. 计算 $\hat{\theta}^m$ 对其损失函数的偏微分（即梯度）；
4. 用这个梯度将 $\phi^{0}$ 更新为 $\phi^{1}$；
5. 重复第 1-4 步。

![](https://raw.githubusercontent.com/bighuang624/pic-repo/master/Hung-yi-Lee-MAML-real-implementation.png)

作为对比，模型预训练的区别是，$\phi$ 的更新是利用每次采样后的第一次参数更新的梯度。

### Reptile

同样，我们的初始参数 $\phi$ 有一个初始值 $\phi^{0}$。Reptile 的过程如下：

1. 采样得到一个训练任务 m；
2. Reptile 不限制在一个训练任务上的参数更新次数，因此我们**多次**更新参数，得到 $\hat{\theta}^m$；
3. 让 $\phi^{0}$ 沿着 $\phi^{0}$ 到 $\hat{\theta}^m$ 的方向更新一次，得到 $\phi^{1}$；
4. 重复第 1-3 步。

![](https://raw.githubusercontent.com/bighuang624/pic-repo/master/Hung-yi-Lee-Reptile.png)

可以看到，Reptile 的过程与模型预训练有相似之处。为了区分 Reptile、MAML 和模型预训练，我们有下图所示例子：

![](https://raw.githubusercontent.com/bighuang624/pic-repo/master/Hung-yi-Lee-Meta-Learning-comparision.png)

可以看到，当我们要决定初始参数 $\phi$ 的更新方向时，我们先利用采样得到的训练任务进行两次更新，方向分别为 $g\_1$ 和 $g\_2$。那么，模型预训练中的 $\phi$ 更新方向为 $g\_1$，MAML 中的 $\phi$ 更新方向为 $g\_2$，而 Reptile 中的 $\phi$ 更新方向为 $g\_1 + g\_2$（当然，如之前所说，Reptile 没有限制只能走两步，这里只是以两次更新为例）。

### 其他

MAML 和 Reptile 的目的都是找到神经网络最好的初始参数。当然也有一系列方法找更多有价值的东西，例如神经网络的架构（architecture），激活函数（activation function），或者更新参数的方法。这些都是通过训练一个神经网络，这个神经网络来输出以上内容。因为神经网络输出神经网络无法微分，因此需要利用强化学习或者遗传算法等来训练用来输出神经网络的神经网络。

相关内容（自动调参，AutoML）可见李老师的视频: [
https://www.youtube.com/watch?v=c10nxBcSH14](
https://www.youtube.com/watch?v=c10nxBcSH14)

有一个值得思考的问题是，我们通过梯度下降学习到了最好的初始化参数 $\phi$，但是 $\phi$ 也需要一个初始化的值 $\phi^0$。因此，我们现在在做 learn to learn，未来我们会不会需要做 learn to learn to learn？

无论 MAML 还是 Reptile，都需要梯度下降算法。还有一个想法是，能不能让学习算法就是一个神经网络，不管它是否做梯度下降，我们就是输入训练数据，它输出的就直接是要训练的模型的参数。而我们如何设计这个神经网络就决定我们的学习算法是什么样。更疯狂的是，神经网络输出的参数本身也要用来测试来评估，我们能否学习一个更大的函数，将训练和测试的过程都包含在内，这个函数就输入训练数据和测试数据，直接输出测试数据的结果，而不知道内部的模型是什么样？我们将在之后讨论。

## 参考资料

* 本节内容对应的 [PPT](http://speech.ee.ntu.edu.tw/%7Etlkagk/courses/ML\_2019/Lecture/Meta1%20(v6).pdf)
* Demo of Reptile: [https://openai.com/blog/reptile](https://openai.com/blog/reptile)

<script type="text/x-mathjax-config">
MathJax.Hub.Config({
  tex2jax: {inlineMath: [ ['$', '$'] ],
        displayMath: [ ['$$', '$$']]}
});
</script>

<script type="text/javascript" src="https://cdn.bootcss.com/mathjax/2.7.2/MathJax.js?config=default"></script>