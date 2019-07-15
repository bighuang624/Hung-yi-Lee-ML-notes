## 元学习简介

**元学习（Meta Learning）**指 learn to learn，我们希望机器根据过去任务上汲取的经验变成更厉害的学习者，学习到学习的技巧（而不只是学习如何执行特定任务）。这样，当新的任务来临时，机器可以学习得更快更好。换言之，我们希望能从以往的经验中明确地学习到提高下游学习效率的先验。

元学习和**终身学习（Life-long Learning）**的异同点：

* 共同点：都是先让机器看过很多任务，希望机器能够在新的任务上仍然做的好。
* 区别：终身学习用同一个模型学习，希望同一个模型能同时学会很多技能；元学习中，不同的任务有不同模型，希望机器可以从过去的学习经验中学到一些共用的先验知识，使得在训练新任务所用的模型时可以训练得又快又好。

在标准的监督学习过程中，我们通过**学习算法（Learning Algorithm）**得到模型，这个模型可以被看作是一个映射函数，输入样本数据，输出预测标签。而学习算法帮助找到这个映射函数的一组参数，使得函数可以近似真实的映射关系；在元学习中，学习算法也被看作一个函数，这个函数的输入是训练数据，输出另一个函数，这个输出的函数就是模型。

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

我们在这里介绍两种元学习的算法：

* **MAML**: Chelsea Finn, Pieter Abbeel, Sergey Levine, "[Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks](https://arxiv.org/pdf/1703.03400.pdf)", ICML, 2017

* **Reptile**: Alex Nichol, Joshua Achiam, John Schulman, "[On First-Order Meta-Learning Algorithms](https://arxiv.org/pdf/1803.02999.pdf)", arXiv, 2018

### MAML

过去的神经网络的初始化参数都是从某一个分布中采样得到，**MAML（Model-Agnostic Meta-Learning）**算法要做的事情是学习出一组用于初始化的最好的神经网络参数 $\phi$。注意，虽然 MAML 可以翻译为“模型无关的元学习”，但是 MAML 要求训练和测试阶段的模型结构必须一致，因为它们需要共用同一组初始化参数。“模型无关”指训练和测试阶段的模型结构只需要一致，就可以自由选择使用的模型结构。

![](https://raw.githubusercontent.com/bighuang624/pic-repo/master/Hung-yi-Lee-MAML.png)

MAML 有以下优点：

* 不需要为元学习过程引入额外的、需要学习的参数；
* 可以在任何适合用基于梯度下降方法训练的模型，以及不同的可微分任务（包括分类、回归和强化学习）上使用；
* 方法只生成一个初始化权重，可以使用任意数量的数据和梯度下降步骤来执行自适应。

#### 符号表示

我们首先来规定在接下来的叙述中使用的符号表示：

* $\phi$：对于所有任务使用的初始化参数；
* $\phi^{i}$：$\phi$ 进行第 i 次更新后的值；
* $\widehat{\theta}^{n}$：模型从任务 n 中学习到的参数，这个参数由 $\phi$ 进行更新得到；
* $l^n(\widehat{\theta}^{n})$：模型参数为 $\widehat{\theta}^{n}$ 时在任务 n 的测试集上计算得到的损失；
* $L(\phi) = \sum^N\_{n=1}l^n(\widehat{\theta}^{n})$：总体的损失函数；
* $\eta$, $\varepsilon$：两个不同的学习率（值可以相同，但更多情况下不同）。

#### 算法过程

MAML 的训练过程描述如下：

1. 随机初始化 $\phi$（即得到 $\phi^0$）；
2. 从任务的概率分布中采样得到一批任务；
3. 对于这批任务中的每个任务，计算 $\nabla\_{\phi} l(\phi)$，并得到对于该任务得到的自适应参数 $\hat{\theta}=\phi-\varepsilon \nabla\_{\phi} l(\phi)$；
4. 更新 $\phi$：$\phi \leftarrow \phi-\eta \nabla\_{\phi} L(\phi)$；
5. 循环第 2～4 步，直到训练结束。

注意，我们不在意 $\phi$ 在训练任务上的直接表现，而在意用 $\phi$ 训练出来的 $\widehat{\theta}^{n}$ 在训练任务上的表现。模型预训练（Model Pre-training）看起来与 MAML 有相似之处，区别在于模型预训练试图找到在所有任务上直接表现最好的 $\phi$，但并不保证拿 $\phi$ 去训练以后会得到好的 $\widehat{\theta}^{n}$。

从第 3 步的公式 $\hat{\theta}=\phi-\varepsilon \nabla\_{\phi} l(\phi)$ 可以看到，MAML 在对于每个任务进行参数的更新时只更新一次，原因有：

* 元学习的训练的计算量很大，只更新一次参数能够提高计算速度；
* 假设 MAML 能学习出一个非常好的初始化参数，我们希望能够只进行一次参数更新就得到最好的模型参数，因此将其作为目标来看能否实现；
* 在实际测试时，如果只更新一次时效果不好，可以多更新几次；
* 小样本学习的数据很少，多次更新参数容易导致过拟合。

#### 数学推导

重复一遍 MAML 的训练方法：

$$\begin{array}{l}{\phi \leftarrow \phi-\eta \nabla\_{\phi} L(\phi)} \\\ {L(\phi)=\sum\_{n=1}^{N} l^{n}\left(\hat{\theta}^{n}\right)} \\\ {\hat{\theta}=\phi-\varepsilon \nabla\_{\phi} l(\phi)}\end{array}$$

这里，我们来推导一下梯度项 $\nabla\_{\phi} L(\phi)$ 是什么样子。将 $L(\phi)$ 替换，并将求和提取出来，则有：

$$\nabla\_{\phi} L(\phi)=\nabla\_{\phi} \sum\_{n=1}^{N} l^{n}\left(\hat{\theta}^{n}\right)=\sum\_{n=1}^{N} \nabla\_{\phi} l^{n}\left(\hat{\theta}^{n}\right)$$

梯度是一个向量，其每一个维度代表了某一个参数对损失函数的偏微分的结果，即

$$\nabla\_{\phi} l(\hat{\theta})=\left[\begin{array}{c}{\partial l(\hat{\theta}) / \partial \phi\_{1}} \\\ {\partial l(\hat{\theta}) / \partial \phi\_{2}} \\\ {\vdots} \\\ {\partial l(\hat{\theta}) / \partial \phi\_{i}}\end{array}\right]$$

其中，

$$\frac{\partial l(\hat{\theta})}{\partial \phi\_{i}}=\sum\_{j} \frac{\partial l(\hat{\theta})}{\partial \hat{\theta}\_{j}} \frac{\partial \hat{\theta}\_{j}}{\partial \phi\_{i}}$$

$\phi\_{i}$ 是学习到的初始参数，它通过影响 $\hat{\theta}\_{1}, \hat{\theta}\_{2}, \dots, \hat{\theta}\_{j}$ 来最终影响 $l(\hat{\theta})$。

$\frac{\partial l(\hat{\theta})}{\partial \hat{\theta}\_j}$ 与损失函数的形式，以及训练任务中的测试集有关，可以算出。现在来看 $\frac{\partial \hat{\theta}\_{j}}{\partial \phi\_{i}}$ 。从式子

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

原论文表明这种优化方法使得计算速度提升了约 33%。并且通过测试发现，算法的效果没有受到明显的影响。

### Reptile

Reptile 的训练过程如下：

1. 随机初始化 $\phi$（即得到 $\phi^0$）；
2. 从任务的概率分布中采样得到一批任务；
3. Reptile 在具体任务上的参数更新方式与 MAML 相同，只是不限制在一个训练任务上的参数更新次数，因此我们**多次**更新参数，得到 $\hat{\theta}^m$；
4. 让 $\phi^{0}$ 沿着 $\phi^{0}$ 到 $\hat{\theta}^m$ 的方向更新一次，得到 $\phi^{1}$；
5. 循环第 2～4 步，直到训练结束。

![](https://raw.githubusercontent.com/bighuang624/pic-repo/master/Hung-yi-Lee-Reptile.png)

可以看到，Reptile 的过程与模型预训练有相似之处。为了区分 Reptile、MAML 和模型预训练，我们有下图所示例子：

![](https://raw.githubusercontent.com/bighuang624/pic-repo/master/Hung-yi-Lee-Meta-Learning-comparision.png)

可以看到，当我们要决定初始参数 $\phi$ 的更新方向时，我们先利用采样得到的训练任务进行两次更新，方向分别为 $g\_1$ 和 $g\_2$。那么，模型预训练中的 $\phi$ 更新方向为 $g\_1$，MAML 中的 $\phi$ 更新方向为 $g\_2$，而 Reptile 中的 $\phi$ 更新方向为 $g\_1 + g\_2$（当然，如之前所说，Reptile 没有限制只能走两步，这里只是以两次更新为例）。

### 其他讨论

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