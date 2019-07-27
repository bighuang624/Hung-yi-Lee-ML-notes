**Life-Long Learning（终身学习）**，又称 Continuous Learning, Never Ending Learning, Incremental Learning。人类在学习不同事情、在不同阶段的学习中都使用同一个大脑，但是在进行机器学习时，对于不同任务我们训练不同的模型。因此，我们很容易想到，我们能否对于所有任务都训练同一个模型，让这个模型最终能够拥有解决多个任务的能力。

为了实现这个目标，我们需要至少解决三个问题：

* **Knowledge Retention（知识保留）**
* **Knowledge Transfer（知识迁移）**
* **Model Expansion（模型扩张）**

## 知识保留

**Knowledge Retention（知识保留）**是要实现终身学习要面临的最难的一个问题。我们要求模型在不能忘记旧的知识的同时，还要持续学会新的东西。然而，在新的任务上更新参数后，同一个模型在原来的任务上的表现不可避免地会下降。我们将模型这种学会新任务就忘记旧任务的情况叫做 Catastrophic Forgetting（灾难性遗忘），因为和人类相比，机器的这种遗忘更加让人难以接受。下图形象地展示了模型在学习新任务后在旧任务上表现较差的原因。

![](https://raw.githubusercontent.com/bighuang624/pic-repo/master/Hung-yi-Lee-Catastrophic-Forgetting.png)

考虑到将任务作为序列输入时，模型无法保持对所有任务都有好的表现，我们会想到**多任务学习（Multi-Task Learning）**，让模型**同时**学习多个任务。这样，在其他任务上学习的经验还能够帮助模型在新任务上提升表现。

看起来多任务学习能够解决模型会遗忘过去的任务的问题。然而，就长远而言，如果我们要学习 1000 个任务，那么在模型学习第 1000 个模型时，需要保留前 999 个任务的所有数据，这样需要过于庞大的存储空间。然后，我们还需要同时把所有任务的训练数据放在一起，让模型同时学习，这样会导致海量的计算。因此，我们希望能够在不做多任务学习、不使用过去的数据的同时，让模型能够不遗忘已经学过的技能，这就是终身学习要探讨的问题。

一类经典的方法叫做**弹性权重巩固（Elastic Weight Consolidation，EWC）**，这类方法的核心思想是，模型中的部分参数对于过去的任务是重要的。当学习新任务时，我们保留这部分参数，只修改那些对过去的任务没有太大影响的参数。在 EWC 方法中，我们设 $\theta^b$ 是过去的任务学习完成后的模型参数，每一个参数 $\theta\_i^b$ 有一个“守卫”，它是一个用于代表该参数对过去任务的重要度的数值，用 $b\_i$ 表示。由此，我们的损失函数变为

$$ L^{\prime}(\theta)=L(\theta)+\lambda \sum\_{i} b\_{i}\left(\theta\_{i}-\theta\_{i}^{b}\right)^{2} $$

其中，$L(\theta)$ 是当前任务的 loss，$\theta\_i$ 是当前任务学完后得到的参数，而 $\theta\_{i}^{b}$ 是从过去任务中学得的参数。$\sum\_{i} b\_{i}\left(\theta\_{i}-\theta\_{i}^{b}\right)^{2} $ 可以被看作是一种另类的正则化项。

那么，$b\_i$ 如何计算得到？一种方法是，算对应参数 $\theta\_{i}$ 的二次微分。如果二次微分较小，说明改变 $\theta\_{i}$ 的影响比较小，因此可以给一个较小的 $b\_i$；而如果 $\theta\_{i}$ 的二次微分比较大，就需要给一个较大的 $b\_i$ 值，使得 $\theta\_{i}$ 不发生较大的变化。

![](https://raw.githubusercontent.com/bighuang624/pic-repo/master/Hung-yi-Lee-EWC-b-i-computation.png)

EWC 的效果对比如下图所示，可以看到普通的 SGD 容易在学习新任务的时候遗忘旧任务；L2 在防止模型遗忘的同时，会选择不去学习新的任务；而 EWC 可以有效地在学习新任务地同时，保持在旧任务上的表现：

![](https://raw.githubusercontent.com/bighuang624/pic-repo/master/Hung-yi-Lee-EWC-performance.png)

EWC 有很多变形，这里提供三篇相关论文：

* Elastic Weight Consolidation (EWC): [Overcoming catastrophic forgetting in neural networks](https://www.pnas.org/content/114/13/3521?__hstc=200028081.1bb630f9cde2cb5f07430159d50a3c91.1524182400081.1524182400082.1524182400083.1&__hssc=200028081.1.1524182400084&__hsfp=1773666937), 2017
* Synaptic Intelligence (SI): [Continual Learning Through Synaptic Intelligence](https://arxiv.org/pdf/1703.04200.pdf), ICML 2017
* Memory Aware Synapses (MAS): [Memory Aware Synapses: Learning what (not) to forget](https://arxiv.org/pdf/1711.09601.pdf), ECCV 2018
  * 计算二次微分需要有 loss，因此需要标签，但是这个方法的特别之处是不需要标签

### 生成历史数据的多任务学习

虽然多任务学习不能作为终身学习的解决方法，但是在终身学习的论文中，多任务学习经常被当作论文提出的方法的表现的上界。考虑到多任务学习非常好用，有人提出，既然机器的记忆容量有限，不能记忆历史数据，那么能不能训练一个能够生成历史数据的生成器，这样我们就不需要再保存历史数据，从而降低所需要的记忆容量。新任务到来时，我们就通过这个生成器生成历史数据，然后做多任务学习。

以下是两篇相关的论文：

* [Continual Learning with Deep Generative Replay](https://arxiv.org/pdf/1705.08690.pdf), NIPS 2017
* [FearNet: Brain-Inspired Model for Incremental Learning](https://arxiv.org/pdf/1711.10563.pdf), ICLR 2018

不过，考虑到目前的生成技术对于某些数据（例如高清视频）的效果一般，因此上述方法能否得到广泛应用尚待研究。

### 模型结构更改

在之前的讨论中，我们都假设不同的任务用的模型架构都是一样的。然而，一个容易遇到的问题是，新任务需要与旧任务不一样的结构（例如两个分类任务的类别数目不同），因此需要更改架构。这里有两篇相关论文作为参考：

* [Learning without Forgetting](https://arxiv.org/pdf/1606.09282.pdf), ECCV 2016
* [iCaRL: Incremental Classifier and Representation Learning](https://arxiv.org/pdf/1611.07725.pdf), CVPR 2017

## 知识迁移

我们希望模型不仅仅记住过去的知识，还希望模型能够在学习不同的任务时，将知识进行迁移。这样，在学习一个任务以后，我们会期待模型能够“触类旁通”，在学习后续任务时做的更好。

我们很容易想到，这就是迁移学习（Transfer Learning）的目标。区别在于，迁移学习**只**关心将知识迁移到新任务时能否提高模型在新任务上的表现，而在终身学习中，我们不仅希望提高模型在新任务上的表现，并且能够保持在历史任务中的好表现。

### 终身学习模型评估

如何评估一个终身学习模型的表现？通常我们会做一个表格如下图所示：

![](https://raw.githubusercontent.com/bighuang624/pic-repo/master/Hung-yi-Lee-Life-Long-Evaluation-2.png)

表格的横轴表示在不同任务上的测试表现，纵轴代表学习完成的任务。$R\_{i, j}$ 代表在任务 $i$ 上训练完成后，模型在任务 $j$ 上的表现。如果 $i > j$，代表我们想通过 $R\_{i, j}$ 来看模型在学习任务 $i$ 后，对任务 $j$ 还记得多少；如果 $i < j$，代表我们想通过 $R\_{i, j}$ 来看模型能否将在任务 $i$ 上的知识迁移到任务 $j$ 上。

之后，我们就可以在这个表格上定义一些评估指标，例如：

1. 准确率：Accuracy = $\frac{1}{T} \sum\_{i=1}^{T} R\_{T, i}$
2. 评估模型知识保留的能力（通常是一个负值）：Backward Transfer = $\frac{1}{T-1} \sum\_{i=1}^{T-1} R\_{T, i}-R\_{i, i}$
3. 评估模型知识迁移的能力：Forward Transfer = $\frac{1}{T-1} \sum\_{i=2}^{T} R\_{i-1, i}-R\_{0, i}$

![](https://raw.githubusercontent.com/bighuang624/pic-repo/master/Hung-yi-Lee-Life-Long-Evaluation-1.png)

如果有一个模型，能够使 Backward Transfer 是一个正值，说明它不仅不会遗忘，还能在学习新的任务后“触类旁通”。能够实现这点的一个模型是 **Gradient Episodic Memory (GEM)**，它在新的任务上算出梯度后，修改梯度的方向，希望能够以此提高在旧的任务上的表现。我们首先计算在新任务上的梯度的负方向 $g$，然后再计算旧的任务的梯度的负方向 $g^1, g^2$（说明我们从旧的任务的数据中进行了采样并存储），如果几个方向的内积是正的，梯度下降就按照 $g$ 的方向进行；如果不是，那么我们稍微调整 $g$ 的方向得到 $g^{'}$，使得 $g^{'} \cdot g^1 \geq 0$，$g^{'} \cdot g^2 \geq 0$ 的同时，$g$ 和 $g^{'}$ 的方向尽可能近。

![](https://raw.githubusercontent.com/bighuang624/pic-repo/master/Hung-yi-Lee-GEM.png)

视频中有同学提出，既然 GEM 需要旧任务的部分数据，可以用前文提到的用生成器生成历史数据的方法对其进行改进，使其不需要存储旧的数据。李老师表示想法不错，可以尝试。

两篇参考文献如下：

* [Gradient Episodic Memory for Continual Learning](https://arxiv.org/pdf/1706.08840.pdf), NIPS 2017
* [Efficient Lifelong Learning with A-GEM](https://arxiv.org/pdf/1812.00420.pdf), ICLR 2019

## 模型扩张

在上述讨论中，我们假设模型参数足够多，因此能够学习所有的新任务。但是，也有可能任务数量多过模型的极限，因此模型无法再学习。我们希望模型能够自动扩张大小，即在发现自己的学习能力不够时，自动生成一些神经元。同时，我们希望这种模型不是随意扩张，而是有效率的，即模型扩张的速度要小于新任务输入的速度。

![](https://raw.githubusercontent.com/bighuang624/pic-repo/master/Hung-yi-Lee-Progressive-Neural-Networks.png)

一个例子是论文 "[Progressive Neural Networks](https://arxiv.org/pdf/1606.04671.pdf)" 提出的 Progressive Neural Networks。它的方法是对每个任务建立一个模型，新任务对应的模型会将旧任务对应的模型的输出也当作输入。这种方法的缺点是当训练任务过多时，参数量会越来越多，最终难以负荷。

![](https://raw.githubusercontent.com/bighuang624/pic-repo/master/Hung-yi-Lee-Expert-Gate.png)

CVPR 2017 上的论文 "[Expert Gate: Lifelong Learning with a Network of Experts](https://arxiv.org/pdf/1611.06194.pdf)" 提出的 Expert Gate 也对每一个任务建立一个模型。为了实现知识迁移，Expert Gate 训练了一个任务检测器，每当一个新任务到达时，会检测这个新任务与哪个旧任务最像，将旧任务已经训练好的模型权重复制过来作为新模型对应模型的初始化。当然，这个方法仍然需要一个任务对应一个模型，需要较大的存储空间。

![](https://raw.githubusercontent.com/bighuang624/pic-repo/master/Hung-yi-Lee-Net2Net.png)

ICLR 2016 上的论文 "[Net2Net: Accelerating Learning via Knowledge Transfer](https://arxiv.org/pdf/1511.05641.pdf)" 提出 Net2Net，希望能够扩张模型。如果扩张模型的方法是直接增加新的神经元，可能会破坏原有模型对应的映射函数，从而使得模型忘记过去的任务。Net2Net 的方法在增加新神经元的同时，将原有神经元进行分裂，从而使得新的神经元和分裂后的旧神经元相同的同时，承担分裂前的神经元的工作。这种做法是使得增加新神经元前后的两个模型实际上是一样的，没什么作用，因此会在相关神经元的参数上加上一些小噪声，使得前后两个模型不同的同时，能够不影响模型在旧任务上的表现。这个方法不是每有一个新任务就增加神经元，而是在发现模型在新任务上的 loss 不会降低，准确率不会提高的时候才增加新神经元。

### 任务顺序对终身学习的影响

如果终身学习变成一个常见技术的话，我们会产生一个新的问题：任务的顺序应该怎么排列？我们会发现，如果先学习有噪音的任务 1，再学习没有噪音的任务 2，那么在学习完任务 2 时，模型会忘记从任务 1 中学到的知识。但是，如果先学习没有噪音的任务 2，再学习有噪音的任务 1，模型不但学会带噪音的任务，在处理没有噪音的旧任务时也能表现得很好。因此，任务的先后顺序，确实会对终身学习的效果产生影响。

CVPR 2018 的最佳论文 "[Taskonomy: Disentangling Task Transfer Learning](http://taskonomy.stanford.edu/taskonomy_CVPR2018.pdf)" 提出了 [Taskonomy](http://taskonomy.stanford.edu/#abstract)（任务学），旨在研究视觉任务之间的相互关联性，应该先学什么任务，后学什么任务，能够使得模型给出更好的表现。

## 参考资料

* 本节内容对应的 [PPT](http://speech.ee.ntu.edu.tw/~tlkagk/courses/ML_2019/Lecture/Lifelong%20Learning%20(v9).pdf)


<script type="text/x-mathjax-config">
MathJax.Hub.Config({
  tex2jax: {inlineMath: [ ['$', '$'] ],
        displayMath: [ ['$$', '$$']]}
});
</script>

<script type="text/javascript" src="https://cdn.bootcss.com/mathjax/2.7.2/MathJax.js?config=default"></script>