## LSTM for Gradient Descent

![](https://raw.githubusercontent.com/bighuang624/pic-repo/master/Hung-yi-Lee-learning-algorithm-looks-like-RNN.png)

重看上图的学习算法过程，容易产生一种感觉：这个过程非常像 RNN，参数的每一次更新就像是 RNN 的一个时间步，参数可以看作 RNN 的 memory。

把整个梯度下降学习算法当作 LSTM 的技术主要出自以下两篇论文：

* S Ravi, H Larochelle, "[Optimization as a model for few-shot learning](https://openreview.net/pdf?id=rJY0-Kcll)", ICLR, 2017
* M Andrychowicz, M Denil, S Gomez, et al., "[Learning to learn by gradient descent by gradient descent](http://papers.nips.cc/paper/6461-learning-to-learn-by-gradient-descent-by-gradient-descent.pdf)", NIPS, 2016

先放一张 LSTM 的示意图：

![](https://raw.githubusercontent.com/bighuang624/pic-repo/master/Hung-yi-Lee-LSTM.png)

在 LSTM 中，$c^t$ 的更新公式为

$$c^{t}=z^{f} \odot c^{t-1}+z^{i} \odot z$$

而 MAML 中有

$$\theta^{t}=\theta^{t-1}-\eta \nabla\_{\theta} l$$

那么，如果我们将 $c^{t}$ 看作 $\theta^{t}$，$c^{t-1}$ 看作 $\theta^{t-1}$，$z^{f}$ 看作 $[1, 1, \dots, 1]^T$，$z^{i}$ 看作 $[\eta, \eta, \dots, \eta]^T$，$z$ 看作 $-\nabla\_{\theta} l$，则第一个公式就变成了第二个公式。由此，我们容易想到，在这里的 MAML 中，我们也有输入门和遗忘门，只是 $z^{f}$ 和 $z^{i}$ 是我们指定的数值。能否让它们像 LSTM 一样，从 $-\nabla\_{\theta} l$ 和其他信息（例如，根据 $\theta^{t-1}$ 算出来的 loss）中学习得到？

如果我们可以做到，那么显然，$z^{i}$ 就是一个动态的学习率，而 $z^{f}$ 把原来的参数进行放缩。

因此，我们有了 LSTM for Gradient Descent：

![](https://raw.githubusercontent.com/bighuang624/pic-repo/master/Hung-yi-Lee-LSTM-for-Gradient-Descent.png)

注意每次的梯度 $-\nabla\_{\theta} l$ 是不同的（虽然在上图中用的符号表示是一样的）。

![](https://raw.githubusercontent.com/bighuang624/pic-repo/master/Hung-yi-Lee-Typical-and-gradient-descent-LSTM.png)

传统的 LSTM 和这里的 LSTM for Gradient Descent 存在的一点区别是，传统 LSTM 中，$c$ 和 $x$ 是相互独立的，而在 LSTM for Gradient Descent 中，输入的梯度信息 $-\nabla\_{\theta} l$ 的计算是与现在的参数 $\theta$ 是有关系的，而反向传播也会通过两条路影响 $\theta$。在实际实现中，我们还是当作两者相互独立来训练

还有一点是，在 LSTM for Gradient Descent 中，memory 的初始值 $\theta^0$ 也可以被当作是参数，一起学习出来。

### 具体实现

在以上讨论中，LSTM 的 memory cell 的值就是神经网络的参数。有一个问题是，神经网络的参数量可能非常大，这样 LSTM for Gradient Descent 难以训练。在实际实现时，LSTM for Gradient Descent 只有一个 cell，所有参数共用同一个 LSTM for Gradient Descent。这样的方法是合理的，因为我们在传统的梯度下降方式中，也是所有的参数都遵循同一套更新规则。同时，MAML 中要求训练时和测试时的模型架构需要一致，但如果所有的参数都用同一个 LSTM for Gradient Descent 来更新，可以允许训练时和测试时的模型架构不同。

通过下图的实验结果可以看出，作为遗忘门的 $z^{f}$ 实际上基本保持着一个略小于 1 的恒定值，而作为输入门的 $z^{i}$ 的值一直在动态变化。

![](https://raw.githubusercontent.com/bighuang624/pic-repo/master/Hung-yi-Lee-LSTM-for-Gradient-Descent-experiemtal-results.png)

### 另一版本

人们设计的常用梯度下降的方法，我们在决定学习率时，不是只看现在的梯度，还会考虑过去的梯度，例如 RMSProp、Momentum 等。一个新版本的 LSTM for Gradient Descent 根据这个思想被提出，如下图所示：

![](https://raw.githubusercontent.com/bighuang624/pic-repo/master/Hung-yi-Lee-LSTM-for-Gradient-Descent-v2.png)

我们在将梯度信息输入 LSTM for Gradient Descent 之前，又加了一个新的 LSTM。我们可以认为这个新增的 LSTM 的作用就是处理过去的梯度。实验结果如下图所示：

![](https://raw.githubusercontent.com/bighuang624/pic-repo/master/Hung-yi-Lee-LSTM-for-Gradient-Descent-v2-experiemtal-results.png)

可以看到，新版的 LSTM for Gradient Descent 的效果比几种常用的优化算法都要好。值得注意的是第二行的三个结果。论文中在训练阶段使用的模型架构含有 20 个神经元，而到测试阶段时，将模型架构换成含有 40 个神经元，或者换成两层的结构，可以看到效果依然很好，说明 LSTM for Gradient Descent 允许训练时和测试时的模型架构不同。但是将激活函数更换后，效果会变得很差。

## 参考资料

* 本节内容对应的 [PPT](http://speech.ee.ntu.edu.tw/~tlkagk/courses/ML_2019/Lecture/Meta2%20(v4).pdf)



<script type="text/x-mathjax-config">
MathJax.Hub.Config({
  tex2jax: {inlineMath: [ ['$', '$'] ],
        displayMath: [ ['$$', '$$']]}
});
</script>

<script type="text/javascript" src="https://cdn.bootcss.com/mathjax/2.7.2/MathJax.js?config=default"></script>