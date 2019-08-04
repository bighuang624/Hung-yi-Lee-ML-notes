<!-- 现在我们都是宅宅工程师了 -->

## Hyperparameters Search

Random Search 通常能够比 Grid Search 在更短时间内获得较好的参数组合。有一些方法能够帮助自动决定超参数，其中一种是 Bayesian Optimization。

![](https://blog.nanonets.com/content/images/2019/03/Unknown-5.png)

## AutoML

![](https://raw.githubusercontent.com/bighuang624/pic-repo/master/Hung-yi-Lee-AutoML.png)

AutoML 可以看作元学习（meta learning）的一种。AutoML 的做法是学习一个有点像 Seq2Seq 的神经网络模型，每一次输出子模型架构的一个参数的值。然后通过强化学习（Reinforcement Learning）来让这个会设计模型的模型越来越好。当然，我们会想到这个会设计模型的模型又是由谁来设计呢？所以，最后可能还是需要人工来设计网络。

![](https://raw.githubusercontent.com/bighuang624/pic-repo/master/Hung-yi-Lee-AutoML-Optimizers-Structure.png)

现有的优化器大多都可以看作是三种操作的组合：Operands（图中蓝色圆圈）, Unary functions（图中黄色方框）和 Binary functions（图中紫色方框）。因此，我们也可以用 AutoML 的方式来选择要选择哪些操作，从而设计出一种新的优化器。

同理，我们也可以用上述方法设计出新的激活函数。这样，将所有环节拼接在一起，机器完全可以自己设计模型结构。但是，这种方式需要极大的计算量。

论文 "[Efficient Neural Architecture Search via Parameter Sharing](https://arxiv.org/pdf/1802.03268.pdf)" 提出只需要一张 Nvidia GTX 1080Ti 的 GPU，在 16 小时内就能完成训练。之前每次需要重新训练一个新的神经网络，该文提出，如果这个新的神经网络内部的某些组件之前已经有网络训练好，就拿来做初始化，而非从随机初始化开始训练。

## 参考资料

* 本节内容对应的 [PPT](http://speech.ee.ntu.edu.tw/~tlkagk/courses/MLDS_2018/Lecture/Learn2learn.pdf)
* [【神经网络架构搜索相关资源大列表】awesome-NAS - A curated list of neural architecture search (NAS) resources by D-X-Y GitHub](https://github.com/D-X-Y/awesome-NAS)


<script type="text/x-mathjax-config">
MathJax.Hub.Config({
  tex2jax: {inlineMath: [ ['$', '$'] ],
        displayMath: [ ['$$', '$$']]}
});
</script>

<script type="text/javascript" src="https://cdn.bootcss.com/mathjax/2.7.2/MathJax.js?config=default"></script>