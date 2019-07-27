这里讨论两个关于 Autoencoder 的问题：

* 为什么要最小化重构误差？有没有其他做法？
* 怎么让学习到的表征更具可解释性？

## Autoencoder 的另一种形式

显然，我们期待学习到的表征具有代表性。那么我们现在有一个编码器（encoder），想要评估它的质量，就需要知道这个编码器输出的表征是否具有代表性。因此，我们训练一个判别器（discriminator），可以认为是一个二元分类器（binary classifier），每次输入一张图片和一个通过编码器得到的表征，分辨它们是否是对应的。设判别器的参数为 $\phi$，那么我们训练 $\phi$ 来最小化这个分类任务的损失 $L\_D$，来得到损失的最小值 $L\_{D}^{'}=\min \_{\phi} L\_{D}$。如果 $L\_{D}^{'}$ 很小，说明训练的结果很好，认为表征非常具有代表性，二元分类器可以容易地判断哪些图片和表征是对应的；如果 $L\_{D}^{'}$ 很大，说明不同图片得到的表征也很相似，因此学习到的表征不具有代表性。

![](https://raw.githubusercontent.com/bighuang624/pic-repo/master/Hung-yi-Lee-Autoencoder-beyond-reconstruction.png)

根据以上分析，我们可以提出一个新的训练编码器的方法，那就是训练编码器，让已经训练好的这个判别器进行评估时可以得到好的结果，即训练编码器 $\theta$ 使得 $L\_{D}^{'}$ 最小，因此有 $\theta^{'}=\arg \min \_{\theta} L\_{D}^{'} = \arg \min \_{\theta} \min \_{\phi} L\_{D}$。

从这个式子可以看出，我们要同时训练一个最好的编码器 $\theta$ 和一个最好的判别器 $\phi$ 来一起最小化 $L\_{D}$。这种技术被被 ICLR 2019 上的论文 "[Learning deep representations by mutual information estimation and maximization](https://arxiv.org/pdf/1808.06670.pdf)" 提出的 Deep InfoMax (DIM) 方法所使用。

我们会发现，这个形式与通常的 Autoencoder 中，我们要同时训练编码器和解码器来最小化重构误差的情况非常相似。因此，其实我们熟悉的 Autoencoder 就是上述使用判别器方法的一个特例。上述方法中，判别器将一张图片和一个表征向量作为输入，输入一个分数值来表示图片和表征向量是否对应。我们假设判别器的运作方式是其内部有一个解码器，这个解码器把表征向量作为输入，输出一张与判别器输入图片大小相同的图片，然后将生成的这张图片与判别器输入图片相减，那么输出的分数实际就是重构误差。只不过 Autoencoder 中我们只考虑正例（positive examples），而使用判别器时我们还会考虑负例（negative examples），也就是输入的图片与表征向量不对应的情况。

## 序列数据

![](https://raw.githubusercontent.com/bighuang624/pic-repo/master/Hung-yi-Lee-sequential-data.png)

如果数据是序列数据，那么可以做更多事。例如，可以训练一个模型，输入是当前的句子，模型输出这个句子的前一个句子和后一个句子。这种方法被 NIPS 2015 上的论文 "[Skip-Thought Vectors](https://papers.nips.cc/paper/5950-skip-thought-vectors.pdf)" 称为 **skip thought**。这个概念和 word embedding 的训练有些相似，训练 word embedding 时，如果两个不同的词汇，它们的上下文十分相似，这两个不同的词汇的语义是相似的。skip thought 把这个思路扩展到句子的级别。

ICLR 2018 上的论文 "[An efficient framework for learning sentence representations](https://arxiv.org/pdf/1803.02893.pdf)" 又扩展 skip thought 得到 **quick thought**。skip thought 的训练计算时间比较大，因为要训练编码器和解码器，解码器还要生成前一个句子和后一个句子。在 quick thought 中，不再需要编码器，只需要训练解码器。思想是让编码器从当前句子和下一个句子学到的 embedding 更相似，因此除开下一个句子，还要随机采样几个句子，所有句子得到的 embedding 输入到一个分类器来判断哪一个句子是下一个句子。随机采样句子的原因是避免编码器将所有输入句子都编码成一样的，因此要让当前句子和随机采样句子的 embedding 区别更大。

其他技术也采用过类似 quick thought 的这种思想，例如论文 "[Representation Learning with Contrastive Predictive Coding](https://arxiv.org/pdf/1807.03748.pdf)" 中提出的 Contrastive Predictive Coding (CPC)。

## 更具可解释性的表征

这里要讲的第一个概念是 **feature disentangle**，"disentangle" 直译是“解开”。我们要编码的对象都包含了很多方面的信息，例如一段语音，它包含了语音内容、说话者的声音、环境噪音等；对于一段文字，包含了语义、文法的信息等。当我们使用一个表征向量来表示这个对象时，我们不知道哪些向量的哪些维度表示了哪些方面的信息。如果编码器能够将表征向量每个维度表示的信息都清楚地区分开来，那么这个表征的可解释性就会更强。

这一小节中我们接下来都会以语音为例来进行解释。根据上述思想，我们可能希望编码一段语音时，得到的表征向量的前一半维度代表语音内容信息，后一半维度代表说话人的声音信息（简单起见，我们这里假设只有语音只包含两方面的信息）。这个想法也可以有一个变形，就是直接用两个编码器来编码同一段语音，一个编码器专门编码这段语音的内容信息，另一个专门编码说话人的声音信息，然后我们将两个编码器输出的向量进行拼接。

这样有什么作用？如果我们可以将一段语音中不同方面的信息在表征中完全分类开，我们就可以实现变声器，用 A 的声音说 B 说过的话。

### 对抗式训练

![](https://raw.githubusercontent.com/bighuang624/pic-repo/master/Hung-yi-Lee-feature-disentangle-adversarial-training.png)

那么，如何实现 feature disentangle？其中一种做法是使用对抗式训练（adversarial training）。下图的例子中，我们将编码器得到的表征向量的前一半维度输入到一个针对说话人性别的分类器中。我们希望编码器学会“欺骗”分类器，即让分类器的正确率越低越好，这样编码器就会学习把说话人的声音信息放在表征向量的后一半维度中。在 GAN 的思想中，这个分类器也就是一个判别器，我们需要交替地训练判别器和编码器。

### 设计新的架构

![](https://raw.githubusercontent.com/bighuang624/pic-repo/master/Hung-yi-Lee-feature-disentangle-designed-network-architecture.png)

更简洁的方式是设计新的编码器架构，能够过滤不要的信息，只包含我们需要的信息。例如，我们可以给第一个编码器加上 instance normalization（IN），这是一个特别设计的层，能够过滤掉一些全局的、处理对象的每一个部分都有的信息。这样，第一个编码器在处理语音时就能够过滤掉说话人的声音信息，只保留内容信息。

如果要实现实现变声器，我们就需要第二个编码器只保留说话人的声音信息。这里，我们可以在**解码器**上加上一个 adaptive instance normalization（AdaIN），而将第二个编码器的输出在输入解码器时先经过这个层。在 AdaIN 层中，解码器首先通过 IN 层对全局信息进行正则化，然后让第二个编码器来提供这些全局信息。第二个编码器能够通过调整输出的全局信息改变整个句子的所有部分，因此会学习将语音内容的信息过滤掉，因为如果保留，语音内容会改变整个句子，导致无法重构。这样，解码器将第一个编码器的输入当作通常的输入，而将第二个编码器的输出做特别的处理，最后重构语音信号。

上述内容出自 Interspeech 2019 上的论文 "[One-shot Voice Conversion by Separating Speaker and Content Representations with Instance Normalization](https://arxiv.org/pdf/1904.05742.pdf)"。

## 离散表征

编码器过去得到的 embedding 都是低维连续向量。我们现在考虑编码器能否输出离散的表征（向量的每一维的值不是 0 就是 1）。这样，解释这个表征或者对其做聚类都会更容易。将连续向量变为离散向量其实比较简单，我们可以通过只把数值最大的一维变为 1 而将其他维度的值都变为 0 来把连续向量变为 one-hot 向量，也可以为每个维度的值划定一个阈值，高于这个阈值的值变为 1，低于这个阈值的值变为 0，将其变为一个 binary 向量。之后，我们再对这个离散表征向量做重构。

问题在于，我们提到的将连续向量变为离散向量的方法无法微分，因此我们通过通常的训练方式是无法微分的。ICLR 2017 上的论文 "[Categorical Reparameterization with Gumbel-Softmax](https://arxiv.org/pdf/1611.01144.pdf)" 提出了一些技巧能够帮助进行上述的端到端训练。

将 one-hot 向量和 binary 向量相比较，我们会认为 binary 向量更好。一方面，表示同样的信息，binary 向量所需要的维度更少。1024 维度的 one-hot 向量，用 bianry 向量表示只需要 10 维。另一方面，bianry 向量有机会处理训练数据中没有出现过的类别，意思是同等维数下，bianry 向量所能表示的类别更多，可能有一些类别再训练集中没有出现，但在测试集中出现了，使用 bianry 向量就不用担心处理不了这些类别。

![](https://raw.githubusercontent.com/bighuang624/pic-repo/master/Hung-yi-Lee-VQ-VAE.png)

一种学习离散表征的著名方法是 NIPS 2017 上的论文 "[Neural Discrete Representation Learning](https://arxiv.org/pdf/1711.00937.pdf)" 所提出的 **Vector Quantised-Variational AutoEncoder (VQ-VAE)**。VQ-VAE 的做法是，在编码器和解码器之外，还有一个 Codebook，Codebook 里是一组向量，这些向量也是学习得到的。当编码器将一张输入的图片变为表征向量后，我们将这个向量与 Codebook 里每个向量计算相似度，取出相似度最高的那个向量作为解码器的输入。这种方法其实也不能微分，请看原论文中训练这个无法微分的网络的技巧。

根据论文 "[Unsupervised speech representation learning using WaveNet autoencoders](https://arxiv.org/pdf/1901.08810.pdf)" 所述，学出来的离散表征会保留离散的信息（例如一段语音中的文字内容），而全局的信息（例如说话人的声音信息、持续的噪声等）就会被过滤掉。

### Sequence as Embedding

我们甚至可以考虑让 embedding 不再是一个向量。例如，我们让编码器和解码器都是 Seq2Seq 结构，然后输入一篇文章，编码器将其编码为一个词序列，然后解码器再将这个词序列重构为原文章。这个词序列可能就是原文章的摘要。不过，直接训练编码器和解码器可能无法得到人类能够阅读的摘要，因为编码器和解码器只需要学习到同一套“解读暗号”就行。想让用这种方式生成的摘要能够让人类阅读理解，我们需要使用 GAN。EMNLP 2018 上的论文 "[Learning to Encode Text as Human-Readable Summaries using Generative Adversarial Networks](https://arxiv.org/pdf/1810.02851.pdf)" 提出，将这个词序列输入到一个二分类器，判断这个词序列是否是人类写的。然后训练编码器（此时也可以说是生成器）让它“欺骗”判别器来认为作为表征的词序列是人类写出来的。这样对抗式训练过后，最终得到的词序列就会比较自然，人类能够理解。注意，这里的中间表征是词序列，也是离散的，因此整个过程也不能微分，实际训练要采用强化学习。如果有一个损失函数无法微分，可以把损失函数当作 reward，把网络当作 agent，用强化学习硬做。

### Tree as Embedding

![](https://raw.githubusercontent.com/bighuang624/pic-repo/master/Hung-yi-Lee-Autoencoder-tree-as-embedding.png)

除开词序列，我们还可以让树作为 embedding。两篇代表性的文章为 ACL 2018 上的论文 "[StructVAE: Tree-structured Latent Variable Models for Semi-supervised Semantic Parsing](https://arxiv.org/pdf/1806.07832.pdf)"，以及 NAACL 2019 上的论文 "[Unsupervised Recurrent Neural Network Grammars](https://arxiv.org/pdf/1904.03746.pdf)"。

## 参考资料

* 本节内容对应的 [PPT](http://speech.ee.ntu.edu.tw/~tlkagk/courses/ML_2019/Lecture/Auto%20(v3).pdf)
* [深度学习的互信息：无监督提取特征 - 科学空间|Scientific Spaces](https://kexue.fm/archives/6024)
* Skip-Thought Vectors 相关论文
  * Shuai Tang, Hailin Jin, Chen Fang,
et al., "[Trimming and Improving Skip-thought Vectors](https://arxiv.org/pdf/1706.03148.pdf)". arXiv, 2017
  * Afroz Ahamad, "[Generating Text through Adversarial Training Using Skip-Thought Vectors](https://www.aclweb.org/anthology/N19-3008)". NAACL-HLT (Student Research Workshop) 2019
* [连续特征的离散化：在什么情况下将连续的特征离散化之后可以获得更好的效果？ - 知乎](https://www.zhihu.com/question/31989952)


<script type="text/x-mathjax-config">
MathJax.Hub.Config({
  tex2jax: {inlineMath: [ ['$', '$'] ],
        displayMath: [ ['$$', '$$']]}
});
</script>

<script type="text/javascript" src="https://cdn.bootcss.com/mathjax/2.7.2/MathJax.js?config=default"></script>