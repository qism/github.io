---
layout:     post
title:      attention is all you need
subtitle:   Google 论文 《attention is all you need》
date:       2019-10-28
author:     qism
header-img: img/post-bg-coffee.jpeg
catalog: true
tags:    
        - NLP
        - Deep Learing
        - attention
        - transformer
---

# 前序

attention模型一直在用，对原理的理解不深入，因此找了个时间好好研究一番。
看了很多网上大佬写的博客，以及谷歌的论文，经过多个小时的奋战，今天终于可以理直气壮地说自己懂attention啦~

话不多说，直接开始今日份的学习分享~

*说明下，本人并不是为了写博客而写博客，只是想记录下自己平时的学习笔记，因此，写得可能有点飘，不喜请移步其他大佬的博客，感谢你们~*

# 一、传统语言模型

传统语言模型，常见的有***CNN、RNN***等，典型的递归结构，当前token依赖其前置token，以此捕获句子中的序列关系。RNN无法实现并行，且很难获取全局信息，其本质是一个*马尔科夫决策过程*。基于此，RNN序列模型对句子的长度比较挑剔，尽管一些研究尝试去发现并提取不同位置的token间的关系，借助一些特征工程和条件计算，能够缓解但没有解决根本问题.

CNN可以实现并行，需要不断堆叠来获取全局信息。RNN和CNN的空间复杂度是*O(1)*

**************************************

# 二、Encoder - Decoder

在正式学习attention和transformer之前先了解一下encode-decoder框架，磨刀不误砍材工，相信我，非常有必要~ encode-decoder框架见下图：

![Encoder - Decoder](/img/encode-decoder.jpeg ''
 Encoder_Decoder框架'')

常用序列模型RNN用在encoder阶段，将一个句子（文章）编码成一个向量序列。

![](http://latex.codecogs.com/gif.latex?seq:<x_1,x_2,x_3,...,x_n>)


![](http://latex.codecogs.com/gif.latex?encoder:f(x_1,x_2,x_3,...,x_n))


![](http://latex.codecogs.com/gif.latex?encoder\_output/decoder_input)


![](http://latex.codecogs.com/gif.latex?decoder\_output:<y_1,y_2,y_3,...y_m>)


![](http://latex.codecogs.com/gif.latex?y_i=g(encoder\_output,y_1,y_2,...y_i-1)


每个yi都依次这么产生，那么看起来就是整个系统根据输入句子Source生成了目标句子Target。

在nlp领域，Encoder-Decoder的应用领域相当广泛。将该框架简单描述为由一个句子去生成另一个句子（机器翻译问题，对话问题等），或者由一个句子去生成一篇文章（文章生成问题等），或者由一篇文章去生成一个句子（文章摘要问题等）等。


Encoder-Decoder框架不仅仅在文本领域广泛使用，在语音识别、图像处理等领域也经常使用。比如对于语音识别来说，图2所示的框架完全适用，区别无非是Encoder部分的输入是语音流，输出是对应的文本信息；而对于“图像描述”任务来说，Encoder部分的输入是一副图片，Decoder的输出是能够描述图片语义内容的一句描述语。一般而言，文本处理和语音识别的Encoder部分通常采用RNN模型，图像处理的Encoder一般采用CNN模型。

attention是一种通用的思想，可以独立于任何框架，但是现阶段大多attention机制都是依附于Encoder-Decoder框架。

*******************************************************

# 三、Attention

encoder_output是整个句子的中间序列，对decoder阶段的不同词输出的作用应该是不一样的，这是理解attention的关键

在Encoder-Decoder框架上添加attention

在encoder每个单词时候添加权重
然后再用对各个单词的编码进行加权求和，为每一个输出单词的中间语义表现C，那么我们怎么知道这个权重呢？

source中每个单词的隐层状态与目标单词的前置token的隐层去一一匹配，通过一个F函数寻找对齐的可能性，这个对齐在机器翻译问题上是很好理解的。
这个F函数在不同论文里可能会采取不同的方法，具体有哪些方法我后期找找

![Encoder - Decoder-attention](/img/encoder-decoder-attention.jpeg ''加入attention后的Encoder_Decoder框架'')


![attention](/img/attention.jpeg ''attention机制'')

attention无序，加上position embedding,
seq2seq的输出attention是单向的
self-attention
multi-head attention

如果单看这个问题，attention的表现就比较好了，它可以抛弃位置依赖，获取任意位置token间的关系，没有距离限制

attention的一个好处是可以并行获取全局信息，如果不看position，attention就是一个词袋模型，它的空间复杂度是O(n)


问题：通过Self Attention到底学到了哪些规律或者抽取出了哪些特征呢？或者说引入Self Attention有什么增益或者好处呢？

很明显，引入Self Attention后会更容易捕获句子中长距离的相互依赖的特征，因为如果是RNN或者LSTM，需要依次序序列计算，对于远距离的相互依赖的特征，要经过若干时间步步骤的信息累积才能将两者联系起来，而距离越远，有效捕获的可能性越小。

但是Self Attention在计算过程中会直接将句子中任意两个单词的联系通过一个计算步骤直接联系起来，所以远距离依赖特征之间的距离被极大缩短，有利于有效地利用这些特征。除此外，Self Attention对于增加计算的并行性也有直接帮助作用。这是为何Self Attention逐渐被广泛使用的主要原因。

********************************************************

# 三、transformer


对模型结构的解释：

encoder
6层，每一层还有2个子层
decoder 

transformer只依赖attention机制，获取输入和输出间的全局依赖，能够更好地实现并行，在翻译任务上表现好于之前的模型


参考
https://blog.csdn.net/malefactor/article/details/78767781
讲得真好，一看就明白，已经粉他了~~
另外就是苏神的文章，角度新颖，一起观赏下

********************************************************

# 后记

代码实现：

等我几天，我整理一下放上来


























































