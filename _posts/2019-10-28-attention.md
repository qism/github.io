---
layout:     post
title:      Attention
subtitle:   自注意力机制
date:       2019-10-28
author:     qism
header-img: img/post-bg-coffee.jpeg
catalog: true
tags:    
        - NLP
        - Deep Learing
        - attention
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


![](http://latex.codecogs.com/gif.latex?y_i=g(encoder\_output,y_1,y_2,...y_i-1))

*这部分详细内容可以参考张博士的博客*

以翻译任务为例，输入英文source：「Tom loves Jimmy」,输出target：「汤姆爱慕吉米」。encoder阶段编码整个source，例如用RNN将seq编码成一个中间语义序列c，decoder阶段逐个解码target的每个yi，每次解码的输入是c和前一个token的解码结果，即y_i-1。如果y_i是第一个字符，则decoder输出是c和一个起始符。

上述任务看起来就是***整个系统根据输入句子Source生成了目标句子Target***，这就是一个典型的encode-decoder框架的应用。

在nlp领域，Encoder-Decoder的应用领域相当广泛。将该框架简单描述为由一个句子去生成另一个句子（机器翻译问题，对话问题等），或者由一个句子去生成一篇文章（文章生成问题等），或者由一篇文章去生成一个句子（文章摘要问题等）等。Encoder-Decoder框架也可用在语音和图像领域，比如语音识别，Encoder部分的输入是语音流，输出是对应的文本信息；对「图像描述」任务，Encoder部分的输入是图片，Decoder的输出是能够描述图片语义内容的一句描述语。一般而言，文本处理和语音识别的Encoder部分通常采用RNN模型，图像处理的Encoder一般采用CNN模型。

*******************************************************

# 三、Attention

## 从「翻译任务对齐」理解attetion

Attention（注意力机制）其实是一种通用的思想，可以独立于任何框架，但是现阶段大多attention机制都是依附于Encoder-Decoder框架。Encoder-Decoder上一节已经讲过了。

首先来看上面这个翻译任务：输入英文source：「Tom loves Jimmy」,输出target：「汤姆爱慕吉米」。传统的Encoder-Decoder框架，在decoder阶段的一个输入前置token，另一个输入是固定的，为整个source的编码语义表示（encoder_output）,但是，对于预测不同target中的y_i的重要性是不一样的，例如要decoder「汤姆」，明显「Tom」的重要性应该更大些，换句话说，翻译时候能够体现***token『对齐』***。

***这个很直观，source的中间语义（encoder_output）在不同的decoder阶段应该不一样，体现不同token的重要性，这就是理解attention的关键了~~***

在Encoder-Decoder框架上添加attention，也就是下面这张图：

![Encoder - Decoder-attention](/img/encoder-decoder-attention.jpeg ''加入attention后的Encoder_Decoder框架'')

--------------------------------

于是乎，decoder阶段就变成了下面这样的形式：

![](http://latex.codecogs.com/gif.latex?y_1=g(c_1))

![](http://latex.codecogs.com/gif.latex?y_2=g(c_2,y_1))

![](http://latex.codecogs.com/gif.latex?y_i=g(c_i,y_i-1))

***那么![](http://latex.codecogs.com/gif.latex?c_i)怎么来？***

常用的方法就是对每个token的隐状态进行加权求和，得到![](http://latex.codecogs.com/gif.latex?c_i)

对于上述翻译任务，就有下面这样3个c:

![](http://latex.codecogs.com/gif.latex?c_{汤姆}=a_{汤姆,Tom}f(h_{Tom})+ a_{汤姆,loves}f(h_{loves}) + a_{汤姆,Jimmy}f(h_{Jimmy}) )

![](http://latex.codecogs.com/gif.latex?c_爱慕=a_{爱慕,Tom}f(h_{Tom})+ a_{爱慕,loves}f(h_{loves}) + a_{爱慕,Jimmy}f(h_{Jimmy}) )

![](http://latex.codecogs.com/gif.latex?c_{吉米}=a_{吉米,Tom}f(h_{Tom})+ a_{吉米,loves}f(h_{loves}) + a_{吉米,Jimmy}f(h_{Jimmy}) )

也就是

![](http://latex.codecogs.com/gif.latex?c_i=\sum_{j=1}^{seq_len}
a_{ij}h_j)


source中每个单词的隐层状态与目标单词的前置token的隐层去一一匹配，通过一个F函数寻找对齐的可能性，这个对齐在机器翻译问题上是很好理解的。


***那么问题来了，我们怎么知道这个权重呢？***


以下引用张博士的博客内容：**假设编码和解码端都使用RNN,可以用Target输出句子i-1时刻的隐层节点状态去一一和输入句子Source中每个单词对应的RNN隐层节点状态hj进行对比，即通过函数![](http://latex.codecogs.com/gif.latex?F(h_j,H_{i-1}) ),其中，H为target中前置token的隐状态来获得目标单词和每个输入单词对应的对齐可能性，这个F函数在不同论文里可能会采取不同的方法，然后函数F的输出经过Softmax进行归一化就得到了符合概率分布取值区间的注意力分配概率分布数值。**

这个F函数在不同论文里可能会采取不同的方法，我常用的方法就是除以key的维度开根号后的值，再做一次softmax（谷歌论文中self-attention中的做法）

其他还有哪些方法我后期找找

## attention的更一般形式

![attention](/img/attention.jpeg ''attention机制'')

<Key,Value>为Source，Query为Target。对某个query，首先计算Query和各个Key的相似性（相关性）(方法有点积、向量cos或者MLP等)，得到各个Key对应Value的权重系数，然后对Value进行加权求和，即得到了最终的Attention数值。所以本质上Attention机制是对Source中元素的Value值进行加权求和，而Query和Key用来计算对应Value的权重系数。

## Self-Attention

Self-Attention(自注意力机制)即query=key=value的情况，这时候考虑的是Source或target内部元素间的Attention机制，计算方式和过程跟一般形式的attention是一样的，在此不赘述。


## Multi_Head_Attention




### 问题：

***Self Attention 的作用是什么？Self Attention 能够学习到什么特征？***

很明显，引入Self Attention能够捕捉任意两个词间的关联关系，而不受词间的距离的限制，轻松实现并行，相比RNN通过递归，依次计算序列依赖，对于短文本，问题不大，随着词间距离的增加，捕获词间数据关联性的可能性越小。

***self-attention捕获任意两个词间的关联依赖关系，但是有没有发现忽略了语序关系？***

如果只看第一个问题，attention的表现就比较好了，它可以抛弃位置依赖，获取任意位置token间的关系，没有距离限制，但是，忽略了词序和位置依赖的attention就是一个词袋模型，它的空间复杂度是O(n)

于是乎，谷歌论文中还引入了position embedding

## position embedding


********************************************************


参考
https://blog.csdn.net/malefactor/article/details/78767781
讲得真好，一看就明白，已经粉他了~~
另外就是苏神的文章，角度新颖，一起观赏下

********************************************************

# 后记

代码实现：用bilstm+attention实现虚假新闻识别
[git地址]()

核心代码：
```python
class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
        self.lstm = rnn_encoder(config.embed, config.hidden_size* 2, config.encoder_mode)
        self.tanh1 = nn.Tanh()
        self.w = nn.Parameter(torch.Tensor(config.hidden_size * 2))
        self.tanh2 = nn.Tanh()
        self.fc1 = nn.Linear(config.hidden_size * 2, config.hidden_size2)
        self.fc = nn.Linear(config.hidden_size2, config.num_classes)  
    
    def forward(self, x, mask):
        batch_size, time_step = x.size()
        emb = self.embedding(x) 
        H = self.lstm(emb,mask).expand(batch_size,time_step,-1)  
        M = self.tanh1(H)  
        alpha = F.softmax(torch.matmul(M, self.w), dim=1).unsqueeze(-1) 
        out = H * alpha 
        out = torch.sum(out, 1) 
        out = F.relu(out)
        out = self.fc1(out) 
        out = self.fc(out) 
        return out
```

```python
#
def forward(self, Q, K, V, scale=None):
    '''

    Args:

        Q: [batch_size, len_Q, dim_Q]

        K: [batch_size, len_K, dim_K]

        V: [batch_size, len_V, dim_V]

        scale: 缩放因子 论文为根号dim_K  为什么缩放？

    Return:

        self-attention后的张量，以及attention张量

    '''

    attention = torch.matmul(Q, K.permute(0, 2, 1))
    if scale:
        attention = attention * scale   
    attention = F.softmax(attention, dim=-1)
    context = torch.matmul(attention, V)
    return context
```
























































