---
layout:     post
title:      Transformer
subtitle:   Google 论文 《attention is all you need》
date:       2019-11-07
author:     qism
header-img: img/post-bg-coffee.jpeg
catalog: true
tags:    
        - NLP
        - Deep Learing
        - attention
        - transformer
---

# 一、transformer

## transformer的框架

见下图：

![transformer](/img/transformer.jpeg ''图解transformer'')

***Transformer模型也是使用经典的encoer-decoder架构，由encoder和decoder两部分组成。上图的左半边用Nx框出来的，就是我们的encoder的一层。encoder一共有6层这样的结构。上图的右半边用Nx框出来的，就是我们的decoder的一层。decoder一共有6层这样的结构。输入序列经过word embedding和positional encoding相加后，输入到encoder。输出序列经过word embedding和positional encoding相加后，输入到decoder。最后，decoder输出的结果，经过一个线性层，然后计算softmax。***

encoders包含6层，每一层encoder还有2个子层，第一部分是一个multi-head self-attention mechanism
第二部分是一个position-wise feed-forward network，是一个全连接层
两个部分，都有一个 残差连接(residual connection)，然后接着一个Layer Normalization。每个encoder除了参数不共享，网络结构和参数数量都是相同的。

decoders也包含6个decoder，每一个层包括以下3个部分：第一个部分是multi-head self-attention mechanism；第二部分是multi-head context-attention mechanism；第三部分是一个position-wise feed-forward network。还是和encoder类似，上面三个部分的每一个部分，都有一个残差连接，后接一个Layer Normalization。

但是，decoder出现了一个新的东西multi-head context-attention mechanism。这个东西其实也不复杂，理解了multi-head self-attention你就可以理解multi-head context-attention。中间还内嵌了一个encoder-decoder attention

![框架示意](/img/图解encode-decoder.jpeg ''图解encode-decoder'')

Transformer中的每个Encoder接收一个512维度的向量的列表作为输入，然后将这些向量传递到‘self-attention’层，self-attention层产生一个等量512维向量列表，然后进入前馈神经网络，前馈神经网络的输出也为一个512维度的列表，然后将输出向上传递到下一个encoder。

其中，底层encoder实现word embedding（假如维度为512）,然后输出至self attention和feed forward，输出同样维度的向量作为下一个encoder的输入。

6层encoder的输出为***一组attention的集合（K,V）***，将被输入至***每个***decoder的「encoder-decoder attetion」(有助于decoder捕获输入序列的位置信息)

decoder底层也是Word embedding + position embedding输入

Decoder中的self attention不同于Encoder的self attention，在Decoder中，self attention***只关注输出序列中的较早的位置***。这是在self attention计算中的softmax步骤之前屏蔽了特征位置（设置为 -inf）来完成的。

“Encoder-Decoder Attention”层的工作方式与"Multi-Headed Self-Attention"一样，只是它***从下面的层创建其Query矩阵，并在Encoder堆栈的输出中获取Key和Value的矩阵。***

***decoder层语句的self-attention对未来的词进行mask ,Encoder-Decoder Attention的query来自未来要预测的词***

「在Encoder堆栈的输出中获取Key和Value的矩阵」这句话怎么理解？

上面两层怎么做连接的？

最后decoder的输出连接一个全连接神经网络，再做一层softmax ,得到预测的单词（在翻译任务上）

---------------------------------------------
## self attention

计算self attention的第一步是从每个Encoder的输入向量上创建3个向量（在这个情况下，对每个单词做词嵌入）。所以，对于每个单词，我们创建一个Query向量，一个Key向量和一个Value向量。这些向量是通过词嵌入乘以我们训练过程中创建的3个训练矩阵而产生的。

注意这些新向量的维度比嵌入向量小。我们知道嵌入向量的维度为512，而这里的新向量的维度只有64维。新向量并不是必须小一些，这是网络架构上的选择使得Multi-Headed Attention（大部分）的计算不变。

***想了解具体，请看前一篇《attention》***
----------------------------------------------
## Residuals

残差结构的作用是，通过增加了一项x，使得该层网络对x求偏导的时候，多了一个常数项1！所以在反向传播过程中，梯度连乘，也不会造成梯度消失！

![Residuals](/img/transformer_resideual_layer_norm_2.jpeg ''图解encode-decoder的Residuals'')

## layer-normalization



encode

transformer只依赖attention机制，获取输入和输出间的全局依赖，能够更好地实现并行，在翻译任务上表现好于之前的模型


seq2seq的输出attention是单向的
self-attention
multi-head attention



# 代码
```python
import torch
import torch.nn as nn


class EncoderLayer(nn.Module):
	"""Encoder的一层。"""

    def __init__(self, model_dim=512, num_heads=8, ffn_dim=2018, dropout=0.0):
        super(EncoderLayer, self).__init__()

        self.attention = MultiHeadAttention(model_dim, num_heads, dropout)
        self.feed_forward = PositionalWiseFeedForward(model_dim, ffn_dim, dropout)

    def forward(self, inputs, attn_mask=None):

        # self attention
        context, attention = self.attention(inputs, inputs, inputs, padding_mask)

        # feed forward network
        output = self.feed_forward(context)

        return output, attention


class Encoder(nn.Module):
	"""多层EncoderLayer组成Encoder。"""

    def __init__(self,
               vocab_size,
               max_seq_len,
               num_layers=6,
               model_dim=512,
               num_heads=8,
               ffn_dim=2048,
               dropout=0.0):
        super(Encoder, self).__init__()

        self.encoder_layers = nn.ModuleList(
          [EncoderLayer(model_dim, num_heads, ffn_dim, dropout) for _ in
           range(num_layers)])

        self.seq_embedding = nn.Embedding(vocab_size + 1, model_dim, padding_idx=0)
        self.pos_embedding = PositionalEncoding(model_dim, max_seq_len)

    def forward(self, inputs, inputs_len):
        output = self.seq_embedding(inputs)
        output += self.pos_embedding(inputs_len)

        self_attention_mask = padding_mask(inputs, inputs)

        attentions = []
        for encoder in self.encoder_layers:
            output, attention = encoder(output, self_attention_mask)
            attentions.append(attention)

        return output, attentions
```

```python
import torch
import torch.nn as nn


class DecoderLayer(nn.Module):

    def __init__(self, model_dim, num_heads=8, ffn_dim=2048, dropout=0.0):
        super(DecoderLayer, self).__init__()

        self.attention = MultiHeadAttention(model_dim, num_heads, dropout)
        self.feed_forward = PositionalWiseFeedForward(model_dim, ffn_dim, dropout)

    def forward(self,
              dec_inputs,
              enc_outputs,
              self_attn_mask=None,
              context_attn_mask=None):
        # self attention, all inputs are decoder inputs
        dec_output, self_attention = self.attention(
          dec_inputs, dec_inputs, dec_inputs, self_attn_mask)

        # context attention
        # query is decoder's outputs, key and value are encoder's inputs
        dec_output, context_attention = self.attention(
          enc_outputs, enc_outputs, dec_output, context_attn_mask)

        # decoder's output, or context
        dec_output = self.feed_forward(dec_output)

        return dec_output, self_attention, context_attention


class Decoder(nn.Module):

    def __init__(self,
               vocab_size,
               max_seq_len,
               num_layers=6,
               model_dim=512,
               num_heads=8,
               ffn_dim=2048,
               dropout=0.0):
        super(Decoder, self).__init__()

        self.num_layers = num_layers

        self.decoder_layers = nn.ModuleList(
          [DecoderLayer(model_dim, num_heads, ffn_dim, dropout) for _ in
           range(num_layers)])

        self.seq_embedding = nn.Embedding(vocab_size + 1, model_dim, padding_idx=0)
        self.pos_embedding = PositionalEncoding(model_dim, max_seq_len)

    def forward(self, inputs, inputs_len, enc_output, context_attn_mask=None):
        output = self.seq_embedding(inputs)
        output += self.pos_embedding(inputs_len)

        self_attention_padding_mask = padding_mask(inputs, inputs)
        seq_mask = sequence_mask(inputs)
        self_attn_mask = torch.gt((self_attention_padding_mask + seq_mask), 0)

        self_attentions = []
        context_attentions = []
        for decoder in self.decoder_layers:
            output, self_attn, context_attn = decoder(
            output, enc_output, self_attn_mask, context_attn_mask)
            self_attentions.append(self_attn)
            context_attentions.append(context_attn)

        return output, self_attentions, context_attentions
```










