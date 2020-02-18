## 过拟合、欠拟合及解决方案

### 过拟合

 模型的训练误差远小于它在测试数据集上的误差 。

#### 解决办法：

- 降低模型复杂度
- 正则化
- 设置一定的丢弃率
- 降低训练样本数

### 欠拟合

 模型无法得到较低的训练误差，我们将这一现象 。

#### 解决办法：

- 增加模型复杂度



## 梯度消失及梯度爆炸

当多层感知机层数较多时，梯度的计算容易出现消失或者爆炸。

### 协变量偏移

 根源在于特征分布的变化（即协变量的变化）。可以简单认为是测试时输入了没有见过的样本。

### 标签偏移

可以简单认为是预测结果出现了不同的标签。

### 概念偏移

标签本身的定义发生变化。



## 循环神经网络进阶ModernRNN

RNN存在的问题：梯度较容易出现衰减或爆炸（BPTT）
⻔控循环神经⽹络：捕捉时间序列中时间步距离较⼤的依赖关系 

LSTM和GRU主要是用来解决循环神经网络中梯度消失和梯度爆炸问题提出来，并且还具有保留长的历史信息的功能。它们都是基于门控的RNN，而门控可以简单的理解为对本来没有门控的输入每个元素乘上了一个0-1的权重，进行有选择性的忘记和记忆，这样就可以在有限的记忆容量(我们可以认为参数的最大容量)下记住更加重要的信息，而忘记不重要的信息，虽然GRU没有和LSTM一样的遗忘门和输入门，但是它的重置门和更新门也可以起到选择性的忘记与记忆的功能。  

### GRU

重置⻔有助于捕捉时间序列⾥短期的依赖关系； 

更新⻔有助于捕捉时间序列⾥⻓期的依赖关系。

### LSTM

 **长短期记忆long short-term memory** :
遗忘门:控制上一时间步的记忆细胞 输入门:控制当前时间步的输入
输出门:控制从记忆细胞到隐藏状态
记忆细胞：⼀种特殊的隐藏状态的信息的流动 

### 深度循环神经网络

### 双向神经网络

 双向循环神经网络前向和后向RNN连结的方式是  前向的H_t和后向的H_t用`concat`进行连结 



## 机器翻译及相关技术

 机器翻译（MT）：将一段文本从一种语言自动翻译为另一种语言，用神经网络解决这个问题通常称为神经机器翻译（NMT）。 主要特征：输出是单词序列而不是单个单词。 输出序列的长度可能与源序列的长度不同。 

### 数据预处理

 将数据集清洗、转化为神经网络的输入minbatch 

### Encoder-Decoder

encoder：输入到隐藏状态
decoder：隐藏状态到输出 

可以应用在对话系统、生成式任务中。 

Sequence to Sequence模型

## 注意力机制与Seq2seq模型

在“编码器—解码器（seq2seq）”⼀节⾥，解码器在各个时间步依赖相同的背景变量（context vector）来获取输⼊序列信息。当编码器为循环神经⽹络时，背景变量来⾃它最终时间步的隐藏状态。将源序列输入信息以循环单位状态编码，然后将其传递给解码器以生成目标序列。然而这种结构存在着问题，尤其是RNN机制实际中存在长程梯度消失的问题，对于较长的句子，我们很难寄希望于将输入的序列转化为定长的向量而保存所有的有效信息，所以随着所需翻译句子的长度的增加，这种结构的效果会显著下降。

与此同时，解码的目标词语可能只与原输入的部分词语有关，而并不是与所有的输入有关。例如，当把“Hello world”翻译成“Bonjour le monde”时，“Hello”映射成“Bonjour”，“world”映射成“monde”。在seq2seq模型中，解码器只能隐式地从编码器的最终状态中选择相应的信息。然而，注意力机制可以将这种选择过程显式地建模。

### 注意力机制框架

 Attention 是一种通用的带权池化方法，输入由两部分构成：询问（query）和键值对（key-value pairs）。attention layer得到输出与value的维度一致。对于一个query来说，attention layer 会与每一个key计算注意力分数并进行权重的归一化，输出的向量o则是value的加权求和，而每个key计算的权重与value一一对应 。

- 注意力层显式地选择相关的信息。
- 注意层的内存由键-值对组成，因此它的输出接近于键类似于查询的值。

### 引入注意力机制的Seq2Seq模型

 将注意机制添加到sequence to sequence 模型中，以显式地使用权重聚合states。下图展示encoding 和decoding的模型结构，在时间步为t的时候。此刻attention layer保存着encodering看到的所有信息——即encoding的每一步输出。在decoding阶段，解码器的t时刻的隐藏状态被当作query，encoder的每个时间步的hidden states作为key和value进行attention聚合. Attetion model的输出当作成上下文信息context vector，并与解码器输入Dt拼接起来一起送到解码器： 

```python
class Seq2SeqAttentionDecoder(d2l.Decoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqAttentionDecoder, self).__init__(**kwargs)
        self.attention_cell = MLPAttention(num_hiddens,num_hiddens, dropout)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.LSTM(embed_size+ num_hiddens,num_hiddens, num_layers, dropout=dropout)
        self.dense = nn.Linear(num_hiddens,vocab_size)

    def init_state(self, enc_outputs, enc_valid_len, *args):
        outputs, hidden_state = enc_outputs
#         print("first:",outputs.size(),hidden_state[0].size(),hidden_state[1].size())
        # Transpose outputs to (batch_size, seq_len, hidden_size)
        return (outputs.permute(1,0,-1), hidden_state, enc_valid_len)
        #outputs.swapaxes(0, 1)
        
    def forward(self, X, state):
        enc_outputs, hidden_state, enc_valid_len = state
        #("X.size",X.size())
        X = self.embedding(X).transpose(0,1)
#         print("Xembeding.size2",X.size())
        outputs = []
        for l, x in enumerate(X):
#             print(f"\n{l}-th token")
#             print("x.first.size()",x.size())
            # query shape: (batch_size, 1, hidden_size)
            # select hidden state of the last rnn layer as query
            query = hidden_state[0][-1].unsqueeze(1) # np.expand_dims(hidden_state[0][-1], axis=1)
            # context has same shape as query
#             print("query enc_outputs, enc_outputs:\n",query.size(), enc_outputs.size(), enc_outputs.size())
            context = self.attention_cell(query, enc_outputs, enc_outputs, enc_valid_len)
            # Concatenate on the feature dimension
#             print("context.size:",context.size())
            x = torch.cat((context, x.unsqueeze(1)), dim=-1)
            # Reshape x to (1, batch_size, embed_size+hidden_size)
#             print("rnn",x.size(), len(hidden_state))
            out, hidden_state = self.rnn(x.transpose(0,1), hidden_state)
            outputs.append(out)
        outputs = self.dense(torch.cat(outputs, dim=0))
        return outputs.transpose(0, 1), [enc_outputs, hidden_state,
                                        enc_valid_len]
```



## Transformer

 为了整合CNN和RNN的优势，[[Vaswani et al., 2017\]](https://d2l.ai/chapter_references/zreferences.html#vaswani-shazeer-parmar-ea-2017) 创新性地使用注意力机制设计了Transformer模型。该模型利用attention机制实现了并行化捕捉序列依赖，并且同时处理序列的每个位置的tokens，上述优势使得Transformer模型在性能优异的同时大大减少了训练时间。 

与9.7节的seq2seq模型相似，Transformer同样基于编码器-解码器架构，其区别主要在于以下三点：

1. Transformer blocks：将seq2seq模型重的循环网络替换为了Transformer Blocks，该模块包含一个多头注意力层（Multi-head Attention Layers）以及两个position-wise feed-forward networks（FFN）。对于解码器来说，另一个多头注意力层被用于接受编码器的隐藏状态。
2. Add and norm：多头注意力层和前馈网络的输出被送到两个“add and norm”层进行处理，该层包含残差结构以及层归一化。
3. Position encoding：由于自注意力层并没有区分元素的顺序，所以一个位置编码层被用于向序列元素里添加位置信息。

### 多头注意力层

 自注意力模型是一个正规的注意力模型，序列的每一个元素对应的key，value，query是完全一致的。  多头注意力层包含h个并行的自注意力层，每一个这种层被成为一个head。对每个头来说，在进行注意力计算之前，我们会将query、key和value用三个现行层进行映射，这h个注意力头的输出将会被拼接之后输入最后一个线性层进行整合。 

### 基于位置的前馈网络

 Transformer 模块另一个非常重要的部分就是基于位置的前馈网络（FFN），它接受一个形状为（batch_size，seq_length, feature_size）的三维张量。Position-wise FFN由两个全连接层组成，他们作用在最后一维上。因为序列的每个位置的状态都会被单独地更新，所以我们称他为position-wise，这等效于一个1x1的卷积。 

 与多头注意力层相似，FFN层同样只会对最后一维的大小进行改变；除此之外，对于两个完全相同的输入，FFN层的输出也将相等。 

### Add and norm

 除了上面两个模块之外，Transformer还有一个重要的相加归一化层，它可以平滑地整合输入和其他层的输出，因此我们在每个多头注意力层和FFN层后面都添加一个含残差连接的Layer Norm层。这里 Layer Norm 与7.5小节的Batch Norm很相似，唯一的区别在于Batch Norm是对于batch size这个维度进行计算均值和方差的，而Layer Norm则是对最后一维进行计算。层归一化可以防止层内的数值变化过大，从而有利于加快训练速度并且提高泛化性能。 [(ref)](https://zhuanlan.zhihu.com/p/54530247) 

### 位置编码

 与循环神经网络不同，无论是多头注意力网络还是前馈神经网络都是独立地对每个位置的元素进行更新，这种特性帮助我们实现了高效的并行，却丢失了重要的序列顺序的信息。为了更好的捕捉序列信息，Transformer模型引入了位置编码去保持输入序列元素的位置。 

## 卷积神经网络基础



## leNet



## 卷积神经网络进阶