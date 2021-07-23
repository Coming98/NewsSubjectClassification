# -*- coding:utf-8 -*-

import torch.nn as nn
import torch.nn.functional as F


class TextSentiment(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class, batch_size, cutlen):
        """
        :param vocab_size: 整个语料包含的不同词汇总数
        :param embed_dim: 指定词嵌入的维度
        :param num_class: 文本分类的类别总数
        """
        # super(TextSentiment, self).__init__()
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_class = num_class
        self.batch_size = batch_size
        self.cutlen = cutlen

        # sparse=True 代表每次对该层更新时只更新部分权重
        self.embedding = nn.Embedding(vocab_size, embed_dim, sparse=True)
        self.fc = nn.Linear(embed_dim, num_class)

        self.init_weights()

    def init_weights(self, ):
        # 指定初始化权重的取值范围 - 一般小于1
        initrange = 0.5
        # Tips: 初始化为全 0 的网络十分难以训练
        # uniform - 均匀分布
        self.embedding.weight.data.uniform_(-initrange, initrange)
        # 偏置根据其含义初始化为 0 则没有太大影响
        self.fc.bias.data.zero_()

    def forward(self, text):
        """
        :param text: 文本数值映射后的结构
        :return: 与类别数尺寸相同的张量
        """
        # input: (batch * m) - m 表示句子长度 - 将 batch 横向拼接了, 多个语句拼接处理为了一个语句
        # label: (batch * 1)
        # example m=4 : input = [ 1 2 3 4 1 2 3 4 1 2 3 4 ...], [ 1 2 1 ...]
        embedded = self.embedding(text)  # ((batch * m), embed_dim)
        embedded = embedded.transpose(1, 0).unsqueeze(0)  # (1, embed_dim, (batch * m))

        # avg pool - 作用在行上，需求三维
        embedded = F.avg_pool1d(embedded, kernel_size=self.cutlen)  # 将同一个语句中，不同词的嵌入取平均

        return F.softmax(self.fc(embedded[0].transpose(1, 0)), dim=1)
