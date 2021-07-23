# 新闻主题分类任务

本任务来自于 `黑马NLP中间的实践项目` ，但是在实践过程中发现，课程给出的实践代码不完善，代码版本较老等问题，因此值得记录这一实践过程。

# 1 加载数据集

数据集为：`AG_NEWS`，新闻主题分类数据集

数据集在线下载地址 - [HERE](https://pan.baidu.com/s/1AjMmYy7YmshAKBG3-CfskA) - 提取码：k0hs，下载完成后放到 `DATA_DIR` 中即可绕过下载，直接使用

训练数据共 12w 条，测试数据共 0.76w 条

Tips：数据集返回结果是迭代器，使用一次后会失效哦~

```python
from torchtext.datasets import AG_NEWS
train_datasets, test_datasets = AG_NEWS(root=DATA_DIR, split=('train', 'test'))
```

一条数据集由 `label` 和 `content` 组成

```shell
("3","Wall St. Bears Claw Back Into the Black (Reuters)","Reuters - Short-sellers, Wall Street's dwindling\band of ultra-cynics, are seeing green again.")
```

# 2 处理数据集

使用 `keras.preprocessing.text.Tokenizer` 工具完成以下任务：

1. 语料的序列化：将词转为字典中的索引值
2. 对序列化后的语料进行截断或补齐：使用`keras.preprocessing.sequence`
3. 返回序列化后的训练数据集 `train_datasets`
4. 返回序列化后的测试数据集 `test_datasets`
5. 返回字典中词的总个数 `vocab_size`
6. 返回总类别数 `num_class`

```python
def process_datasets_by_Tokenizer(train_datasets, test_datasets, cutlen=256):
    tokenizer = Tokenizer()

    train_datasets_texts = []
    train_datasets_labels = []
    test_datasets_taxts = []
    test_datasets_labels = []
    for item in train_datasets:
        train_datasets_labels.append(item[0] - 1) # 注意将标签映射到 [0, 3]
        train_datasets_texts.append(item[1])
    for item in test_datasets:
        test_datasets_labels.append(item[0] - 1)
        test_datasets_taxts.append(item[1])

    all_datasets_texts = train_datasets_texts + test_datasets_taxts
    all_datasets_labels = train_datasets_labels + test_datasets_labels

    tokenizer.fit_on_texts(all_datasets_texts)
    train_datasets_seqs = tokenizer.texts_to_sequences(train_datasets_texts)
    test_datasets_seqs = tokenizer.texts_to_sequences(test_datasets_taxts)

    # 将序列化后的语料进行截断或补齐，使它们长度一致
    train_datasets_seqs = sequence.pad_sequences(train_datasets_seqs, cutlen)
    test_datasets_seqs = sequence.pad_sequences(test_datasets_seqs, cutlen)

    train_datasets = list(zip(train_datasets_seqs, train_datasets_labels))
    test_datasets = list(zip(test_datasets_seqs, test_datasets_labels))

    vocab_size = len(tokenizer.index_word.keys())
    num_class = len(set(all_datasets_labels))

    return train_datasets, test_datasets, vocab_size, num_class
""" 最后返回数据格式如下：
array([    0,     0,     0,     0,     0,     0,     0,     0,     0,
           ...
        3938,  2289,    15,  6459,     7,   209,   368,     4,     1,
         129]), 2)

vocab_size = 72002, num_class = 4
"""
```

# 3 构建模型

十分简单的模型：一层词嵌入，一层全连接，`softmax` 激活

Tips：初始化权重这一点我之前没有很好的涉及到…

```python
model = TextSentiment(vocab_size, embed_dim, num_class, batch_size, cutlen)
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
        # input: (batch * cutlen) - cutlen 表示句子长度 - 将 batch 横向拼接了, 多个语句拼接处理为了一个语句
        # label: (batch * 1)
        # example cutlen=4 : input = [ 1 2 3 4 1 2 3 4 1 2 3 4 ...], [ 1 2 1 ...]
        embedded = self.embedding(text)  # ((batch * cutlen), embed_dim)
        embedded = embedded.transpose(1, 0).unsqueeze(0)  # (1, embed_dim, (batch * cutlen))

        # avg pool - 作用在行上，需求三维
        embedded = F.avg_pool1d(embedded, kernel_size=self.cutlen)  # 将同一个语句中，不同词的嵌入取平均

        return F.softmax(self.fc(embedded[0].transpose(1, 0)), dim=1)
```

# 4 train&valid

* 保留最优模型 min_valid_loss
* 简单的 `SGD` 优化器但是装备了动态学习率调节 scheduler

```python
scheduler = StepLR(optimizer, step_size=1, gamma=0.9)
```

**更新策略**：step_size 控制着更新频率，相当于每 $step\_size$ 次进行一次 $\gamma$ 的更新，更新状态是叠加的
$$
lr^{(new)} = lr^0 \times \gamma^{epoch//step\_size}
$$

## main

```python
# 超参数
embed_dim = 128
batch_size = 64
n_epochs = 10
cutlen = 64

model = TextSentiment(vocab_size, embed_dim, num_class, batch_size, cutlen)

# 存储验证集效果最好的模型
min_valid_loss = float("inf")
model_save_path = os.path.join(MODEL_DIR, "emb({})_batch({})_epoch({})_cutlen({}).pth")

criterion = torch.nn.CrossEntropyLoss().to(device) # 损失函数
optimizer = torch.optim.SGD(model.parameters(), lr=4.0) # 优化器
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9) # 学习率动态调节

train_len = int(len(train_datasets) * 0.95)
sub_train_, sub_valid_ = random_split(
    train_datasets, [train_len, len(train_datasets) - train_len])

for epoch in range(n_epochs):
    start_time = time.time()

    train_loss, train_acc = fit(sub_train_,
                                batch_size,
                                model,
                                criterion,
                                optimizer=optimizer,
                                scheduler=scheduler)
    valid_loss, valid_acc = fit(sub_valid_,
                                batch_size,
                                model,
                                criterion,
                                optimizer=None,
                                scheduler=None)

    if(valid_loss < min_valid_loss):
        min_valid_loss = valid_loss
        torch.save(model.state_dict(), model_save_path.format(embed_dim, batch_size, n_epochs, cutlen))

    secs = int(time.time() - start_time)
    mins = secs / 60
    secs = secs % 60

    print("Epoch: %d" % (epoch + 1),
          " | time in %d minutes, %d seconds" % (mins, secs))
    print(
        f"\tLoss: {train_loss:.4f}(train)\t|\tAcc: {train_acc * 100:.1f}%(train)"
    )
    print(
        f"\tLoss: {valid_loss:.4f}(valid)\t|\tAcc: {valid_acc * 100:.1f}%(valid)"
    )
```

## fit

```python
def generate_batch(batch):
    """[summary]

    Args:
        batch ([type]): [description] 由样本张量和对应标签的元祖 组成的 batch_size 大小的列表
            [(sample1, label1), (sample2, label2), ..., (samplen, labeln)]
    :return 样本张量和标签各自的列表形式(Tensor)
    """

    text = []
    label = []
    for item in batch:
        text.extend(item[0])
        label.append(item[1])

    return torch.tensor(text), torch.tensor(label)

def fit(
        data,
        batch_size,
        model,
        criterion,
        optimizer=None,  # optimizer = None 则表示当前为 test / validation
        scheduler=None):
    """模型训练函数"""
    total_loss = 0.
    total_acc = 0.

    data = DataLoader(data,
                      batch_size=batch_size,
                      shuffle=True,
                      collate_fn=generate_batch)

    for i, (text, cls) in enumerate(data):
        if (optimizer):
            optimizer.zero_grad()
            output = model(text)
            loss = criterion(output, cls)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            total_acc += (output.argmax(1) == cls).sum().item() / len(cls)
        else:
            with torch.no_grad():
                output = model(text)
                loss = criterion(output, cls)
                total_loss += loss.item()
                total_acc += (output.argmax(1) == cls).sum().item() / len(cls)

    # 调整优化器学习率
    if (scheduler):
        scheduler.step()

    return total_loss / len(data), total_acc / len(data)
```

# 5 test

后续我自己做了简单的 `king - woman + man =? queen` 的测试，测试结果并不理想，但是新闻主题分类的准确率确实不低，可能还是任务侧重点不同~

```python
print("test...")

model.load_state_dict(
    torch.load(model_save_path.format(embed_dim, batch_size, n_epochs,
                                      cutlen)))
test_loss, test_acc = fit(test_datasets,
                          batch_size,
                          model,
                          criterion,
                          optimizer=None,
                          scheduler=None)
print(
    f"\tLoss: {test_loss:.4f}(test)\t|\tAcc: {test_acc * 100:.1f}%(test)"
)
```

