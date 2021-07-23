# -*- coding:utf-8 -*-
import os
import torch
from torchtext.datasets import AG_NEWS
from torch.utils.data.dataset import random_split
from models import TextSentiment
import time
from utils import fit, process_datasets_by_Tokenizer

DATA_DIR = "./data"
MODEL_DIR = "./model"
for dir in [DATA_DIR, MODEL_DIR]:
    if not os.path.exists(dir):
        os.mkdir(dir)

# 选取torchtext中的文本分类数据集'AG_NEWS'即新闻主题分类数据, 保存在指定目录下
# 并将数值映射后的训练和验证数据加载到内存中
print("loading data...")
train_datasets, test_datasets = AG_NEWS(root=DATA_DIR, split=('train', 'test'))

embed_dim = 128
batch_size = 64
n_epochs = 10
cutlen = 64

train_datasets, test_datasets, vocab_size, num_class = process_datasets_by_Tokenizer(
    train_datasets, test_datasets, cutlen=cutlen)

print("train: \n", train_datasets[:2])
print("test: \n", test_datasets[:2])
print("vocab_size = {}, num_class = {}".format(vocab_size, num_class))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = TextSentiment(vocab_size+1, embed_dim, num_class, batch_size,
                      cutlen).to(device)
print(model)

min_valid_loss = float("inf")
model_save_path = os.path.join(MODEL_DIR,
                               "emb({})_batch({})_epoch({})_cutlen({}).pth")

criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=4.0)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)

# train_len = int(len(train_datasets) * 0.95)
# sub_train_, sub_valid_ = random_split(
#     train_datasets, [train_len, len(train_datasets) - train_len])

# for epoch in range(n_epochs):
#     start_time = time.time()

#     train_loss, train_acc = fit(sub_train_,
#                                 batch_size,
#                                 model,
#                                 criterion,
#                                 optimizer=optimizer,
#                                 scheduler=scheduler)
#     valid_loss, valid_acc = fit(sub_valid_,
#                                 batch_size,
#                                 model,
#                                 criterion,
#                                 optimizer=None,
#                                 scheduler=None)

#     if (valid_loss < min_valid_loss):
#         min_valid_loss = valid_loss
#         torch.save(
#             model.state_dict(),
#             model_save_path.format(embed_dim, batch_size, n_epochs, cutlen))

#     secs = int(time.time() - start_time)
#     mins = secs / 60
#     secs = secs % 60

#     print("Epoch: %d" % (epoch + 1),
#           " | time in %d minutes, %d seconds" % (mins, secs))
#     print(
#         f"\tLoss: {train_loss:.4f}(train)\t|\tAcc: {train_acc * 100:.1f}%(train)"
#     )
#     print(
#         f"\tLoss: {valid_loss:.4f}(valid)\t|\tAcc: {valid_acc * 100:.1f}%(valid)"
#     )

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

# king = model.embedding[1354]
# woman = model.embedding.weight.data[1197]
# man = model.embedding.weight.data[320]
# queen = model.embedding.weight.data[4554]

# pre_queen = king - man + woman