import torch
from torch.utils.data import DataLoader
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
# 数据封装为 batch


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
        optimizer=None,  # optimizer = None 则表示当前为 test
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


def process_datasets_by_Tokenizer(train_datasets, test_datasets, cutlen=256):
    tokenizer = Tokenizer()

    train_datasets_texts = []
    train_datasets_labels = []
    test_datasets_taxts = []
    test_datasets_labels = []
    for item in train_datasets:
        train_datasets_labels.append(item[0] - 1)
        train_datasets_texts.append(item[1])
    for item in test_datasets:
        test_datasets_labels.append(item[0] - 1)
        test_datasets_taxts.append(item[1])

    all_datasets_texts = train_datasets_texts + test_datasets_taxts
    all_datasets_labels = train_datasets_labels + test_datasets_labels

    tokenizer.fit_on_texts(all_datasets_texts)
    train_datasets_seqs = tokenizer.texts_to_sequences(train_datasets_texts)
    test_datasets_seqs = tokenizer.texts_to_sequences(test_datasets_taxts)

    train_datasets_seqs = sequence.pad_sequences(train_datasets_seqs, cutlen)
    test_datasets_seqs = sequence.pad_sequences(test_datasets_seqs, cutlen)

    train_datasets = list(zip(train_datasets_seqs, train_datasets_labels))
    test_datasets = list(zip(test_datasets_seqs, test_datasets_labels))

    vocab_size = len(tokenizer.index_word.keys())
    num_class = len(set(all_datasets_labels))

    return train_datasets, test_datasets, vocab_size, num_class


# ---------------------------------- TEST


def generate_batch_test():
    batch = [(torch.tensor([3, 23, 2, 8]), 1), (torch.tensor([3, 45, 21,
                                                              6]), 0)]
    res = generate_batch(batch)
    print(res)


def main():
    generate_batch_test()


if __name__ == "__main__":
    main()
