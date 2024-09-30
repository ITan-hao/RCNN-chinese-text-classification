# coding: UTF-8
import os
import torch
import numpy as np
import pickle as pkl
from tqdm import tqdm
import time
from datetime import timedelta


MAX_VOCAB_SIZE = 10000  # 词表长度限制
UNK, PAD = '<UNK>', '<PAD>'  # 定义两个特殊符号UNK和PAD，分别表示未知单词和填充符号


def build_vocab(file_path, tokenizer, max_size, min_freq):
    vocab_dic = {}#定义一个空字典vocab_dic来存储单词及其频率
    with open(file_path, 'r', encoding='UTF-8') as f:
        for line in tqdm(f):#使用tqdm迭代器打印进度条，遍历文件每一行
            lin = line.strip()
            if not lin:
                continue
            content = lin.split('\t')[0]
            for word in tokenizer(content):#对内容应用tokenizer（这里假设tokenizer返回的是单词列表）
                vocab_dic[word] = vocab_dic.get(word, 0) + 1#更新单词频率计数。
        vocab_list = sorted([_ for _ in vocab_dic.items() if _[1] >= min_freq], key=lambda x: x[1], reverse=True)[:max_size]
        #创建一个列表，仅保留频率大于等于min_freq的单词项，按频率降序排列，并取前max_size个
        vocab_dic = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
        vocab_dic.update({UNK: len(vocab_dic), PAD: len(vocab_dic) + 1})
        #将排序后的单词及其索引转换为新字典vocab_dic，并添加特殊标记（如UNK和PAD）
    return vocab_dic


def build_dataset(config, ues_word):#于构建基于Transformer模型的数据集
    if ues_word:
        tokenizer = lambda x: x.split(' ')  # 以空格隔开，word-level
    else:
        tokenizer = lambda x: [y for y in x]  # char-level
    if os.path.exists(config.vocab_path):
        vocab = pkl.load(open(config.vocab_path, 'rb'))
    else:
        """tokenizer 是一个用于文本预处理的函数，在这个上下文中，它负责将输入的文本转换成模型可以理解的形式。具体来说：
        当 ues_word 参数为 True（即词级别处理），tokenizer 函数会接收英文单词并返回一个空格分割后的单词列表。例如，给定字符串 “hello world”，tokenizer 会返回 ['hello', 'world']。
        当 ues_word 为 False（即字符级别处理），tokenizer 函数会接收中文字符并返回一个单个字符组成的列表。比如输入 “你好世界”，tokenizer 会返回 ['你', '好', '世', '界']。
        这样做的目的是为了适应不同的模型需求，如词嵌入（word embeddings）通常用于处理词级别的数据，而字符级别的处理可能适用于序列标注任务或者对字符级语义有较高敏感度的情况"""
        vocab = build_vocab(config.train_path, tokenizer=tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=1)
        pkl.dump(vocab, open(config.vocab_path, 'wb'))
        #如果vocab_path不存在，说明需要创建词汇表。
        # 调用build_vocab函数生成词汇表，指定训练路径、tokenizer、最大词汇大小和最小频率，然后保存到文件中
    print(f"Vocab size: {len(vocab)}")

    def load_dataset(path, pad_size=32):#路径参数path以及可选的pad_size参数
        contents = []
        with open(path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f):
                lin = line.strip()
                if not lin:
                    continue
                content, label = lin.split('\t')#对于非空行，分割成内容（content）和标签（label）。
                words_line = []
                token = tokenizer(content)#使用tokenizer对内容进行处理，得到token列表token
                seq_len = len(token)#计算seq_len，即tokens的数量。

                if pad_size:
                    if len(token) < pad_size:
                        token.extend([vocab.get(PAD)] * (pad_size - len(token)))
                    else:
                        token = token[:pad_size]
                        seq_len = pad_size
                #如果pad_size有效，用vocab.get(PAD)填充不足的部分；否则只保留原长度。

                # 将每个单词转换为对应的ID 使用vocab.get(word)
                for word in token:
                    words_line.append(vocab.get(word, vocab.get(UNK)))
                contents.append((words_line, int(label), seq_len))
                #对每个样本（words_line和label），以及它们的序列长度seq_len打包成元组，并添加到contents列表中。
        return contents  # [([...], 0), ([...], 1), ...]
    train = load_dataset(config.train_path, config.pad_size)
    dev = load_dataset(config.dev_path, config.pad_size)
    test = load_dataset(config.test_path, config.pad_size)
    return vocab, train, dev, test


class DatasetIterater(object):#定义了一个用于迭代训练数据集的工具
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches#batches（数据集）
        self.n_batches = len(batches) // batch_size#计数器
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0:
            self.residue = True
            #当residue变量被赋值为True时，这表示存在残余的批次，即最后一个批次可能不会恰好包含batch_size个样本，而是少于这个数量。为了处理这种情况，在__next__()1方法中，会特别处理这个残余批次，即使在迭代完成一轮后仍然返回它，直到所有数据都被遍历过。这样设计是为了充分利用数据，避免浪费。
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)

        # pad前的长度(超过pad_size的设为pad_size)
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)#seq_len 是每个样本的有效序列长度
        return (x, seq_len), y

    def __next__(self):
        """如果有剩余批次（residue为真），则选取最后一批未处理的完整批次并更新索引。
            如果已经遍历完所有正常批次，返回下一个完整的批次（如果存在剩余）。
            否则，处理标准大小的一批数据，更新索引后返回。"""
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index > self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self#返回迭代器自身，使得这个类可以用于for循环中

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config):
    iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter
#DatasetIterater 用于迭代 TensorFlow 数据集（dataset）的对象，它允许你按照配置（如批量大小和设备）从给定的数据集中逐批抽取元素。这个类可能用于执行并行计算，或者是在训练模型时分批次处理数据。


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))#通过timedelta转换为以秒为单位的整数并四舍五入


if __name__ == "__main__":
    '''提取预训练词向量'''
    # 下面的目录、文件名按需更改。
    train_dir = "./THUCNews/data/train.txt"
    vocab_dir = "./THUCNews/data/vocab.pkl"
    pretrain_dir = "./THUCNews/data/sgns.sogou.char"#字符串变量，它指向预训练词向量文件的路径
    emb_dim = 300
    filename_trimmed_dir = "./THUCNews/data/embedding_SougouNews"
    if os.path.exists(vocab_dir):
        word_to_id = pkl.load(open(vocab_dir, 'rb'))
        #判断vocab_dir是否存在，如果存在则读取词表；若不存在，则创建词表并保存。
    else:
        # tokenizer = lambda x: x.split(' ')  # 以词为单位构建词表(数据集中词之间以空格隔开)
        tokenizer = lambda x: [y for y in x]  # tokenizer函数用于分词，这里选择按字符分割。
        word_to_id = build_vocab(train_dir, tokenizer=tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=1)
        pkl.dump(word_to_id, open(vocab_dir, 'wb'))

    embeddings = np.random.rand(len(word_to_id), emb_dim)#初始化词向量矩阵embeddings为随机数组，形状为词汇表大小与嵌入维度
    f = open(pretrain_dir, "r", encoding='UTF-8')
    for i, line in enumerate(f.readlines()):
        # if i == 0:  # 若第一行是标题，则跳过
        #     continue
        lin = line.strip().split(" ")
        if lin[0] in word_to_id:
            idx = word_to_id[lin[0]]#lin[0] 应该是一个单词，而 word_to_id 是一个映射，其中每个单词（键）对应一个唯一的整数ID（值）。
            emb = [float(x) for x in lin[1:301]]
            embeddings[idx] = np.asarray(emb, dtype='float32')#将提取的词向量转换为浮点数数组，并赋值给embeddings[idx]
    f.close()
    np.savez_compressed(filename_trimmed_dir, embeddings=embeddings)
#使用np.savez_compressed函数将处理后的词向量保存为.npz压缩文件