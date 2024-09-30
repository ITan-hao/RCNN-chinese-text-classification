import time
import torch
import numpy as np
from train_eval import train, init_network
from importlib import import_module
import argparse
from tensorboardX import SummaryWriter

# 创建一个ArgumentParser对象，设置了描述信息，表示这是一个针对中文文本分类的脚本。
parser = argparse.ArgumentParser(description='Chinese Text Classification')#argparse库用于解析命令行输入并将其转换成程序所需的结构
parser.add_argument('--model', type=str, required=True, help='choose a model: TextCNN, TextRNN')
parser.add_argument('--embedding', default='pre_trained', type=str, help='random or pre_trained')#指定词嵌入模型来源（预训练或随机）
parser.add_argument('--word', default=False, type=bool, help='True for word, False for char')#基于词还是字符级别的分析
args = parser.parse_args()#运行这个解析器来读取命令行参数，将它们存储在Namespace对象args中。这些参数会被后续的代码片段用来配置模型和数据加载过程。


if __name__ == '__main__':
    dataset = 'THUCNews'  # 数据集

    # 搜狗新闻:embedding_SougouNews.npz, 腾讯:embedding_Tencent.npz, 随机初始化:random
    embedding = 'embedding_SougouNews.npz' #词嵌入模型
    if args.embedding == 'random':
        embedding = 'random'
    model_name = args.model  #TextCNN, TextRNN,
    if model_name == 'FastText':
        from utils_fasttext import build_dataset, build_iterator, get_time_dif
        embedding = 'random'
    else:
        from utils import build_dataset, build_iterator, get_time_dif

    x = import_module('models.' + model_name)
    #x.Config 是一个类，用于存储训练模型时相关的超参数，比如数据集名称、使用的词嵌入模型等
    config = x.Config(dataset, embedding)#根据传入的 dataset 和 embedding 参数来初始化一个配置对象 Config
    np.random.seed(1)#设置NumPy的随机种子，使得每次运行时的随机数生成一致
    torch.manual_seed(1)#设置PyTorch CPU上的随机种子
    torch.cuda.manual_seed_all(1)#设置所有GPU的随机种子
    torch.backends.cudnn.deterministic = True  # 开启CuDNN的确定性模式，这会确保即使在不同的硬件上，同一张GPU卡上执行卷积神经网络(CNN)时得到的结果也是相同的，这对于研究和复现实验非常有用。

    start_time = time.time()#计算后续操作（如加载数据、构建迭代器、训练模型等）所消耗的时间
    print("Loading data...")
    vocab, train_data, dev_data, test_data = build_dataset(config, args.word)#加载数据集并构建词汇表
    train_iter = build_iterator(train_data, config)#将训练数据 train_data 根据配置 config 的设置转换成可以逐批次读取的数据流，以便在训练过程中分批处理数据
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)#build_iterator 函数用于创建数据迭代器，它通常用于处理大规模文本数据并将其转换为适合模型训练的小批次
    time_dif = get_time_dif(start_time)#计算的是从开始执行到当前操作所花费的时间，这里打印出来是为了显示数据加载过程所需的时间，有助于评估性能和优化流程。
    print("Time usage:", time_dif)

    # train
    config.n_vocab = len(vocab)
    model = x.Model(config).to(config.device)
    writer = SummaryWriter(log_dir=config.log_path + '/' + time.strftime('%m-%d_%H.%M', time.localtime()))
    if model_name != 'Transformer':
        init_network(model)
    print(model.parameters)
    train(config, model, train_iter, dev_iter, test_iter,writer)
#config通常代表配置文件或类，它包含了训练模型所需的参数，如数据集信息、词嵌入模型、模型结构设置等
#train_iter, dev_iter, 和 test_iter 是数据迭代器，用于逐批次地从训练、验证和测试数据集中加载样本进行模型训练。