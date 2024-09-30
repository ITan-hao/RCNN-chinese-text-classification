# coding: UTF-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
from utils import get_time_dif
from tensorboardX import SummaryWriter

"""
该函数初始化网络参数，主要针对模型中的权重（weight）进行初始化。
有三种方法可以选择：Xavier 初始化（nn.init.xavier_normal_()）
Kaiming 初始化（nn.init.kaiming_normal_()），默认是Xavier。
对于bias部分，设置为零；对其他非weight和bias的参数不做处理
"""
# 权重初始化，默认xavier
def init_network(model, method='xavier', exclude='embedding', seed=123):
    for name, w in model.named_parameters():
        if exclude not in name:
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass


def train(config, model, train_iter, dev_iter, test_iter,writer):
    start_time = time.time()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # 学习率指数衰减，每次epoch：学习率 = gamma * 学习率
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9) ExponentialLR表示学习率会按照给定的衰减因子gamma以指数方式逐渐降低
    total_batch = 0  # 记录进行到多少batch
    dev_best_loss = float('inf')#初始化 dev_best_loss 变量为正无穷大（Infinity
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升
    #writer = SummaryWriter(log_dir=config.log_path + '/' + time.strftime('%m-%d_%H.%M', time.localtime()))
    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        # scheduler.step() # 学习率衰减
        for i, (trains, labels) in enumerate(train_iter):
            #print (trains[0].shape)
            outputs = model(trains)
            model.zero_grad()
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            if total_batch % 100 == 0:
                # 每多少轮输出在训练集和验证集上的效果
                true = labels.data.cpu()
                predic = torch.max(outputs.data, 1)[1].cpu()
                """
                torch.max(outputs.data, 1)：它计算outputs张量（假设是softmax后的预测概率分布）在第一个维度（通常表示样本）上找到最大值。
                结果是一个包含每个样本最大值及其索引的元组。
                [1]：由于我们只关心最大值对应的索引（因为这代表了模型对每个样本最可能的分类），所以选择索引部分，即取第二个元素（Python数组从0开始计数，所以索引1对应的是最大值的索引）。
                """
                train_acc = metrics.accuracy_score(true, predic)#计算训练集上预测标签（predic）与真实标签（true）之间准确性的代码片段,metrics.accuracy_score 是一个评估指标，它返回分类任务中预测正确的比例

                dev_acc, dev_loss = evaluate(config, model, dev_iter) #使用evaluate函数评估模型在开发集的表现，并保存最佳模型状态
                if dev_loss < dev_best_loss:#更新最佳损失并保存模型状态
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), config.save_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}'
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))
                writer.add_scalar("loss/train", loss.item(), total_batch)#标签名 添加到日志的数值，即当前批次的损失 累计的批次计数，用来作为时间戳或迭代次数的标识
                writer.add_scalar("loss/dev", dev_loss, total_batch)
                writer.add_scalar("acc/train", train_acc, total_batch)
                writer.add_scalar("acc/dev", dev_acc, total_batch)
                model.train()
            total_batch += 1
            if total_batch - last_improve > config.require_improvement:
                #如果当前迭代（total_batch）与上一次验证集损失（dev_loss）有所下降（last_improve）的时间间隔超过了config.require_improvement设定的数量，那么就会执行相应的操作
                # 验证集loss超过1000batch没下降，结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break
    writer.close()
    test(config, model, test_iter)


def test(config, model, test_iter):
    # test
    model.load_state_dict(torch.load(config.save_path))#加载预训练模型的状态字典，通常从保存的模型文件中读取。
    model.eval()
    start_time = time.time()
    test_acc, test_loss, test_report, test_confusion = evaluate(config, model, test_iter, test=True)#配置、模型、测试迭代器和一个标志表示这是测试阶段
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)


def evaluate(config, model, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():# 开启一个上下文管理器，关闭自动梯度追踪，因为测试不需要反向传播。
        for texts, labels in data_iter:
            outputs = model(texts)
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    acc = metrics.accuracy_score(labels_all, predict_all)#使用 sklearn 的 accuracy_score 函数
    if test:
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)#生成分类报告，包括精度、召回率和F1分数。
        confusion = metrics.confusion_matrix(labels_all, predict_all)#创建混淆矩阵。
        return acc, loss_total / len(data_iter), report, confusion#准确率、平均损失、分类报告和混淆矩阵
    return acc, loss_total / len(data_iter)

#Precision (精确度): 表示模型预测为正类的样本中有多少是真正的正类。
# 公式为 Precision = TP / (TP + FP)，其中 TP（True Positive）是真正例，FP（False Positive）是假正例。高精度意味着模型很少误判负例为正例。

# Recall (召回率): 表示实际为正类的样本中，模型能正确识别出来的比例。
# 公式为 Recall = TP / (TP + FN)，其中 FN（False Negative）是真负例未被识别出来。高召回率意味着模型对正例的识别能力强。

# F1-Score: 是 Precision 和 Recall 的调和平均数，综合了两者的表现。
# F1 值越高，说明模型在这两个方面都表现得较好。它避免了过于偏向某一指标导致的结果。

