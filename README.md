# RCNN-chinese-text-classification
run.py ：
# 中文文本分类
使用argparse库解析命令行输入并配置模型和数据加载

  - `--model`：选择模型：TextCNN, TextRNN，TextRCNN。
  - `--embedding`：指定词嵌入模型来源（预训练或随机）。
  - `--word`：基于词还是字符级别的分析。

## 主函数
- 数据集：THUCNews。
- 词嵌入模型：根据命令行参数指定。
- 模型名称：根据命令行参数指定。

### 根据配置初始化数据和模型
- 设置NumPy和PyTorch的随机种子，以确保每次运行时的随机数生成一致。
- 加载数据集并构建词汇表。
- 根据配置创建数据迭代器，以便在训练过程中分批处理数据。

### 训练模型
- 根据配置和迭代器训练模型。
- 保存训练过程和模型的摘要信息到日志目录。

详情：https://zhuanlan.zhihu.com/p/73176084
