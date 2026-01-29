# Daily-DeepLearning

![GitHub Stars](https://img.shields.io/github/stars/zkywanhing/Daily-DeepLearning)
![GitHub Forks](https://img.shields.io/github/forks/zkywanhing/Daily-DeepLearning)
![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)

欢迎来到 **Daily-DeepLearning**，涵盖[计算机基础课程](07-BaseClass/)、[Python快速入门](01-Python/)、[数据科学包的使用](05-Machine-Learning-Code/数据分析工具/)、[机器学习](02-Machine-Learning/)、[深度学习](03-Deep-Learning/)、[自然语言处理](04-NLP/)、[LLM](08-LLM/)等。

## Quick Start

### Prerequisites / 前置要求
- Python 3.8+
- Jupyter Notebook

### Installation / 安装
```bash
git clone https://github.com/zkywanhing/Daily-DeepLearning.git
cd Daily-DeepLearning
```

### Learning Path / 学习路径
1. Start with [Python Basics](01-Python/)
2. Learn [Data Science Tools](05-Machine-Learning-Code/数据分析工具/)
3. Study [Machine Learning Theory](02-Machine-Learning/)
4. Dive into [Deep Learning](03-Deep-Learning/)
5. Explore [Large Language Models](08-LLM/)

## Project Status / 项目状态

🟢 **Active Development / 活跃开发**

This project is actively maintained and updated. Phase 1 focuses on content audit and basic fixes. See [CONTRIBUTING.md](CONTRIBUTING.md) for how to contribute.

本项目正在积极维护和更新中。第一阶段专注于内容审核和基础修复。查看 [CONTRIBUTING.md](CONTRIBUTING.md) 了解如何参与贡献。

## 2017年：Attention is All you need

**Background**

在《Attention is all you need》之前，因为李飞飞教授推动的ImageNet数据集、GPU算力的提升，像CNN刚刚开始流行起来，如用TensorFlow或者Theano写一个手写数字识别。后来开始有人在NLP领域，用word2vec和LSTM的组合，在很多领域里做到SOTA的效果。后来就是2017年，由Google团队提出的这篇里程碑式的论文。

[核心解析](08-LLM/Attentionisallyouneed/核心解析.md) | [论文链接](08-LLM/Attentionisallyouneed/attentionisallyouneed.pdf)  | [简单例子](08-LLM/Attentionisallyouneed/example.md) | [自注意力机制](08-LLM/Attentionisallyouneed/selfattention.md) | [多头注意力](08-LLM/Attentionisallyouneed/multihead.md) | [位置编码](08-LLM/Attentionisallyouneed/positionalencoding.md) | [Harvard NLP PyTorch实现Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html) | [Transformer复现](08-LLM/Attentionisallyouneed/Transformer_code.md)

## 2018年 BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding

**Background**

BERT比较特殊的地方在于采用了双向上下文建模，通过掩码语言模型（Masked language Model），同时利用左右两侧上下文，解决传统模型中的单向性问题。还有很重要的一点，从BERT看来是，形成了“预训练+微调”的新范式，统一了多种NLP任务的框架，仅需在预训练模型基础上添加简单任务头即可适配下游任务。当时在11项NLP任务上刷新SOTA，开启了大规模预训练模型（Pre-trained Language Model, PLM）时代。

[BERT解析](08-LLM/BERT/核心解析.md) | [论文链接](https://arxiv.org/abs/1810.04805) | [BERT复现](08-LLM/BERT/BERT_code.md)

## 2018年 GPT-1：Generative Pre-trained Transformer

**Background**

在NLP任务依赖定制化模型、传统单向语言模型（如LSTM）难以建模长距离上下文的背景下，GPT-1首次将Transformer解码器架构与无监督预训练结合，提出“生成式预训练+微调”范式。通过自回归预训练（预测下一个词）学习通用文本表示，仅需简单微调即可适配分类、推理等任务，在12项NLP任务中9项达到SOTA，验证了大模型规模化训练的潜力，为后续GPT系列奠定了基础。

## 2018年 ELMo：Embeddings from Language Models

**Background**

ELMo这个工作主要还是对词向量的改进，从静态的词向量转变成动态词向量，从而提升各项NLP任务上的性能。虽然和GPT、BERT在同一年的工作，但其实应该放在这两项工作前面的，从马后炮的角度来说，主要用的还是双向LSTM，相较于Transformer这样支持并行计算的架构，再配合上MLM来捕捉双向上下文。

[ELMo解析](08-LLM/ELMo/核心解析.md) | [论文链接](https://arxiv.org/abs/1802.05365)

More....

---

### 🖥️ **计算机基础课程**  
**数据结构**  

[基本概念和算法评价](07-BaseClass/Ds/01基本概念和算法评价.md) | [线性表](07-BaseClass/Ds/02线性表.md) | [栈和队列](07-BaseClass/Ds/03栈和队列.md) | [树和二叉树](07-BaseClass/Ds/04树和二叉树.md) | [图](07-BaseClass/Ds/05图.md) | [查找](07-BaseClass/Ds/06查找.md) | [排序](07-BaseClass/Ds/07排序.md)  

**操作系统**  

[操作系统的基本概念](07-BaseClass/Os/01操作系统的基本概念.md) | [操作系统的发展和分类](07-BaseClass/Os/02操作系统的发展和分类.md) | [操作系统的运行环境](07-BaseClass/Os/03操作系统的运行环境.md) | [进程和线程](07-BaseClass/Os/04进程与线程.md) | [处理机调度](07-BaseClass/Os/05处理机调度.md) | [进程同步](07-BaseClass/Os/06进程同步.md) | [死锁](07-BaseClass/Os/07死锁.md) | [内容管理概念](07-BaseClass/Os/08内容管理概念.md) | [虚拟内存管理](07-BaseClass/Os/09虚拟内存管理.md) | [文件系统基础](07-BaseClass/Os/10文件系统基础.md)  

**计算机网络**  

[计算机网络概述](07-BaseClass/Cn/01计算机网络概述.md) | [计算机网络结构体系](07-BaseClass/Cn/02计算机网络结构体系.md) | [通信基础](07-BaseClass/Cn/03通信基础.md) | [奈氏准则和香农定理](07-BaseClass/Cn/04奈氏准则和香农定理.md) | [传输介质](07-BaseClass/Cn/05传输介质.md) | [物理层设备](07-BaseClass/Cn/06物理层设备.md) | [数据链路层的功能](07-BaseClass/Cn/07数据链路层的功能.md)  

---

### 🐍 **Python 快速入门**  
[变量、字符串、数字和运算符](01-Python/Day01.md) |  [列表、元组](01-Python/Day02.md) |  [字典、集合](01-Python/Day03.md) |  [条件语句、循环](01-Python/Day04.md) | [函数的定义与调用](01-Python/Day05.md) | [迭代、生成器、迭代器](01-Python/Day06.md) | [高阶函数、装饰器](01-Python/Day07.md) | [面向对象编程](01-Python/Day08.md) | [类的高级特性](01-Python/Day09.md) | [错误处理与调试](01-Python/Day10.md) | [文件操作](01-Python/Day11.md) | [多线程与多进程](01-Python/Day12.md) | [日期时间、集合、结构体](01-Python/Day13.md) | [协程与异步编程](01-Python/Day14.md) | [综合实践](01-Python/Day15.md)  

---

### 📊 **数据科学包的使用**  
**NumPy**  

[创建 ndarray](05-Machine-Learning-Code/数据分析工具/Numpy/创建ndarray.md) | [数据类型和运算](05-Machine-Learning-Code/数据分析工具/Numpy/数据类型和运算.md) | [索引和切片](05-Machine-Learning-Code/数据分析工具/Numpy/索引和切片.md) | [矩阵操作](05-Machine-Learning-Code/数据分析工具/Numpy/矩阵操作.md)  

**Pandas**  

[加载数据](05-Machine-Learning-Code/数据分析工具/Pandas/1_Loading.ipynb) | [行列选择](05-Machine-Learning-Code/数据分析工具/Pandas/2_Select_row_and_columns.ipynb) | [索引操作](05-Machine-Learning-Code/数据分析工具/Pandas/3_Set_reset_use_indexes.ipynb) | [数据过滤](05-Machine-Learning-Code/数据分析工具/Pandas/4_Filtering.ipynb) | [更新行列](05-Machine-Learning-Code/数据分析工具/Pandas/5_update_rows_columns.ipynb) | [数据排序](05-Machine-Learning-Code/数据分析工具/Pandas/7_sort_data.ipynb) | [数据聚合](05-Machine-Learning-Code/数据分析工具/Pandas/8_Grouping_Aggregating.ipynb) | [数据清洗](05-Machine-Learning-Code/数据分析工具/Pandas/9_Cleaning_Data.ipynb) | [时间数据处理](05-Machine-Learning-Code/数据分析工具/Pandas/10_WorkingWithDatesAndTimeSertesData.ipynb)  

**Matplotlib**  

[直线图](05-Machine-Learning-Code/数据分析工具/Matplotlib/1_creating_and_customizing_plots.ipynb) | [柱状图](05-Machine-Learning-Code/数据分析工具/Matplotlib/2_Bar_charts.ipynb) | [饼状图](05-Machine-Learning-Code/数据分析工具/Matplotlib/3_Pie.ipynb) | [堆叠图](05-Machine-Learning-Code/数据分析工具/Matplotlib/4_stack.ipynb) | [填充图](05-Machine-Learning-Code/数据分析工具/Matplotlib/5_Line_Filling_Area.ipynb) | [直方图](05-Machine-Learning-Code/数据分析工具/Matplotlib/6_histograms.ipynb) | [散点图](05-Machine-Learning-Code/数据分析工具/Matplotlib/7_Scatter.ipynb) | [时序图](05-Machine-Learning-Code/数据分析工具/Matplotlib/8_Time_Series_Data.ipynb) | [子图](05-Machine-Learning-Code/数据分析工具/Matplotlib/10_subplot.ipynb)  

---

### 🤖 **机器学习理论与实战**  
**理论**  

[逻辑回归](02-Machine-Learning/逻辑回归.md) | [EM 算法](02-Machine-Learning/EM算法.md) | [集成学习](02-Machine-Learning/集成学习入门.md) | [随机森林与 GBDT](02-Machine-Learning/随机森林和GBDT.md) | [ID3/C4.5 算法](02-Machine-Learning/ID3和C4.5算法.md) | [K-means](02-Machine-Learning/K-means.md) | [K 最近邻](02-Machine-Learning/K最近邻.md) | [贝叶斯](02-Machine-Learning/贝叶斯.md) | [XGBoost/LightGBM](02-Machine-Learning/XgBoost和LightGBM.md) | [Gradient Boosting](02-Machine-Learning/Gradient_Boosting.md) | [Boosting Tree](https://mp.weixin.qq.com/s/Cdi0CcWDLgS6Kk7Kx71Vaw) | [回归树](https://mp.weixin.qq.com/s/XiTH-8FY5Aw-p_1Ifhx4oQ) | [XGBoost](02-Machine-Learning/XgBoost.md) | [GBDT 分类](02-Machine-Learning/GBDT分类.md) | [GBDT 回归](02-Machine-Learning/GBDT回归.md) | [LightGBM](02-Machine-Learning/LightGBM.md) | [CatBoost](02-Machine-Learning/CatBoost.md)  

---

### 🏊‍♀️ **深度学习理论与实战**  
**理论**  

[Word2Vec](03-Deep-Learning/Word2Vec.md) | [BatchNorm](03-Deep-Learning/BatchNorm.md) | [Dropout](03-Deep-Learning/Dropout.md) | [CNN](03-Deep-Learning/Network/01_CNN.md) | [RNN](03-Deep-Learning/RNN.md) | [LSTM](03-Deep-Learning/LSTM.md) | [Attention](03-Deep-Learning/Attention.md) | [ELMo](03-Deep-Learning/ELMo.md) | [Transformer](03-Deep-Learning/Transformer.md) | [BERT](03-Deep-Learning/BERT.md) | [ALBERT](03-Deep-Learning/ALBERT.md) | [XLNet](03-Deep-Learning/XLNet.md)  

**实战**  

- **TensorFlow**  
  
  [Hello World](06-Deep-Learning-Code/Tensorflow/Helloworld.md) | [线性回归](06-Deep-Learning-Code/Tensorflow/linear_regression.md) | [逻辑回归](06-Deep-Learning-Code/Tensorflow/logistic_regression.md) | [基本图像分类](06-Deep-Learning-Code/Tensorflow/基本图像分类.ipynb)  
- **PyTorch**  
  
  [入门](06-Deep-Learning-Code/pytorch/gettingstart.md) | [自动求导](06-Deep-Learning-Code/pytorch/autograd.ipynb) | [神经网络](06-Deep-Learning-Code/pytorch/NeuralNetworks.ipynb)  

---

### 🀄 **NLP 相关**  
[Word2Vec](03-Deep-Learning/Word2Vec.md) | [LSTM](03-Deep-Learning/LSTM.md) | [ELMo](03-Deep-Learning/ELMo.md) | [ALBERT](03-Deep-Learning/ALBERT.md) | [XLNet](03-Deep-Learning/XLNet.md)  

---

### 📫 **联系我们**  

如果你有任何问题或建议，欢迎通过以下方式联系我们：  

- **邮箱**：[lauzanhing@gmail.com](mailto:lauzanhing@gmail.com)  
- **GitHub Issues**：[https://github.com/zkywanhing/Daily-DeepLearning/issues](https://github.com/zkywanhing/Daily-DeepLearning/issues)  

---
