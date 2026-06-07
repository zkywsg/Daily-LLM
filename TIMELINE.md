# TIMELINE

> 自动生成自各家族节点的 frontmatter。**请勿手工编辑。**
> 重新生成：`python3 scripts/generate_timeline.py`

| 年份 | 名字 | 家族 | 关键思想 | 路径 |
|------|------|------|---------|------|
| 1998 | **LeNet-5** | `01-cnn` | 把卷积+池化+全连接这套范式第一次系统化定义出来，在手写数字识别上跑通 | [01-cnn/01-lenet.md](01-cnn/01-lenet.md) |
| 2012 | **AlexNet** | `01-cnn` | 把深 CNN + ReLU + Dropout + 双 GPU 训练打包一起拿出来，第一次把 ImageNet Top-5 错误率从 26% 砸到 15.3% | [01-cnn/02-alexnet.md](01-cnn/02-alexnet.md) |
| 2014 | **VGG** | `01-cnn` | 把网络深度做到 16/19 层、并把所有卷积统一成 3×3，证明深度本身就是性能来源 | [01-cnn/03-vgg.md](01-cnn/03-vgg.md) |
| 2014 | **GoogLeNet (Inception v1)** | `01-cnn` | 用 1×1 卷积降维 + 多尺度并行的 Inception 模块，把参数量压到 VGG 的 1/12 同时拿下 ImageNet 冠军 | [01-cnn/04-inception.md](01-cnn/04-inception.md) |
| 2015 | **ResNet** | `01-cnn` | 用 shortcut 让网络只学残差修正而不是从零重建映射，把 152 层稳定训练变成可能 | [01-cnn/05-resnet.md](01-cnn/05-resnet.md) |
