### 问题描述

要解决的是一个医学图像的二分类问题，有`AK`和`SK`两种病症，根据一定量数据，进行训练，对图像进行预测。

**给定图片数据的格式：**

![](http://ww1.sinaimg.cn/large/e52819eagy1g3hceviwbvj20ab0eqgm5.jpg)



### 解决思路

整体上采用迁移学习来训练神经网络，使用InceptionV3结构，框架采用keras.

**具体思路：**

1. 读取图片数据，保存成`.npy`格式，方便后续加载
2. 标签采用one-hot形式，由于标签隐藏在文件夹命名中，所以需要自行添加标签，并保存到`.npy`文件中，方便后续加载
3. 将数据分为训练集、验证集、测试集
4. 使用keras建立InceptionV3基本模型，不包括顶层，使用预训练权重，在基本模型的基础上自定义几层神经网络，得到最后的模型，对模型进行训练
5. 优化模型，调整超参数，提高准确率
6. 在测试集上对模型进行评估，使用精确率、召回率
7. 对单张图片进行预测，并输出每种类别的概率



### 代码结构

![](http://ww1.sinaimg.cn/large/e52819eagy1g2sv3ux8klj20uq0fgmyf.jpg)



### 运行结果

**1. 训练结果**

![](http://ww1.sinaimg.cn/large/e52819eagy1g2svdxnpamj217v0fbjso.jpg)

**2. 评估结果**

![](http://ww1.sinaimg.cn/large/e52819eagy1g2svg4hlb4j20lq07i748.jpg)

**3. 预测结果**

![](http://ww1.sinaimg.cn/large/e52819eagy1g2svk9htyij20di07eaa8.jpg)



### 知识点总结

1. 如何加载实际数据，如何保存成npy文件，如何打乱数据，如何划分数据，如何进行交叉验证
2. 如何使用keras进行迁移学习
3. keras中数据增强、回调函数的使用，回调函数涉及：学习速率调整、保存最好模型、tensorboard可视化
4. 如何使用sklearn计算准确率，精确率，召回率，F1_score
5. 如何对单张图片进行预测，并打印分类概率
6. 如何指定特定GPU训练，如何指定使用GPU的内存情况








