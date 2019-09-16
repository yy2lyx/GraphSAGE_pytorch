# GraphSAGE_pytorch详解

## 1. 数据源

把reddit数据集放到data文件夹下。

reddit下载地址： 
[reddit_adj.npz](https://drive.google.com/open?id=174vb0Ws7Vxk_QTUtxqTgDHSQ4El4qDHt), 
[reddit.npz](https://drive.google.com/open?id=19SphVl_Oe8SJ1r87Hr5a6znx3nJu1F2J) 

## 2.代码

运行：reddit_supervised.py

数据：data文件夹下

GraphSAGE模型的邻接矩阵采集器：model/neibor_sampler.py

GraphSAGE模型的输入及其模型定义：model/GraphSAGE

## 3. 代码来源

https://github.com/GaoYangIIE/pytorch-graphsage

自己在这个代码的基础上增加了自己对代码的批注和理解

