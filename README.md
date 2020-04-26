# Star-Image-Classification
通过使用百度开源的深度学习开发工具paddlehub配置的图像分类神经网络，来识别《青春有你2》中第一轮排名前五的五位选手。

## 人工智能框架

在这里我们将借助的是百度研发的paddlepaddle框架，并且百度还提供开发工具PaddleHub，用PaddleHub可以直接下载并，使用定义好的网络结构，大大简便了网络的设计过程。

**安装方法：** windows下通过pip安装paddlepaddle和paddlehub即可

PaddleHub的更多内容可以去他们GitHub上看看，内容相当丰富：
https://github.com/PaddlePaddle/PaddleHub
## 图片数据标注预处理
对于训练监督式的神经网络来说最重要的还是数据了，我们需要人为的给数据加标签，这样才能让网络训练的时候明白每张图实际上是谁，这对于网络最后的计算损失值和优化来说相当重要

**简单提一下收集数据**：可以通过网络爬虫的方法去每个人的百科，微博等地爬取她们的个人照，然后分别存储在各自的文件夹下，方便之后的数据标注。下方图片是我本地的图片保存方式，其中data下面五个文件分别是： '安崎'， '王承渲'，'许佳琪'，'虞书欣'， '赵小棠'。（文件目录如下）

* data
  * anqi
  * wangchengxuan
  * xujiaqi
  * zhaoxiaotang

### 将收集到的数据用代码来批量标注
神经网络训练需要将数据集分成训练集，验证集，测试集。另外由于我们使用的是PaddleHub开发工具，还需要生成一个标签集（文件目录如下）。

* dataset
  * label_list.txt
  * test_list.txt
  * train_list.txt
  * val_list.txt

最后main文件夹下面包含的文件只有两个，其中一个是数据批量标注的代码文件，另一个是神经网络训练和预测的代码文件

* main
  * create_data.py
  * main.py
