# “猫眼慧识”--基于人脸图像的情绪分析

### 一、**选题背景**

人类情感交流是一种基于表情、姿态和语言等多种形式的非语言交流方式，而其中表情交流是最为重要的一种。表情变化可以传达出其内心的情绪变化，表情是人类内心世界的真实写照。上世纪70年代，美国著名心理学家保罗•艾克曼经过大量实验之后，将人类的基本表情定义为悲伤、害怕、厌恶、快乐、气愤和惊讶六种。同时，他们根据不同的面部表情类别建立了相应的表情图像数据库。随着研究的深入，中性表情也被研究学者加入基本面部表情中，组成了现今的人脸表情识别研究中的七种基础面部表情。

![img](file:///C:\Users\王瀚业\AppData\Local\Temp\ksohtml19760\wps1.jpg) 

面部表情的识别对于研究人类行为和心理活动，具有十分重要的研究意义和实际应用价值。现如今，面部表情识别主要使用计算机对人类面部表情进行分析识别，从而分析认得情绪变化，这在人机交互、社交网络分析、远程医疗以及刑侦监测等方面都具有重要意义。因此，研究如何通过计算机视觉技术对人脸进行情绪识别，已成为计算机视觉领域中的一个重要研究方向。本项目旨在研究基于图像处理的人脸情绪识别方法，通过深度学习等技术实现对人脸表情的自动化识别。



### 二、**相关研究综述**

近年来，人脸情绪识别的研究已经取得了不少进展，涉及到的技术包括传统的机器学习算法和深度学习算法。其中，深度学习算法因其强大的表达能力和适应性，在人脸情绪识别领域中表现优异。主要应用的深度学习模型包括卷积神经网络（CNN）、循环神经网络（RNN）和深度置信网络（DBN）等。

在数据集方面，由于人脸情绪识别领域的研究需要大量的标注数据，已经涌现出一系列常用的数据集，例如FER2013、CK+和Oulu-CASIA等。这些数据集提供了不同的情绪类别和典型的情感表情样本，对于情绪识别算法的训练和评估具有重要意义。

 

### **三、拟解决的问题和研究内容**

本项目的主要研究目标是基于图像处理技术实现对人脸情绪的自动化识别，其中需要解决的***\*主要问题包括：\****

1.提取人脸图像中的情感表达信息，包括面部特征、表情动态等。

2.建立合适的深度学习模型，实现对不同情绪的自动化识别。

3.针对不同数据集的特点，进行模型训练和优化，提高情绪识别准确率和稳定性。

***\*研究内容主要包括：\****

1.人脸检测与标注：采用Haar或Cascade检测算法，实现对人脸的检测和标注。目前打算主要使用已有数据集来代替该部分

2.特征提取和表情分析：通过对人脸图像进行预处理，提取面部特征，实现对不同表情的分析和识别。

3.建立深度学习模型：通过构建卷积神经网络（CNN）等深度学习模型，实现对人脸情绪的自动化识别。

4.数据集选择与处理：选择常用的数据集进行模型训练和测试，并进行数据预处理、数据增强等操作，提高模型的泛化能力和准确率。

5.模型评估与优化：对训练出的模型进行评估，采用交叉验证等方法，进行模型参数调优，提高情绪识别的准确率和稳定性。

 

### **四、可行性分析**

本项目的研究内容和目标符合当前计算机视觉领域的研究方向，且具有一定的应用价值。当前，深度学习技术已经成为人脸情绪识别领域的主要技术手段，各类模型的表现均在不断提升。虽然小组成员暂未有过AI相关开发经验。但是，我们有较好的编程能力与一定的自信，且网络上已经存在大量的开源数据集和代码实现，为本项目的实现提供了必要的支持。如GitHub上存在较多可供学习的源代码，Google旗下的用于各种感知和语言理解任务的机器学习开源软件库Tensorflow以及基于Torch库的深度学习框架PyTorch。对于模型训练所需要的算力，现存的各种云服务器也提供了支持。



### **五、计划进度安排**

本项目的预计时间为2个月，具体进度安排如下：

第1个月：熟悉深度学习算法和相关技术，学习常用的数据集和代码实现，完成人脸检测和标注等预处理操作。实现人脸情绪识别的深度学习模型，完成数据集的选择和处理，并进行模型训练和测试。(当前阶段) 

第2个月：对训练出的模型进行评估和优化，提高情绪识别的准确率和稳定性。完成实验结果分析和撰写论文，进行项目总结和报告撰写。



### 六、参考文献

**参考博客：**

1、https://cloud.tencent.com/developer/article/2085474

2、https://blog.csdn.net/HXBest/article/details/121981276

3、https://blog.csdn.net/guyuealian/article/details/129505205

------

**参考项目：**

1、https://github.com/oarriaga/face_classification

2、https://github.com/omar178/Emotion-recognition

3、https://github.com/He-Xiang-best/Facial-Expression-Recognition

4、https://github.com/amineHorseman/facial-expression-recognition-using-cnn

------

**数据集相关：**

1、FER2013数据集官网：https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data

2、Pytorch中正确设计并加载数据集方法：https://ptorch.com/news/215.html

3、pytorch加载自己的图像数据集实例：http://www.cppcns.com/jiaoben/python/324744.html

4、python中的图像处理框架进行图像的读取和基本变换：https://oldpan.me/archives/pytorch-transforms-opencv-scikit-image

------

**CNN相关：**

1、常见CNN网络结构详解：https://blog.csdn.net/u012897374/article/details/79199935?spm=1001.2014.3001.5506

2、基于CNN优化模型的开源项目地址：https://github.com/amineHorseman/facial-expression-recognition-using-cnn

3、A CNN based pytorch implementation on facial expression recognition (FER2013 and CK+), achieving 73.112% (state-of-the-art) in FER2013 and 94.64% in CK+ dataset：https://github.com/WuJie1010/Facial-Expression-Recognition.Pytorch

------

**VGG相关：**

1、VGG论文地址：https://arxiv.org/pdf/1409.1556.pdf

2、VGG模型详解及代码分析：https://blog.csdn.net/weixin_45225975/article/details/109220154#18c8c548-e4c5-24bb-e255-cd1c471af2ff

------

**ResNet相关：**

1、ResNet论文地址：https://arxiv.org/pdf/1512.03385.pdf

2、ResNet模型详解及代码分析：https://blog.csdn.net/weixin_44023658/article/details/105843701

3、Batch Normalization（BN）超详细解析：https://blog.csdn.net/weixin_44023658/article/details/105844861

------

**表情识别相关：**

1、基于卷积神经网络的面部表情识别(Pytorch实现)：https://www.cnblogs.com/HL-space/p/10888556.html

2、Fer2013 表情识别 pytorch (CNN、VGG、Resnet)：https://www.cnblogs.com/weiba180/p/12600259.html#resnet

3、OpenCV 使用 pytorch 模型 通过摄像头实时表情识别：https://www.cnblogs.com/weiba180/p/12613764.html



 

