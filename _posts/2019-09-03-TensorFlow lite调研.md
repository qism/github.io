---
layout:     post
title:      TensorFlow Lite调研
subtitle:
        - 简介
        - 移动端SDK比较
        - TensorFlow Lite应用  
date:       2019-09-03
author:     qism
header-img: img/post-bg-coffee.jpeg
catalog: true
tags:    
        - Tensorflow
        - Deep Learing
---

# 一、TensorFlow Lite 简介

TensorFlow Lite 是 Google I/O 2017大会上的推出的，是专门针对移动设备上可运行的深度网络模型简单版，目前还只是开发者预览版，未推出正式版。

## 1、相比TensorFlow Mobile

TensorFlow Mobile是在推出TensorFlow（简称TF）时同时推出的，用于支持手机上的模型嵌入式部署，但对于移动设备上使用深度学习网络还是有相对的限制，主要是计算性能无法达到。

1）TensorFlow Lite更为轻量级，相比TensorFlow Mobile通过 TF Lite 生成的链接库体积很小。

2）TensorFlow Mobile的编译依赖protobuf 等库，而TensorFlow Lite不需要大的依赖库。

3）TensorFlow Mobile成熟，全面，支持的用例丰富。而TensorFlow Lite目前为开发预览版本，没有涵盖所有用例。TensorFlow Lite目前主要支持CNN的实现，对其他网络中的算子还没有很好的支持，若是想把RNN迁移到移动端，目前是不OK的。针对CNN，大多数算子都有提供，但反向卷积算子并没有提供。不过缺失的算子可以自定义，并添加TF Lite 库，因此扩展性较好。

4）两者都能支持 Android, iOS，跨平台性没问题

5） TensorFlow Lite可以与Android 8.1中发布的神经网络API完美配合，即便在没有硬件加速时也能调用CPU处理，确保模型在不同设备上的运行。

6）可以用上移动端硬件加速。TF Lite 可以通过 Android Neural Networks API (NNAPI) 进行硬件加速，只要加速芯片支持 NNAPI，就能够为 TF Lite 加速。不过目前在大多数 Android 手机上，TF Lite 还是运行在 CPU 上的。 

## 2、TensorFlow Lite组件

1）TensorFlow 模型（TensorFlow Model）：保存在磁盘中的训练模型。

2）TensorFlow Lite 转化器（TensorFlow Lite Converter）：将模型转换成 TensorFlow Lite 文件格式的项目。

3）TensorFlow Lite 模型文件（TensorFlow Lite Model File）：基于 Flat Buffers，适配最大速度和最小规模的模型。

## 3、TensorFlow Lite过程

1）使用现有优秀的网络模型训练自己的模型 

模型需要现在PC端进行训练，若要使用现有模型解决新问题，也可以考虑迁移学习。

2）模型转换成TF Lite模型文件。这一步涉及模型的固化（保存模型的参数和graph），目前主流的方法有两种。

第一种：

Step 1：在算法训练的脚本中保存图模型文件（GraphDef）和变量文件（CheckPoint）。

Step 2：利用freeze_graph工具生成frozen的graphdef文件。

Step 3：利用toco工具，生成最终的TF Lite文件。

参考：https://blog.csdn.net/sinat_34022298/article/details/81569769

备注：网上大多数资料讲的都是这种通过bazel重新编译模型生成依赖方法，但看到一些开发者实践认为这种方法难度很大，建议尝试TensorFlow Lite官方的例子中的方法，也就是下面第二种方法。

第二种：变量转成常量之后写入PB文件，谷歌提供方法快速实现变量转换成常量的方法。

参考：https://www.jianshu.com/p/091415b114e2

实际开发时需要一些支持文件，已在Github找到，下载地址：libandroid_tensorflow_inference_java.jar、libtensorflow_inference.so。

3）在Android Studio进行构建

## 4、版本支持

Android Studio 3.0 

SDK Version API 25，或者API 26 

NDK Version 14

# 二、移动端深度学习SDK比较

国内的巨头百度已经发布了MDL（传送门 ）框架、腾讯发布了NCNN（传送门 ）框架，将其与谷歌的TensorFlow Lite做一下比较。

相同点：

1. 只含推理（inference）功能，使用的模型文件需要通过离线的方式训练得到。

2. 最终生成的库尺寸较小，均小于500kB。

3. 为了提升执行速度，都使用了ARM NEON指令进行加速。

4. 跨平台，iOS和Android系统都支持。

不同点：

1. MDL和NCNN均是只支持Caffe框架生成的模型文件，而TF Lite则毫无意外的只支持自家大哥TensorFlow框架生成的模型文件。
2. MDL支持利用iOS系统的Matal框架进行GPU加速，能够显著提升在iPhone上的运行速度，达到准实时的效果。而NCNN和TF Lite还没有这个功能。

# 三、TensorFlow Lite应用

TensorFlow Lite 提供了有限的预训练人工智能模型，包括Mobile Net 和 InceptionV3 物体识别计算机模型，以及 Smart Replay 自然语言处理模型。开发者用自己的数据集做的定制模型也可以在上面部署。TensorFlow Lite 使用 Android 神经网络应用程序界面（API），可以在没有加速硬件时直接调用 CPU 来处理，确保其可以兼容不同设备

Mobile Net：一类视觉模型，能够识别1000个不同的对象类别，专门为移动和嵌入式设备上的高效执行而设计。

Inception v3：图像识别模型，功能与Mobile Net类似，提供更高的准确性，但更大。

Smart Reply：一种设备上的会话模型，可以对流入的对话聊天消息进行一键式回复。第一方和第三方消息传递应用在Android Wear上使用此功能。

参考：https://arxiv.org/pdf/1610.06918v1.pdf

TensorFlow Mobile、TensorFlow Lite在移动端的解决方案并不完善（TF Mobile 的内存管理与 TF Lite 的 Operators 的缺失），在实践中可能需要更多的修正与完善。
kika输入法是应用TensorFlow Lite实现将基于循环神经网络的深度学习模型应用到安卓版的手机输入法引擎中，在克服工程化问题的情况下大大提升了输入体验：不仅使基于上下文的词预测更加准确，同时还使得词纠错功能更加强大。
可以参考这个案例中Android 移动端轻量化部署所遇到的工程化挑战极其解决方法，为后期实践做准备。

算法和实践原文参考：https://baijiahao.baidu.com/s?id=1598247649132552549&wfr=spider&for=pc

# 四、结语

1、TensorFlow Lite目前还是预览版本，相比TensorFlow mobile能够降低大约4倍的链接库体积，为移动端部署深度学习模型提供了更多的可行性。

2、只支持CNN模型，暂时不支持其他模型。对于CNN模型，没有反向卷积算子，若需要得自己添加TF Lite 库，扩展性经网友实践较好。因此，CNN在TensorFlow Lite的实践方案可行。

3、目前，在线的实践案例多是图像和语音识别，NLP部署没有太多参考，相关的解决方案也不完善，后期估计会遇到一些工程化挑战。
