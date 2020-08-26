# Convolutional Neural Networks卷积神经网络

## 计算机视觉(Computer Vision)

**计算机视觉（Computer Vision）**的高速发展标志着新型应用产生的可能，例如自动驾驶、人脸识别、创造新的艺术风格。人们对于计算机视觉的研究也催生了很多机算机视觉与其他领域的交叉成果。一般的计算机视觉问题包括以下几类：

- 图片分类（Image Classification）；
- 目标检测（Object detection）；
- 神经风格转换（Neural Style Transfer）。

应用计算机视觉时要面临的一个挑战是数据的输入可能会非常大。例如一张 1000x1000x3 的图片，神经网络输入层的维度将高达三百万，使得网络权重 W 非常庞大。这样会造成两个后果：

1. 神经网络结构复杂，数据量相对较少，容易出现过拟合；
2. 所需内存和计算量巨大。

因此，一般的神经网络很难处理蕴含着大量数据的图像。解决这一问题的方法就是使用**卷积神经网络（Convolutional Neural Network, CNN）**。

## 卷积运算

我们之前提到过，神经网络由浅层到深层，分别可以检测出图片的边缘特征、局部特征（例如眼睛、鼻子等），到最后面的一层就可以根据前面检测的特征来识别整体面部轮廓。这些工作都是依托卷积神经网络来实现的。

**卷积运算（Convolutional Operation）**是卷积神经网络最基本的组成部分。我们以边缘检测为例，来解释卷积是怎样运算的。

### 边缘检测(Edge Detection)

图片最常做的边缘检测有两类：**垂直边缘（Vertical Edges）检测**和**水平边缘（Horizontal Edges）检测**。

![img](https://raw.githubusercontent.com/bighuang624/Andrew-Ng-Deep-Learning-notes/master/docs/Convolutional_Neural_Networks/Different-edges.png)

图片的边缘检测可以通过与相应滤波器进行卷积来实现。以垂直边缘检测为例，原始图片尺寸为 6x6，中间的矩阵被称作**滤波器（filter）**，尺寸为 3x3，卷积后得到的图片尺寸为 4x4，得到结果如下（数值表示灰度，以左上角和右下角的值为例）：

![img](https://raw.githubusercontent.com/bighuang624/Andrew-Ng-Deep-Learning-notes/master/docs/Convolutional_Neural_Networks/Vertical-Edge-Detection.jpg)

可以看到，卷积运算的求解过程是从左到右，由上到下，每次在原始图片矩阵中取与滤波器同等大小的一部分，每一部分中的值与滤波器中的值对应相乘后求和，将结果组成一个矩阵。

下图对应一个垂直边缘检测的例子：

![img](https://raw.githubusercontent.com/bighuang624/Andrew-Ng-Deep-Learning-notes/master/docs/Convolutional_Neural_Networks/Convolutional-operation-example.jpg)

如果将最右边的矩阵当作图像，那么中间一段亮一些的区域对应最左边的图像中间的垂直边缘。

这里有另一个卷积运算的动态的例子，方便理解：

![img](https://raw.githubusercontent.com/bighuang624/Andrew-Ng-Deep-Learning-notes/master/docs/Convolutional_Neural_Networks/Convolutional-operation.jpg)

图中的`*`表示卷积运算符号。在计算机中这个符号表示一般的乘法，而在不同的深度学习框架中，卷积操作的 API 定义可能不同：

- 在 Python 中，卷积用`conv_forward()`表示；
- 在 Tensorflow 中，卷积用`tf.nn.conv2d()`表示；
- 在 keras 中，卷积用`Conv2D()`表示。

### 更多边缘检测的例子

如果将灰度图左右的颜色进行翻转，再与之前的滤波器进行卷积，得到的结果也有区别。实际应用中，这反映了由明变暗和由暗变明的两种渐变方式。可以对输出图片取绝对值操作，以得到同样的结果。

![img](https://raw.githubusercontent.com/bighuang624/Andrew-Ng-Deep-Learning-notes/master/docs/Convolutional_Neural_Networks/Another-Convolutional-operation-example.jpg)

垂直边缘检测和水平边缘检测的滤波器如下所示：

![img](https://raw.githubusercontent.com/bighuang624/Andrew-Ng-Deep-Learning-notes/master/docs/Convolutional_Neural_Networks/Vertical-and-Horizontal-Filter.png)

其他常用的滤波器还有 Sobel 滤波器和 Scharr 滤波器。它们增加了中间行的权重，以提高结果的稳健性。

![img](https://raw.githubusercontent.com/bighuang624/Andrew-Ng-Deep-Learning-notes/master/docs/Convolutional_Neural_Networks/Sobel-Filter-and-Scharr-Filter.png)

滤波器中的值还可以设置为**参数**，通过模型训练来得到。这样，神经网络使用反向传播算法可以学习到一些低级特征，从而实现对图片所有边缘特征的检测，而不仅限于垂直边缘和水平边缘。

## 填充(Padding)

假设输入图片的大小为$n \times n$，而滤波器的大小为$f \times f$，则卷积后的输出图片大小为$(n-f+1) \times (n-f+1)$。

这样就有两个问题：

- 每次卷积运算后，输出图片的尺寸缩小；
- 原始图片的角落、边缘区像素点在输出中采用较少，输出图片丢失边缘位置的很多信息。

为了解决这些问题，可以在进行卷积操作前，对原始图片在边界上进行**填充（Padding）**，以增加矩阵的大小。通常将 0 作为填充值。

![img](https://raw.githubusercontent.com/bighuang624/Andrew-Ng-Deep-Learning-notes/master/docs/Convolutional_Neural_Networks/Padding.jpg)

设每个方向扩展像素点数量为$p$，则填充后原始图片的大小为$(n+2p) \times (n+2p)$，滤波器大小保持$f \times f$不变，则输出图片大小为$(n+2p-f+1) \times (n+2p-f+1)$。

因此，在进行卷积运算时，我们有两种选择：

- **Valid 卷积**：不填充，直接卷积。结果大小为$(n-f+1) \times (n-f+1)$；
- **Same 卷积**：进行填充，并使得卷积后结果大小与输入一致，这样$p = \frac{f-1}{2}$。

在计算机视觉领域，$f$通常为奇数。原因包括 Same 卷积中$p = \frac{f-1}{2}$能得到自然数结果，并且滤波器有一个便于表示其所在位置的中心点。

## 卷积步长(Strided Convolutions)

卷积过程中，有时需要通过填充来避免信息损失，有时也需要通过设置**步长（Stride）**来压缩一部分信息。

步长表示滤波器在原始图片的水平方向和垂直方向上每次移动的距离。之前，步长被默认为 1。而如果我们设置步长为 2，则卷积过程如下图所示：

![img](https://raw.githubusercontent.com/bighuang624/Andrew-Ng-Deep-Learning-notes/master/docs/Convolutional_Neural_Networks/Stride.jpg)

设步长为$s$，填充长度为$p$，输入图片大小为$n \times n$，滤波器大小为$f \times f$，则卷积后图片的尺寸为：
$$
\biggl\lfloor \frac{n+2p-f}{s}+1   \biggr\rfloor \times \biggl\lfloor \frac{n+2p-f}{s}+1 \biggr\rfloor
$$
注意公式中有一个向下取整的符号，用于处理商不为整数的情况。向下取整反映着当取原始矩阵的图示蓝框完全包括在图像内部时，才对它进行运算。

目前为止我们学习的“卷积”实际上被称为**互相关（cross-correlation）**，而非数学意义上的卷积。真正的卷积操作在做元素乘积求和之前，要将滤波器沿水平和垂直轴翻转（相当于旋转 180 度）。因为这种翻转对一般为水平或垂直对称的滤波器影响不大，按照机器学习的惯例，我们通常不进行翻转操作，在简化代码的同时使神经网络能够正常工作。

## 高维卷积(Convolution over Volume)

如果我们想要对三通道的 RGB 图片进行卷积运算，那么其对应的滤波器组也同样是三通道的。过程是将每个单通道（R，G，B）与对应的滤波器进行卷积运算求和，然后再将三个通道的和相加，将 27 个乘积的和作为输出图片的一个像素值。

![img](https://raw.githubusercontent.com/bighuang624/Andrew-Ng-Deep-Learning-notes/master/docs/Convolutional_Neural_Networks/Convolutions-on-RGB-image.png)

不同通道的滤波器可以不相同。例如只检测 R 通道的垂直边缘，G 通道和 B 通道不进行边缘检测，则 G 通道和 B 通道的滤波器全部置零。当输入有特定的高、宽和通道数时，滤波器可以有不同的高和宽，但通道数必须和输入一致。

如果想同时检测垂直和水平边缘，或者更多的边缘检测，可以增加更多的滤波器组。例如设置第一个滤波器组实现垂直边缘检测，第二个滤波器组实现水平边缘检测。设输入图片的尺寸为$n \times n \times n_c$（$n_c$为通道数），滤波器尺寸为$f \times f \times n_c$，则卷积后的输出图片尺寸为$(n-f+1) \times (n-f+1) \times n'_c$，$n'_c$为滤波器组的个数。

![img](https://raw.githubusercontent.com/bighuang624/Andrew-Ng-Deep-Learning-notes/master/docs/Convolutional_Neural_Networks/More-Filters.jpg)

## 单层卷积网络(One Layer of a Convolutional Network)

![img](https://raw.githubusercontent.com/bighuang624/Andrew-Ng-Deep-Learning-notes/master/docs/Convolutional_Neural_Networks/One-Layer-of-a-Convolutional-Network.jpg)

与之前的卷积过程相比较，卷积神经网络的单层结构多了激活函数和偏移量；而与标准神经网络：
$$
Z^{[l]} = W^{[l]}A^{[l-1]}+b
$$

$$
A^{[l]} = g^{[l]}(Z^{[l]})
$$

相比，滤波器的数值对应着权重$W^{[l]}$，卷积运算对应着$W^{[l]}$与$A^{[l-1]}$的乘积运算，所选的激活函数变为 ReLU。

对于一个3x3x3的滤波器，包括偏移量$b$在内共有 28 个参数。不论输入的图片有多大，用这一个滤波器来提取特征时，参数始终都是 28 个，固定不变。即**选定滤波器组后，参数的数目与输入图片的尺寸无关**。因此，卷积神经网络的参数相较于标准神经网络来说要少得多。这是 CNN 的优点之一。

### 符号总结

设$l$层为卷积层：

* $f^{[l]}$：**滤波器的高（或宽）**
* $p^{[l]}$：**填充长度**
* $s^{[l]}$：**步长**
* $n^{[l]}_c$：**滤波器组的数量**
* **输入维度**：$n^{[l-1]}_H \times n^{[l-1]}_W \times n^{[l-1]}_c$。其中$n^{[l-1]}_H$表示输入图片的高，$n^{[l-1]}_W$表示输入图片的宽。之前的示例中输入图片的高和宽都相同，但是实际中也可能不同，因此加上下标予以区分。
* **输出维度**：$n^{[l]}_H \times n^{[l]}_W \times n^{[l]}_c$。其中：

$$
n^{[l]}_H = \biggl\lfloor \frac{n^{[l-1]}_H+2p^{[l]}-f^{[l]}}{s^{[l]}}+1   \biggr\rfloor
$$

$$
n^{[l]}_W = \biggl\lfloor \frac{n^{[l-1]}_W+2p^{[l]}-f^{[l]}}{s^{[l]}}+1   \biggr\rfloor
$$

* **每个滤波器组的维度**：$f^{[l]} \times f^{[l]} \times n^{[l-1]}_c$。其中$n^{[l-1]}_c$为输入图片通道数（也称深度）。
* **权重维度**：$f^{[l]} \times f^{[l]} \times n^{[l-1]}_c \times n^{[l]}_c$
* **偏置维度**：$1 \times 1 \times 1 \times n^{[l]}_c$

由于深度学习的相关文献并未对卷积标示法达成一致，因此不同的资料关于高度、宽度和通道数的顺序可能不同。有些作者会将通道数放在首位，需要根据标示自行分辨。

## 简单卷积网络示例

一个简单的 CNN 模型如下图所示：