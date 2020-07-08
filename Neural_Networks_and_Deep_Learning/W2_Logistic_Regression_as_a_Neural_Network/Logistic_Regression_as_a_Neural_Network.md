# 神经网络基础

- 实现一个神经网络时，如果需要遍历整个训练集，并不需要直接使用for循环。
- 神经网络的计算过程中，通常有一个正向过程（forward pass）或者叫**正向传播步骤（forward propagation step）**，接着会有一个反向过程（backward pass）或者叫**反向传播步骤（backward propagation step）**。

## Logistic Regression

Logistic 回归是一个用于二分分类的算法。

Logistic 回归中使用的参数如下：

- 输入的特征向量：$x \in R^{n_x}$，其中$n_x$是特征数量；

- 用于训练的标签：$y \in 0,1$

- 权重：$w \in R^{n_x}$ 

- 偏置： $b \in R$

- 输出：$\hat{y} = \sigma(w^Tx+b)$ 

- Sigmoid函数：
  $$
  s = \sigma(w^Tx+b) = \sigma(z) = \frac{1}{1+e^{-z}}
  $$

为将$w^Tx+b$约束在 [0, 1] 间，引入 Sigmoid 函数。从下图可看出，Sigmoid 函数的值域为 [0, 1]。

<img src="https://raw.githubusercontent.com/bighuang624/Andrew-Ng-Deep-Learning-notes/master/docs/Neural_Networks_and_Deep_Learning/sigmoid-function.png" alt="img" style="zoom:67%;" />

Logistic 回归可以看作是一个非常小的神经网络。下图是一个典型例子：

<img src="https://raw.githubusercontent.com/bighuang624/Andrew-Ng-Deep-Learning-notes/master/docs/Neural_Networks_and_Deep_Learning/LogReg_kiank.png" alt="img" style="zoom:50%;" />

## Loss Function（损失函数）

**损失函数（loss function）**用于衡量预测结果与真实值之间的误差。

最简单的损失函数定义方式为平方差损失：
$$
L(\hat{y},y) = \frac{1}{2}(\hat{y}-y)^2
$$
但 Logistic 回归中我们并不倾向于使用这样的损失函数，因为之后讨论的优化问题会变成非凸的，最后会得到很多个局部最优解，梯度下降法可能找不到全局最优值。

一般使用: 
$$
L(\hat{y},y) = -(y\log\hat{y})-(1-y)\log(1-\hat{y})
$$
损失函数是在单个训练样本中定义的，它衡量了在**单个**训练样本上的表现。而**代价函数（cost function，或者称作成本函数）**衡量的是在**全体**训练样本上的表现，即衡量参数$w$和$b$的效果。
$$
J(w,b) = \frac{1}{m}\sum_{i=1}^mL(\hat{y}^{(i)},y^{(i)})
$$

## Gradient Descent（梯度下降法）

函数的**梯度（gradient）**指出了函数的最陡增长方向。即是说，按梯度的方向走，函数增长得就越快。那么按梯度的负方向走，函数值自然就降低得最快了。

模型的训练目标即是寻找合适的$w$与$b$以最小化代价函数值。简单起见我们先假设$w$与$b$都是一维实数，那么可以得到如下的$J$关于$w$与$b$的图：

<img src="https://raw.githubusercontent.com/bighuang624/Andrew-Ng-Deep-Learning-notes/master/docs/Neural_Networks_and_Deep_Learning/cost-function.png" alt="img" style="zoom:48%;" />

可以看到，成本函数 J 是一个**凸函数**，与非凸函数的区别在于其不含有多个局部最低点；选择这样的代价函数就保证了无论我们初始化模型参数如何，都能够寻找到合适的最优解。

参数$w$的更新公式为：
$$
w := w - \alpha\frac{dJ(w, b)}{dw}
$$
其中$α$表示学习速率，即每次更新的$w$的步伐长度。

当$w$大于最优解$w′$时，导数大于 0，那么$w$就会向更小的方向更新。反之当$w$小于最优解$w′$时，导数小于 0，那么 $w$就会向更大的方向更新。迭代直到收敛。

在成本函数$ J(w, b)$中还存在参数$b$，因此也有：
$$
b := b - \alpha\frac{dJ(w, b)}{db}
$$

## Computation Graph（计算图）

神经网络中的计算即是由多个计算网络输出的前向传播与计算梯度的后向传播构成。所谓的**反向传播（Back Propagation）**即是当我们需要计算最终值相对于某个特征变量的导数时，我们需要利用计算图中上一步的结点定义。

## Logistic Regression中的梯度下降法

假设输入的特征向量维度为 2，即输入参数共有$x_1, w_1, x_2, w_2, b$这五个。可以推导出如下的计算图：

<img src="https://raw.githubusercontent.com/bighuang624/Andrew-Ng-Deep-Learning-notes/master/docs/Neural_Networks_and_Deep_Learning/Logistic-Computation-Graph.png" alt="IMG" style="zoom:33%;" />

首先反向求出$L$对于$a$的导数：
$$
da=\frac{dL(a,y)}{da}=−\frac{y}{a}+\frac{1−y}{1−a}
$$
然后继续反向求出$L$对于$z$的导数：
$$
dz=\frac{dL}{dz}=\frac{dL(a,y)}{dz}=\frac{dL}{da}\frac{da}{dz}=a−y
$$
依此类推求出最终的损失函数相较于原始参数的导数之后，根据如下公式进行参数更新：
$$
w _1:=w _1−\alpha dw _1
$$

$$
w _2:=w _2−\alpha dw _2
$$

$$
b:=b−\alpha db
$$

接下来我们需要将对于单个用例的损失函数扩展到整个训练集的代价函数：
$$
J(w,b)=\frac{1}{m}\sum^m_{i=1}L(a^{(i)},y^{(i)})
$$

$$
a^{(i)}=\hat{y}^{(i)}=\sigma(z^{(i)})=\sigma(w^Tx^{(i)}+b)
$$

我们可以对于某个权重参数$w_1$，其导数计算为：
$$
\frac{\partial J(w,b)}{\partial{w_1}}=\frac{1}{m}\sum^m_{i=1}\frac{\partial L(a^{(i)},y^{(i)})}{\partial{w_1}}
$$
完整的Logistic Regression中某次训练的流程如下，这里仅假设特征向量的维度为 2：

![img](https://raw.githubusercontent.com/bighuang624/Andrew-Ng-Deep-Learning-notes/master/docs/Neural_Networks_and_Deep_Learning/whole-logistic-training-update.png)

然后对$w_1、w_2、b$进行迭代。

*note: $dz$是目标函数对$z$求导的简便写法，$dz^{(i)}$表示第$i$个的导数，无角标表示所有导数*

[Derivation of DL/dz](https://www.coursera.org/learn/neural-networks-deep-learning/discussions/weeks/2/threads/ysF-gYfISSGBfoGHyLkhYg)

上述过程在计算时有一个缺点：你需要编写两个for循环。第一个for循环遍历$m$个样本，而第二个for循环遍历所有特征。如果有大量特征，在代码中显式使用for循环会使算法很低效。**vectorization(向量化)**可以用于解决显式使用for循环的问题。

## Vectorization(向量化)



[Week 2 Quiz - Neural Network Basics]([https://github.com/Kulbear/deep-learning-coursera/blob/master/Neural%20Networks%20and%20Deep%20Learning/Week%202%20Quiz%20-%20Neural%20Network%20Basics.md](https://github.com/Kulbear/deep-learning-coursera/blob/master/Neural Networks and Deep Learning/Week 2 Quiz - Neural Network Basics.md))

