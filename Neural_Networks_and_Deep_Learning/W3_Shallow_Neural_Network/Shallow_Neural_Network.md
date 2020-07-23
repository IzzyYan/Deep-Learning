# Shallow Neural Network浅层神经网络

## 神经网络表示

竖向堆叠起来的输入特征被称作神经网络的**输入层（the input layer）**。

神经网络的**隐藏层（a hidden layer）**。“隐藏”的含义是**在训练集中**，这些中间节点的真正数值是无法看到的。

**输出层（the output layer）**负责输出预测值。

![img](https://raw.githubusercontent.com/bighuang624/Andrew-Ng-Deep-Learning-notes/master/docs/Neural_Networks_and_Deep_Learning/single_hidden_layer_neural_network.png)

如图是一个**双层神经网络**，也称作**单隐层神经网络（a single hidden layer neural network）**。当我们计算网络的层数时，通常不考虑输入层，因此图中隐藏层是第一层，输出层是第二层。

约定俗成的符号表示是：

- 输入层的激活值为$a^{[0]}$；
- 同样，隐藏层也会产生一些激活值，记作$a^{[1]}$隐藏层的第一个单元（或者说节点）就记作$a^{[1]}_1$输出层同理。
- 另外，隐藏层和输出层都是带有参数$W$和$b$的。它们都使用上标`[1]`来表示是和第一个隐藏层有关，或者上标`[2]`来表示是和输出层有关。

## 计算神经网络的输出

![img](https://raw.githubusercontent.com/bighuang624/Andrew-Ng-Deep-Learning-notes/master/docs/Neural_Networks_and_Deep_Learning/neural_network_like_logistic.png)

实际上，神经网络只不过将 Logistic 回归的计算步骤重复很多次。对于隐藏层的第一个节点，有
$$
z _1^{[1]} = (W _1^{[1]})^TX+b _1^{[1]}
$$

$$
a _1^{[1]} = \sigma(z _1^{[1]})
$$

我们可以类推得到，对于第一个隐藏层有下列公式：
$$
z^{[1]} = (W^{[1]})^Ta^{[0]}+b^{[1]}
$$

$$
a^{[1]} = \sigma(z^{[1]})
$$

其中，$a^{[0]}$可以是一个列向量，也可以将多个列向量堆叠起来得到矩阵。如果是后者的话，得到的 z[1]z[1]和 a[1]a[1]也是一个矩阵。

同理，对于输出层有：
$$
z^{[2]} = (W^{[2]})^Ta^{[1]}+b^{[2]}
$$

$$
\hat{y} = a^{[2]} = \sigma(z^{[2]})
$$

值得注意的是层与层之间参数矩阵的规格大小。

- 输入层和隐藏层之间：${(W^{[1]})}^T$的shape 为`(4,3)`，前面的4是隐藏层神经元的个数，后面的3是输入层神经元的个数；$b^{[1]}$的shape 为`(4,1)`，和隐藏层的神经元个数相同。
- 隐藏层和输出层之间：${(W^{[2]})}^T$的shape 为`(1,4)`，前面的 1 是输出层神经元的个数，后面的 4 是隐藏层神经元的个数；$b^{[2]}$的 shape 为`(1,1)`，和输出层的神经元个数相同。

## 激活函数

有一个问题是神经网络的隐藏层和输出单元用什么激活函数。之前我们都是选用sigmoid函数，但有时其他函数的效果会好得多。

可供选用的激活函数有：

* tanh 函数（the hyperbolic tangent function，双曲正切函数）：
  $$
  a = \frac{e^z - e^{-z}}{e^z + e^{-z}}
  $$

效果几乎总比 sigmoid 函数好（除开**二元分类的输出层**，因为我们希望输出的结果介于 0 到 1 之间），因为函数输出介于-1和1之间，激活函数的平均值就更接近0，有类似数据中心化的效果。

然而，tanh函数存在和sigmoid函数一样的缺点：当z趋近无穷大（或无穷小），导数的梯度（即函数的斜率）就趋紧于0，这使得梯度算法的速度大大减缓。

* **ReLU函数（the rectified linear unit，修正线性单元）**：
  $$
  a=max(0,z)
  $$

当 z > 0 时，梯度始终为1，从而提高神经网络基于梯度算法的运算速度，收敛速度远大于sigmoid和tanh。然而当 z < 0 时，梯度一直为0，但是实际的运用中，该缺陷的影响不是很大。

* Leaky ReLU（带泄漏的 ReLU）：
  $$
  a=max(0.01z,z)
  $$

Leaky ReLU保证在z < 0的时候，梯度仍然不为0。理论上来说，Leaky ReLU 有ReLU的所有优点，但在实际操作中没有证明总是好于 ReLU，因此不常用。

![img](https://raw.githubusercontent.com/bighuang624/Andrew-Ng-Deep-Learning-notes/master/docs/Neural_Networks_and_Deep_Learning/The_activation_function.png)

在选择激活函数的时候，如果在不知道该选什么的时候就选择 ReLU，当然也没有固定答案，要依据实际问题在交叉验证集合中进行验证分析。当然，我们可以在不同层选用不同的激活函数。

## 使用非线性激活函数的原因

