
##多分类问题##
多分类问题在生活中很常见。例如，音乐可以被分类为民谣，古典，金属，流行等等。[上一章](http://www.coderjie.com/blog/604c0804dbeb11e7841d00163e0c0e36)介绍的二项逻辑回归可以很好的解决二分类问题，针对多分类问题二项逻辑回归就不太适用，Softmax回归可以很好的解决多分类问题。

##Softmax回归模型##
设特征数量为 $n$ 即，特征为 $x$，特征的参数为 $\theta$ ，我们定义如下：

\begin{align}
\\\
x & =
\begin{bmatrix}
1 \ x\_1 \ \cdots \ x\_n
\end{bmatrix}
\\\
\\\
\theta & =
\begin{bmatrix}
\theta\_0 \\\\
\theta\_1 \\\\
\vdots    \\\\
\theta\_n
\end{bmatrix}
\\\
\\\
x\theta & = \theta\_0 + \theta\_1x\_1 + \theta\_2x\_2 + \cdots + \theta\_nx\_n \\\\
\\\
\end{align}

在在多分类问题中 $y$ 可以取 $k$ 个不同的值，也就是 $y\in\\{1,2,...,k \\}$ 。因为每个类别都需要一个特征参数 $\theta$ ，所以有：

\begin{align}
\\\
\Theta & =
\begin{bmatrix}
\theta^{0} \ \theta^{1} \ \cdots \ \theta^{k-1}
\end{bmatrix}
\\\
\end{align}

针对多分类问题，我们的概率函数应该给出每一种分类结果的概率，所以我们的概率函数应该输出一个 $k$ 维的向量，我们定义概率函数如下：

\begin{align}
\\\
h\_\Theta(x) & = 
\begin{bmatrix}
p(y=1|x;\Theta) \\\\
p(y=2|x;\Theta) \\\\
\vdots    \\\\
p(y=k|x;\Theta)
\end{bmatrix}
= \frac{1}{\sum\_{j=0}^{k-1}e^{x\theta^j}}
\begin{bmatrix}
e^{x\theta^0} \\\\
e^{x\theta^1} \\\\
\vdots    \\\\
e^{x\theta^{k-1}}
\end{bmatrix}
\\\
\end{align}

注意 $\frac{1}{\sum\_{j=0}^{k-1}e^{x\theta^j}}$ 这一项为对概率分布进行归一化，使得所有概率和为1。

设训练样本数为 $m$，训练样本集为 $X$ ，训练输出集为 $Y$ ，如下：
\begin{align}
X & =
\begin{bmatrix}
x^{0}  \\\\
x^{1}  \\\\
\cdots \\\\
x^{m-1}
\end{bmatrix}
\\\
\\\
Y & =
\begin{bmatrix}
y^{0}       \\\\
y^{1}       \\\\
\vdots        \\\\
y^{m-1}
\end{bmatrix}
\\\
\end{align}

我们的目标是已知 $X$ 和 $Y$ 的情况下得到最优的 $\Theta$。