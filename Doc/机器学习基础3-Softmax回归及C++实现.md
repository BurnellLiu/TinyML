
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

在在多分类问题中 $y$ 可以取 $k$ 个不同的值，也就是 $y\in\\{0,1,2,...,k-1 \\}$ 。因为每个类别都需要一个特征参数 $\theta$ ，所以有：

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
p(y=0|x;\Theta) \\\\
p(y=1|x;\Theta) \\\\
\vdots    \\\\
p(y=k-1|x;\Theta)
\end{bmatrix}
= \frac{1}{\sum\_{l=0}^{k-1}e^{x\theta^l}}
\begin{bmatrix}
e^{x\theta^0} \\\\
e^{x\theta^1} \\\\
\vdots    \\\\
e^{x\theta^{k-1}}
\end{bmatrix}
\\\
\end{align}

注意 $\frac{1}{\sum\_{l=0}^{k-1}e^{x\theta^l}}$ 这一项为对概率分布进行归一化，使得所有概率和为1。

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

##似然函数##
哪个 $\Theta$ 是最优的？我们需要先定义似然函数：

\begin{align}
\\\
L(\Theta) &= \prod\_{i=0}^{m-1} p(y^i \mid x^i)
\\\
\\\
L(\Theta) &= \prod\_{i=0}^{m-1} \sum\_{j=0}^{k-1}1 \\{y^i=j\\} \frac{e^{x^i\theta^j}}{\sum\_{l=0}^{k-1}e^{x^i\theta^l}}
\\\
\end{align}

上面的公式中 $1\\{ \cdot \\}$ 是示性函数，其取值规则为：

$$ 1\\{ 值为真的表达式 \\} = 1 $$ 

$$ 1\\{ 值为假的表达式 \\} = 0 $$ 

我们在似然函数中引入自然对数以方便后续的求导，则：

\begin{align}
\\\
L(\Theta) &= \log(\prod\_{i=0}^{m-1} \sum\_{j=0}^{k-1}1 \\{y^i=j\\} \frac{e^{x^i\theta^j}}{\sum\_{l=0}^{k-1}e^{x^i\theta^l}})
\\\
\\\
L(\Theta) &= \sum\_{i=0}^{m-1}\log(\sum\_{j=0}^{k-1}1 \\{y^i=j\\} \frac{e^{x^i\theta^j}}{\sum\_{l=0}^{k-1}e^{x^i\theta^l}})
\\\
\\\
L(\Theta) &= \sum\_{i=0}^{m-1}\sum\_{j=0}^{k-1}1 \\{y^i=j\\} \log(\frac{e^{x^i\theta^j}}{\sum\_{l=0}^{k-1}e^{x^i\theta^l}})
\\\
\\\
L(\Theta) &= \sum\_{i=0}^{m-1}\sum\_{j=0}^{k-1}1 \\{y^i=j\\}(\log e^{x^i\theta^j} - \log \sum\_{l=0}^{k-1}e^{x^i\theta^l})
\\\
\\\
L(\Theta) &= \sum\_{i=0}^{m-1}\sum\_{j=0}^{k-1}1 \\{y^i=j\\}(x^i\theta^j - \log \sum\_{l=0}^{k-1}e^{x^i\theta^l})
\\\
\end{align}

很明显似然函数最大值对应的 $\Theta$ 就是我们求解的目标，所以问题变为：
$$
\max\_\Theta L\_\Theta
$$

##梯度上升法##
使用梯度上升法可以帮助我们找到似然函数的最大值，参数 $\Theta\^t$的梯度为：

\begin{align}
\\\
\frac{\partial L(\Theta)}{\partial \theta^t} &= \frac{\partial}{\partial \theta^t} ( \sum\_{i=0}^{m-1}\sum\_{j=0}^{k-1}1 \\{y^i=j\\}(x^i\theta^j - \log \sum\_{l=0}^{k-1}e^{x^i\theta^l}) )
\\\
\\\
\frac{\partial L(\Theta)}{\partial \theta^t} &= \sum\_{i=0}^{m-1}\frac{\partial}{\partial \theta^t}(\sum\_{j=0}^{k-1}1 \\{y^i=j\\}(x^i\theta^j - \log \sum\_{l=0}^{k-1}e^{x^i\theta^l}) )
\\\
\\\
\frac{\partial L(\Theta)}{\partial \theta^t} &= \sum\_{i=0}^{m-1}\frac{\partial}{\partial \theta^t}(\sum\_{j=0}^{k-1}1 \\{y^i=j\\}x^i\theta^j - \sum\_{j=0}^{k-1}1 \\{y^i=j\\} \log \sum\_{l=0}^{k-1}e^{x^i\theta^l} )
\\\
\\\
\frac{\partial L(\Theta)}{\partial \theta^t} &= \sum\_{i=0}^{m-1}(1 \\{y^i=t\\}(x^i)^T -\frac{\partial}{\partial \theta^t} ( \sum\_{j=0}^{k-1}1 \\{y^i=j\\} \log \sum\_{l=0}^{k-1}e^{x^i\theta^l}) )
\\\
\\\
\frac{\partial L(\Theta)}{\partial \theta^t} &= \sum\_{i=0}^{m-1}(1 \\{y^i=t\\}(x^i)^T - \sum\_{j=0}^{k-1}1 \\{y^i=j\\} \frac{\partial}{\partial \theta^t} (\log \sum\_{l=0}^{k-1}e^{x^i\theta^l}) )
\\\
\\\
\frac{\partial L(\Theta)}{\partial \theta^t} &= \sum\_{i=0}^{m-1}(1 \\{y^i=t\\}(x^i)^T - \sum\_{j=0}^{k-1}1 \\{y^i=j\\} \frac{1}{\sum\_{l=0}^{k-1}e^{x^i\theta^l}} \frac{\partial}{\partial \theta^t} \sum\_{l=0}^{k-1}e^{x^i\theta^l} )
\\\
\\\
\frac{\partial L(\Theta)}{\partial \theta^t} &= \sum\_{i=0}^{m-1}(1 \\{y^i=t\\}(x^i)^T - \sum\_{j=0}^{k-1}1 \\{y^i=j\\} \frac{1}{\sum\_{l=0}^{k-1}e^{x^i\theta^l}} \frac{\partial}{\partial \theta^t} e^{x^i\theta^t} )
\\\
\\\
\frac{\partial L(\Theta)}{\partial \theta^t} &= \sum\_{i=0}^{m-1}(1 \\{y^i=t\\}(x^i)^T - \sum\_{j=0}^{k-1}1 \\{y^i=j\\} \frac{(x^i)^T e^{x^i\theta^t}}{\sum\_{l=0}^{k-1}e^{x^i\theta^l}})
\\\
\\\
\frac{\partial L(\Theta)}{\partial \theta^t} &= \sum\_{i=0}^{m-1}(1 \\{y^i=t\\}(x^i)^T - \frac{(x^i)^T e^{x^i\theta^t}}{\sum\_{l=0}^{k-1}e^{x^i\theta^l}})
\\\
\\\
\frac{\partial L(\Theta)}{\partial \theta^t} &= \sum\_{i=0}^{m-1}(x^i)^T(1 \\{y^i=t\\} - \frac{e^{x^i\theta^t}}{\sum\_{l=0}^{k-1}e^{x^i\theta^l}})
\\\
\\\
\frac{\partial L(\Theta)}{\partial \theta^t} &= \sum\_{i=0}^{m-1}(x^i)^T(1 \\{y^i=t\\} - p(y=t|x^i;\Theta))
\\\
\end{align}


http://m.blog.csdn.net/wangyangzhizhou/article/details/75088106