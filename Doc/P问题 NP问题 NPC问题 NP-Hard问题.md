
##概念##
- P问题(Polynomial Problem)：可以在多项式时间内解决的问题。

- NP问题(Non-Deterministic Polynomial Problem)：可以在多项式时间内验证一个解的问题。

- NPC问题(NP Complete Problem)：所有NP问题都可以在多项式时间内约化(Reducibility)到它并且它本身就是一个NP问题的问题。

- NP-Hard问题(NP Hard Problem)：所有NP问题都可以在多项式时间内约化(Reducibility)到它的问题。


它们的关系如下：

![](http://www.coderjie.com/static/img/2018/5/24174944.png)

多项式时间：我们知道时间复杂度有O(1)，O(n)，O(logn)，O(n^a)，O(a^n)，O(n!)等，我们把O(1)，O(n)，O(logn)，O(n^a)等称为多项式级的复杂度，我们把O(a^n)，O(n!)称为非多项式级的复杂度。

约化：问题A约化为问题B的含义就是，可以用问题B的解法解决A。