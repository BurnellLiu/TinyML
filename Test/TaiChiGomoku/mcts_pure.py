#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
单纯蒙特卡洛树搜索实现

@author: CoderJie
@email: coderjie@outlook.com
"""

import numpy as np


class MCTSNode(object):
    """
    蒙特卡洛树搜索节点
    """

    def __init__(self, parent, prob, c_uct=5.0):
        """
        构造函数
        :param parent: 父节点
        :param prob: 节点先验概率
        :param c_uct: 上限置信区间参数
        """
        self.__parent = parent          # 父节点
        self.__children = {}            # 子节点表

        self.__c_uct = c_uct            # 上限置信区间参数,

        self.__action_prob = prob       # 动作先验概率
        self.__action_q = 0.0           # 动作值
        self.__action_n = 0             # 动作访问次数

    def expand(self, actions_prob):
        """
        扩展
        :param actions_prob: 动作先验概率列表
        """
        for action, prob in actions_prob:
            if action not in self.__children:
                self.__children[action] = MCTSNode(self, prob)

    def select(self):
        """
        选择一个动作
        :return: 返回(动作, 节点)元组
        """
        return max(self.__children.items(),
                   key=lambda action_node: action_node[1].__get_value())

    def backup(self, leaf_value):
        """
        反向更新节点
        :param leaf_value: 叶子节点(终局)评估值
        """
        # 更新父节点, 父节点和当前节点评估值相反
        if self.__parent:
            self.__parent.backup(-leaf_value)
        self.__update(leaf_value)

    def is_leaf(self):
        """
        是否是叶子节点
        :return: 叶子节点返回True, 否正返回False
        """
        return self.__children == {}

    def is_root(self):
        """
        是否是根节点
        :return: 根节点返回True, 否正返回False
        """
        return self.__parent is None

    def __update(self, leaf_value):
        """
        更新节点, 更新动作值为模拟过程的平均值
        :param leaf_value: 叶子节点(终局)评估值
        """
        sum_value = self.__action_q * self.__action_n + leaf_value
        # 访问次数+1
        self.__action_n += 1
        # 更新动作值
        self.__action_q = sum_value/self.__action_n

    def __get_value(self):
        """
        获取调整后的动作值
        :return:
        """
        u = self.__c_uct * self.__action_prob / (1 + self.__action_n)
        return self.__action_q + u

    def __str__(self):
        return "MCTSNode: Prob: {}, Q: {}, N: {}".format(
            self.__action_prob, self.__action_q, self.__action_n)


class MCTSPure:
    def __init__(self):
        self.__root = MCTSNode(None, 1.0)

    def play_out(self, state):
        current_node = self.__root
        while True:
            if current_node.is_leaf():
                break
            action, current_node = current_node.select()
            state.move(action)


if __name__ == '__main__':

    node = MCTSNode(None, 1.0)
    node.expand([(0, 0.1), (1, 0.2), (2, 0.5), (3, 0.2)])

    print(node)
    print(node.select()[1])
