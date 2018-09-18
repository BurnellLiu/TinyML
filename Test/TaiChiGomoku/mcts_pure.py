#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
单纯蒙特卡洛树搜索实现

@author: CoderJie
@email: coderjie@outlook.com
"""

import copy
import numpy as np

from gomoku_game import GomokuPlayer
from gomoku_game import GomokuBoard
from operator import itemgetter


def rollout_policy_value_(state):

    avg_probs = np.ones(len(state.avl_actions))/len(state.avl_actions)
    action_probs = list(zip(state.avl_actions, avg_probs))
    state_value = 0.0

    current_player = state.current_player
    for i in range(state.width * state.height):
        rand_probs = np.random.rand(len(state.avl_actions))
        rand_action_probs = zip(state.avl_actions, rand_probs)
        max_action = max(rand_action_probs, key=itemgetter(1))[0]
        state.move(max_action)
        end, winner = state.check_winner()
        if end:
            if winner == current_player:
                state_value = 1.0
            elif winner == GomokuPlayer.Nobody:
                state_value = 0.0
            else:
                state_value = -1.0
            break

    return action_probs, state_value


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
        self.children = {}              # 子节点表, 键: 动作 值: 节点

        self.__c_uct = c_uct            # 上限置信区间参数

        self.__action_prob = prob       # 动作先验概率
        self.__action_q = 0.0           # 动作值, 代表当前状态下, 接下来下子的玩家的局面状况
        self.action_n = 0               # 动作访问次数

    def expand(self, actions_prob):
        """
        扩展
        :param actions_prob: 动作先验概率列表
        """
        for action, prob in actions_prob:
            if action not in self.children:
                self.children[action] = MCTSNode(self, prob)

    def select(self):
        """
        选择一个动作
        :return: 返回(动作, 节点)元组
        """
        return max(self.children.items(),
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
        return self.children == {}

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
        sum_value = self.__action_q * self.action_n + leaf_value
        # 访问次数+1
        self.action_n += 1
        # 更新动作值
        self.__action_q = sum_value/self.action_n

    def __get_value(self):
        """
        获取调整后的动作值
        :return:
        """
        u = self.__c_uct * self.__action_prob / (1 + self.action_n)
        return self.__action_q + u

    def __str__(self):
        return "MCTSNode: Prob: {}, Q: {}, N: {}".format(
            self.__action_prob, self.__action_q, self.action_n)


class MCTSPure:
    def __init__(self, policy_value_fn=rollout_policy_value_, play_out_n=10000):
        # 根节点
        self.__root = MCTSNode(None, 1.0)
        # 策略值函数
        self.__policy_value = policy_value_fn
        self.__play_out_n = play_out_n

    def get_action_probs(self, state):
        for i in range(self.__play_out_n):
            state_copy = copy.deepcopy(state)
            self.__play_out(state_copy)
        act_visits = [(act, node.action_n)for act, node in self.__root.children.items()]
        acts, visits = zip(*act_visits)
        probs = visits/np.sum(visits)

        self.__root = MCTSNode(None, 1.0)

        return acts, probs

    def __play_out(self, state):
        current_node = self.__root
        while True:
            if current_node.is_leaf():
                break
            action, current_node = current_node.select()
            state.move(action)

        end, winner = state.check_winner()

        if not end:
            probs, state_value = self.__policy_value(state)
            current_node.expand(probs)
            current_node.backup(state_value)
        else:
            # 游戏和棋结束
            if winner == GomokuPlayer.Nobody:
                current_node.backup(0)
            else:
                # 游戏不是和棋结束, 则代表当前状态下, 接下来下子的玩家输棋, 所以设置评估值为-1
                current_node.backup(-1)


if __name__ == '__main__':
    board = GomokuBoard()
    mcts = MCTSPure()

    while True:
        end, winner = board.check_winner()
        if end:
            break

        acts, probs = mcts.get_action_probs(board)
        move = np.random.choice(acts, p=probs)
        board.move(move)
        print(board.action_to_loc(move))

