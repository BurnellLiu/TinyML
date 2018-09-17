#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
五子棋棋盘实现

@author: CoderJie
@email: coderjie@outlook.com
"""

from enum import Enum, unique


@unique
class GomokuPlayer(Enum):
    Nobody = -1  # 无子
    Black = 0    # 黑子
    White = 1    # 白子


class GomokuBoard(object):
    """
    五子棋棋盘
    """

    def __init__(self, **kwargs):
        """
        构造函数
        :param kwargs: 关键字参数
        """
        # 棋盘宽度
        self.__width = int(kwargs.get('width', 8))
        # 棋盘高度
        self.__height = int(kwargs.get('height', 8))
        # 胜利连子数
        self.__n_in_row = int(kwargs.get('n_in_row', 4))
        # 最近动作
        self.__last_action = -1
        # 黑子先手
        self.__current_player = GomokuPlayer.Black
        # 可用的动作
        self.__avl_actions = list(range(self.__width * self.__height))
        # 当前棋盘状态
        self.__state = [[GomokuPlayer.Nobody for col in range(self.__width)] for row in range(self.__height)]
        if self.__width < self.__n_in_row or self.__height < self.__n_in_row:
            raise Exception('GomokuBoard width and height can not be '
                            'less than {}'.format(self.__n_in_row))

    def move(self, action):
        """
        下子
        :param action: 动作
        :return: (游戏是否结束, 赢家)
        """
        row, col = self.__action_to_loc(action)
        self.__state[row][col] = self.__current_player
        self.__avl_actions.remove(action)
        self.__last_action = action

        # 检查游戏是否结束, 以及赢家
        game_end, winder = self.__check_winner()

        # 改变当前玩家
        if self.__current_player == GomokuPlayer.Black:
            self.__current_player = GomokuPlayer.White
        else:
            self.__current_player = GomokuPlayer.Black

        return game_end, winder

    def __check_winner(self):
        """
        检查胜利者
        :return: (游戏是否结束, 赢家)
        """
        act_row, act_col = self.__action_to_loc(self.__last_action)

        # 检查横向是否达成连子
        spot_count = 1
        for col in range(act_col-1, -1, -1):
            if self.__state[act_row][col] == self.__current_player:
                spot_count += 1
            else:
                break
        for col in range(act_col + 1, self.__width):
            if self.__state[act_row][col] == self.__current_player:
                spot_count += 1
            else:
                break
        if spot_count >= self.__n_in_row:
            return True, self.__current_player

        # 检查纵向是否达成连子
        spot_count = 1
        for row in range(act_row-1, -1, -1):
            if self.__state[row][act_col] == self.__current_player:
                spot_count += 1
            else:
                break
        for row in range(act_row + 1, self.__height):
            if self.__state[row][act_col] == self.__current_player:
                spot_count += 1
            else:
                break
        if spot_count >= self.__n_in_row:
            return True, self.__current_player

        # 检查斜线是否连成5子
        spot_count = 1
        row = act_row-1
        col = act_col-1
        while row >= 0 and col >= 0:
            if self.__state[row][col] == self.__current_player:
                spot_count += 1
            else:
                break
            row -= 1
            col -= 1
        row = act_row+1
        col = act_col+1
        while row < self.__height and col < self.__width:
            if self.__state[row][col] == self.__current_player:
                spot_count += 1
            else:
                break
            row += 1
            col += 1
        if spot_count >= self.__n_in_row:
            return True, self.__current_player

        # 检查反斜线是否连成5子
        spot_count = 1
        row = act_row+1
        col = act_col-1
        while row < self.__height and col >= 0:
            if self.__state[row][col] == self.__current_player:
                spot_count += 1
            else:
                break
            row += 1
            col -= 1
        row = act_row-1
        col = act_col+1
        while row >= 0 and col < self.__width:
            if self.__state[row][col] == self.__current_player:
                spot_count += 1
            else:
                break
            row -= 1
            col += 1
        if spot_count >= self.__n_in_row:
            return True, self.__current_player

        if len(self.__avl_actions) == 0:
            return True, GomokuPlayer.Nobody

        return False, GomokuPlayer.Nobody

    def __action_to_loc(self, action):
        """
        动作转换为位置
        3*3 棋盘如下:
        0 1 2
        3 4 5
        6 7 8
        动作5的位置是(1, 2)
        :param action: 动作值
        :return: 位置元组(row, col)
        """
        if action not in range(self.__width * self.__height):
            raise Exception('Action error: {}'.format(action))

        row = action // self.__width
        col = action % self.__width
        return row, col

    def __loc_to_action(self, loc):
        """
        位置转换为动作
        :param loc: 位置
        :return: 动作
        """
        if len(loc) != 2:
            raise Exception('Location error: {}'.format(loc))
        row = loc[0]
        col = loc[1]
        action = row * self.__width + col

        if action not in range(self.__width * self.__height):
            raise Exception('Location error: {}'.format(loc))
        return action


if __name__ == '__main__':
    board = GomokuBoard()