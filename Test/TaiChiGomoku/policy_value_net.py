#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
策略值网络实现
依赖TensorFlow 1.10.0版本

@author: CoderJie
@email: coderjie@outlook.com
"""

import numpy as np
import tensorflow as tf


class PolicyValueNet:
    """
    策略值网络
    """
    def __init__(self, board_width, board_height):
        """
        构造函数
        :param board_width: 棋盘宽度
        :param board_height: 棋盘高度
        """
        self.__board_width = board_width
        self.__board_height = board_height

        # 定义模型输入占位符, 输入数据结构[?, 4, height, width]
        # None表示张量的第一维为任意长度, 第一维即代表样本数量, 就是表示模型接收任意的样本数,
        # 并且每个样本包含4个 board_height*board_width 的表格
        self.__input_data = tf.placeholder(tf.float32,
                                           shape=[None, 4, board_height, board_width])

        # 转置输入数据
        # [0, 2, 3, 1] 0 代表原始数据第一维, 1 代表原始数据第二维, 2 代表原始数据第三维, 3 代表原始数据第四维
        # 转置后数据结构变为[?, height, width, 4], 这里4通常被称为通道数为4
        self.__input_trans = tf.transpose(self.__input_data,
                                          [0, 2, 3, 1])

        # 第一层卷积层, 输入[?, height, width, 4], 输出[?, height, width, 32]
        # 卷积核数量32, 卷积核大小 3*3, 卷积核的通道数就是输入数据的通道数4, 卷积步长默认为1
        #
        self.__conv1 = tf.layers.conv2d(inputs=self.__input_trans,
                                        filters=32,
                                        kernel_size=[3, 3],
                                        padding="same",
                                        data_format="channels_last",
                                        activation=tf.nn.relu)
