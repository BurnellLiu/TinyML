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

        # 定义模型输入占位符, 张量形状为[?, 4, height, width]
        # None表示张量的第一维为任意长度, 第一维即代表样本数量, 就是表示模型接收任意的样本数,
        # 并且每个样本包含4个 board_height*board_width 的表格
        self.__input_data = tf.placeholder(tf.float32,
                                           shape=[None, 4, board_height, board_width])

        # 转置输入数据, 输出张量形状[?, height, width, 4]
        # [0, 2, 3, 1] 0 代表原始数据第一维, 1 代表原始数据第二维, 2 代表原始数据第三维, 3 代表原始数据第四维
        # 转置后可以把棋盘数据看作是通道数为4, 大小为height*width的图片
        self.__input_trans = tf.transpose(self.__input_data,
                                          [0, 2, 3, 1])

        # 第一层卷积层, 32个卷积核, 输出张量形状[?, height, width, 32]
        # 卷积核数量32, 卷积核大小 3*3, 卷积核的通道数就是输入数据的通道数4, 卷积步长默认为1
        # 填充方式为"same" 表示边界填充0
        # channels_last表示通道为最后一维
        # 默认加上偏置, 偏置初始值默认为0
        # 激活函数选择斜坡函数
        self.__conv1 = tf.layers.conv2d(inputs=self.__input_trans,
                                        filters=32,
                                        kernel_size=[3, 3],
                                        padding="same",
                                        data_format="channels_last",
                                        activation=tf.nn.relu)

        # 第二层卷积层, 64个卷积核, 输出张量形状[?, height, width, 64]
        self.__conv2 = tf.layers.conv2d(inputs=self.__conv1,
                                        filters=64,
                                        kernel_size=[3, 3],
                                        padding="same",
                                        data_format="channels_last",
                                        activation=tf.nn.relu)

        # 第三层卷积层, 128个卷积核, 输出张量形状[?, height, width, 128]
        self.__conv3 = tf.layers.conv2d(inputs=self.__conv2,
                                        filters=128,
                                        kernel_size=[3, 3],
                                        padding="same",
                                        data_format="channels_last",
                                        activation=tf.nn.relu)

        # 策略端降维, 输出张量形状[?, height, width, 4]
        self.__policy_conv = tf.layers.conv2d(inputs=self.__conv3,
                                              filters=4,
                                              kernel_size=[1, 1],
                                              padding="same",
                                              data_format="channels_last",
                                              activation=tf.nn.relu)

        # 策略端修改数据维度, 输出张量形状[?, 4 * height * width]
        self.__policy_flat = tf.reshape(self.__policy_conv,
                                        [-1, 4 * board_height * board_width])

        # 策略端全连接层, 输出张量形状[?, height * width]
        # units代表输出大小
        # 默认加上偏置, 偏置初始值默认为0
        # 使用log_softmax激活函数
        self.__policy_fc = tf.layers.dense(inputs=self.__policy_flat,
                                           units=board_height * board_width,
                                           activation=tf.nn.log_softmax)

        # 价值端降维, 输出张量形状[?, height, width, 2]
        self.__value_conv = tf.layers.conv2d(inputs=self.__conv3,
                                             filters=2,
                                             kernel_size=[1, 1],
                                             padding="same",
                                             data_format="channels_last",
                                             activation=tf.nn.relu)
        # 价值端修改数据维度, 输出张量形状[?, 2 * height * width]
        self.__value_flat = tf.reshape(self.__value_conv,
                                       [-1, 2 * board_height * board_width])

        # 价值端64个神经元的全连接层, 输出张量形状[?, 64]
        # 使用斜坡激活函数
        self.__value_fc1 = tf.layers.dense(inputs=self.__value_flat,
                                           units=64,
                                           activation=tf.nn.relu)

        # 价值端1个神经元的全连接层, 输出张量形状[?, 64]
        # 使用斜坡激活函数
        self.__value_fc1 = tf.layers.dense(inputs=self.__value_flat,
                                           units=64,
                                           activation=tf.nn.relu)