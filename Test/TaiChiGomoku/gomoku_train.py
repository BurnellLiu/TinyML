
import random
import numpy as np

from collections import defaultdict, deque

from policy_value_net import PolicyValueNet
from gomoku_mcts import MCTSSelfPlayer, MCTSPlayer, rollout_policy_value
from gomoku_game import GomokuPlayer, GomokuBoard


# 棋盘宽度
board_width = 6
# 棋盘高度
board_height = 6
# 连子胜利数
n_in_row = 4
# MCTS模拟次数
play_out_n = 500
# 自我对弈次数
self_game_num = 5000
check_freq = 50
# 训练次数
train_epochs = 5
# 学习速度
learn_rate = 2e-3
# 批量训练数据大小
batch_size = 500
# 训练池最大大小
buffer_max_size = 10000
# 训练数据缓冲池
data_buffer = deque(maxlen=buffer_max_size)
# 对手模拟次数
rival_play_out_n = 1000

best_win_ratio = 0.0


def collect_train_data(player):
    """
    完成一次自我对弈, 收集训练数据
    :param player:
    :return:
    """

    game_board = GomokuBoard(width=board_width, height=board_height, n_in_row=n_in_row)
    batch_states = []
    batch_probs = []
    current_players = []

    while True:
        action, acts_probs = player.get_action(game_board)
        # 保存当前状态
        batch_states.append(game_board.state())
        # 保存当前状态下进行各个动作的概率
        batch_probs.append(acts_probs)
        # 保存当前玩家
        current_players.append(game_board.current_player)

        # 执行动作
        game_board.move(action)

        # 检查游戏是否结束
        end, winner = game_board.check_winner()
        if end:
            batch_values = np.zeros(len(current_players))
            # 如果不是和局则将胜利者的状态值设置为1, 失败者的状态值设置为-1
            if winner == GomokuPlayer.Nobody:
                batch_values[np.array(current_players) == winner] = 1.0
                batch_values[np.array(current_players) != winner] = -1.0
            batch_values = np.reshape(batch_values, [-1, 1])
            return winner, list(zip(batch_states, batch_probs, batch_values))


def get_equi_data(play_data):
    """
    获取旋转和镜像数据, 增加训练数据
    :param play_data:
    :return:
    """
    extend_data = []
    for state, prob, value in play_data:
        for i in [1, 2, 3, 4]:
            # 旋转数据
            equi_state = np.array([np.rot90(s, i) for s in state])
            equi_prob = np.rot90(prob.reshape(board_height, board_width), i)
            extend_data.append((equi_state, equi_prob.flatten(), value))

            # 左右镜像
            equi_state = np.array([np.fliplr(s) for s in equi_state])
            equi_prob = np.fliplr(equi_prob)
            extend_data.append((equi_state, equi_prob.flatten(), value))

    return extend_data


def policy_evaluate(net):
    net_player = MCTSPlayer(net.policy_value, play_out_n)
    mcts_player = MCTSPlayer(rollout_policy_value, rival_play_out_n)

    net_win = 0

    # 神经网络玩家执黑棋先手
    for i in range(5):
        game_board = GomokuBoard(width=board_width, height=board_height, n_in_row=n_in_row)
        while True:
            action = net_player.get_action(game_board)
            game_board.move(action)
            end, winner = game_board.check_winner()
            if end:
                if winner == GomokuPlayer.Black:
                    net_win += 1
                break

            action = mcts_player.get_action(game_board)
            game_board.move(action)
            end, winner = game_board.check_winner()
            if end:
                break

    # MCTS玩家执黑棋先手
    for i in range(5):
        game_board = GomokuBoard(width=board_width, height=board_height, n_in_row=n_in_row)
        while True:
            action = mcts_player.get_action(game_board)
            game_board.move(action)
            end, winner = game_board.check_winner()
            if end:
                break

            action = net_player.get_action(game_board)
            game_board.move(action)
            end, winner = game_board.check_winner()
            if end:
                if winner == GomokuPlayer.Black:
                    net_win += 1
                break

    return net_win/10


def train():
    net = PolicyValueNet(board_width, board_height)

    self_player = MCTSSelfPlayer(net.policy_value, play_out_n)

    black_win_num = 0
    white_win_num = 0
    nobody_win_num = 0

    for i in range(self_game_num):

        # 收集训练数据
        print("Self Game: {}".format(i))
        winner, play_data = collect_train_data(self_player)
        play_data = get_equi_data(play_data)
        data_buffer.extend(play_data)
        if winner == GomokuPlayer.Black:
            black_win_num += 1
        elif winner == GomokuPlayer.White:
            white_win_num += 1
        else:
            nobody_win_num += 1
        print("Black: {:.2f} White: {:.2f} Nobody: {:.2f}".format(
              black_win_num/(i+1),
              white_win_num/(i+1),
              nobody_win_num/(i+1)))

        # 积累一些数据后, 进行训练
        if len(data_buffer) > batch_size:
            mini_batch = random.sample(data_buffer, batch_size)

            batch_states = [data[0] for data in mini_batch]
            batch_probs = [data[1] for data in mini_batch]
            batch_values = [data[2] for data in mini_batch]

            total_loss = 0.0
            total_entropy = 0.0
            for j in range(train_epochs):
                loss, entropy = net.train(batch_states, batch_probs, batch_values, learn_rate)
                total_loss += loss
                total_entropy += entropy
            print("Loss: {:.2f}, Entropy: {:.2f}".format(total_loss/train_epochs, total_entropy/train_epochs))

        if (i+1) % check_freq == 0:
            net.save_model(".\\CurrentModel\\GomokuAi")
            win_ratio = policy_evaluate(net)
            print("Rival({}), Net Win Ratio: {:.2f}", rival_play_out_n, win_ratio)
            if win_ratio > best_win_ratio:
                global best_win_ratio
                global rival_play_out_n
                best_win_ratio = win_ratio
                rival_play_out_n += 1000
                net.save_model(".\\BestModel\\GomokuAi")


if __name__ == '__main__':
    train()
