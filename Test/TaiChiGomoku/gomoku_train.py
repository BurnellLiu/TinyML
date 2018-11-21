
import numpy as np

from policy_value_net import PolicyValueNet
from gomoku_mcts import MCTSSelfPlayer
from gomoku_game import GomokuPlayer, GomokuBoard

# 棋盘宽度
board_width = 6
# 棋盘高度
board_height = 6
# 连子胜利数
n_in_row = 4
# MCTS模拟次数
play_out_n = 400


def collect_train_data(player):

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


if __name__ == '__main__':
    net = PolicyValueNet(board_width, board_height)
    self_player = MCTSSelfPlayer(net.policy_value, play_out_n)
