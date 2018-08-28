#pragma once

#include "LNeuralNetwork.h"

#define CHESS_BOARD_ROW     15          // 棋盘行数
#define CHESS_BOARD_COLUMN  15          // 棋盘列数

#define SPOT_WHITE          1.0         // 白子
#define SPOT_NONE           0.5         // 无子
#define SPOT_BLACK          0.0         // 黑子

#define GAME_WIN            1.0         // 赢棋
#define GAME_DRAWN          0.5         // 和棋
#define GAME_LOSE           0.0         // 输棋

typedef LMatrix<double> LChessBoard;    // 棋盘

/// @brief 棋子位置
struct LChessPos 
{
    unsigned int Row;       // 行数 
    unsigned int Col;       // 列数
};

/// @brief 训练数据
struct LTrainData
{
    LChessBoard State;      // 当前状态
    LChessPos   Action;     // 执行动作(落子位置)
    double      Reward;     // 回报值, 1.0(白子赢), 0.5(和棋), 0.0(白子输棋)
    LChessBoard NextState;  // 下个状态
};

/// @brief 五子棋Ai, 执白子
class LGomokuAi
{
public:
    /// @brief 构造函数
    LGomokuAi();

    /// @brief 析构函数
    ~LGomokuAi();

    /// @brief 落子
    /// @param[in] chessBoard 当前棋局
    /// @param[in] e 随机执行动作的概率[0-1](不思考, 随机执行)
    /// @param[out] pPos 存储落子位置
    void Action(IN const LChessBoard& chessBoard, IN double e, OUT LChessPos* pPos);

    /// @brief 训练
    /// @param[in] data 训练数据
    void Train(IN const LTrainData& data);
private:
    LBPNetwork* m_pBrain;       // Ai的大脑

    LNNMatrix m_inputCache;         // 输入缓存, 提高程序执行效率
    LNNMatrix m_outputCache;        // 输出缓存, 提高程序执行效率
    LNNMatrix m_trainInputCache1;   // 输入缓存, 提高程序执行效率
    LNNMatrix m_trainoutputCache1;  // 输出缓存, 提高程序执行效率
    LNNMatrix m_trainInputCache2;   // 输入缓存, 提高程序执行效率
    LNNMatrix m_trainoutputCache2;  // 输出缓存, 提高程序执行效率
    
};

