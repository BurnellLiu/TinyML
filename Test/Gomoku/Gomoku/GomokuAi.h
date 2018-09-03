#pragma once

#include "LMatrix.h"

#define CHESS_BOARD_ROW     15          // 棋盘行数
#define CHESS_BOARD_COLUMN  15          // 棋盘列数

#define SPOT_WHITE          1.0         // 白子
#define SPOT_NONE           0.5         // 无子
#define SPOT_BLACK          0.0         // 黑子

#define GAME_WIN_SCORE      0.1         // 赢棋得分
#define GAME_DRAWN_SCORE    0.05        // 和棋得分
#define GAME_LOSE_SCORE     0.0         // 输棋得分

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
    bool        GameEnd;    // 标记游戏是否结束
    LChessBoard State;      // 当前状态
    LChessPos   Action;     // 执行动作(落子位置)
    double      Reward;     // 回报值(得分值), 0.1(白子赢), 0.05(和棋), 0.0(白子输棋)
    LChessBoard NextState;  // 下个状态
};

class CGomokuAi;

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

    /// @brief 将五子棋Ai保存到文件中
    /// @param[in] pFilePath 文件路径
    void Save2File(IN char* pFilePath);

    /// @brief 从文件中加载五子棋Ai
    /// @param[in] pFilePath 文件路径
    void LoadFromFile(IN char* pFilePath);

private:
    CGomokuAi* m_pGomokuAi;         // 五子棋Ai实现对象
};

class CTrainDataPool;

/// @brief 训练数据池
class LTrainDataPool
{
public:
    /// @brief 构造函数
    /// @param[in] maxSize 训练池最大数据个数
    LTrainDataPool(unsigned int maxSize);

    /// @brief 析构函数
    ~LTrainDataPool();

    /// @brief 获取数据池数据数量
    unsigned int Size();

    /// @brief 在数据池中创建新数据
    /// @return 成功创建返回数据地址, 失败返回nullptr, 数据池已满会失败
    LTrainData* NewData();

    /// @brief 从数据池中随机弹出一个数据
    /// @param[out] pData 存储弹出的数据
    /// @return 成功放入返回true, 失败返回false, 数据池为空会失败
    bool Pop(OUT LTrainData* pData);

private:
    CTrainDataPool* m_pDataPool;            // 数据池实现对象
};

