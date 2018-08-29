#include "GomokuAi.h"

#include <cmath>
#include <cstdlib>
#include <vector>
using std::vector;

#ifdef _DEBUG
#define DebugPrint(format, ...) printf(format, __VA_ARGS__)
#else
#define DebugPrint(format, ...)
#endif

#define CHESSMAN_NUM CHESS_BOARD_ROW*CHESS_BOARD_COLUMN     // 棋子总数

/// @brief 产生随机小数, 范围0~1
/// @return 随机小数
static float RandFloat()
{
    return (rand()) / (RAND_MAX + 1.0f);
}

/// @brief 产生随机整数
/// @param[in] min 随机整数的最小值
/// @param[in] max 随机整数的最大值
/// @return 随机整数
inline int RandInt(int min, int max)
{
    return rand() % (max - min + 1) + min;
}

LGomokuAi::LGomokuAi()
{
    LBPNetworkPogology pogology;
    pogology.InputNumber = CHESSMAN_NUM;
    pogology.OutputNumber = CHESSMAN_NUM;
    pogology.HiddenLayerNumber = 2;
    pogology.NeuronsOfHiddenLayer = 128;
    m_pBrain = nullptr;
    m_pBrain = new LBPNetwork(pogology);

    m_inputCache.Reset(1, CHESSMAN_NUM);
}

LGomokuAi::~LGomokuAi()
{
    if (m_pBrain != nullptr)
    {
        delete m_pBrain;
        m_pBrain = nullptr;
    }
}

void LGomokuAi::Action(IN const LChessBoard& chessBoard, IN double e, OUT LChessPos* pPos)
{
    if (chessBoard.RowLen != CHESS_BOARD_ROW || 
        chessBoard.ColumnLen != CHESS_BOARD_COLUMN)
        return;
    if (e < 0.0 || e > 1.0)
        return;
    if (pPos == nullptr)
        return;


    // 随机执行
    if (RandFloat() < e)
    {
        int row = RandInt(0, CHESS_BOARD_ROW - 1);
        int col = RandInt(0, CHESS_BOARD_COLUMN - 1);
        pPos->Row = (unsigned int)row;
        pPos->Col = (unsigned int)col;
        return;
    }

    // 思考后执行
    for (unsigned int row = 0; row < chessBoard.RowLen; row++)
    {
        for (unsigned int col = 0; col < chessBoard.ColumnLen; col++)
        {
            unsigned int idx = row * CHESS_BOARD_COLUMN + col;
            m_inputCache[0][idx] = chessBoard[row][col];
        }
    }

    m_pBrain->Active(m_inputCache, &m_outputCache);

    // 找出最大动作值
    double maxAction = 0.0;
    vector<unsigned int> actionVec;
    for (unsigned int i = 0; i < m_outputCache.ColumnLen; i++)
    {
        if (m_outputCache[0][i] > maxAction)
        {
            actionVec.clear();
            actionVec.push_back(i);
            maxAction = m_outputCache[0][i];
        }
        else if (m_outputCache[0][i] == maxAction)
        {
            actionVec.push_back(i);
        }
    }

    unsigned int actionIdx = 0;
    if (actionVec.size() > 1)
    {
        int i = RandInt(0, int(actionVec.size() - 1));
        actionIdx = i;
    }

    unsigned int action = actionVec[actionIdx];

    pPos->Row = action / CHESS_BOARD_COLUMN;
    pPos->Col = action % CHESS_BOARD_COLUMN;

}

void LGomokuAi::Train(IN const LTrainData& data)
{
    double newActionValue = 0.0;
    unsigned int action = data.Action.Row * CHESS_BOARD_COLUMN + data.Action.Col;

    for (unsigned int row = 0; row < data.State.RowLen; row++)
    {
        for (unsigned int col = 0; col < data.State.ColumnLen; col++)
        {
            unsigned int idx = row * CHESS_BOARD_COLUMN + col;
            m_trainInputCache1[0][idx] = data.State[row][col];
        }
    }
    m_pBrain->Active(m_trainInputCache1, &m_trainoutputCache1);

    if (!data.GameEnd)
    {
        for (unsigned int row = 0; row < data.NextState.RowLen; row++)
        {
            for (unsigned int col = 0; col < data.NextState.ColumnLen; col++)
            {
                unsigned int idx = row * CHESS_BOARD_COLUMN + col;
                m_trainInputCache2[0][idx] = data.NextState[row][col];
            }
        }

        m_pBrain->Active(m_trainInputCache2, &m_trainoutputCache2);

        double currentActionValue = m_trainoutputCache1[0][action];

        double nextActionValueMax = 0.0;
        for (unsigned int col = 0; col < m_trainoutputCache2.ColumnLen; col++)
        {
            if (m_trainoutputCache2[0][col] > nextActionValueMax)
                nextActionValueMax = m_trainoutputCache2[0][col];
        }

        newActionValue = currentActionValue + 0.5 * (data.Reward + 1.0 * nextActionValueMax - currentActionValue);
        if (newActionValue < 0.0)
            newActionValue = 0.0;
        if (newActionValue > 1.0)
            newActionValue = 1.0;
    }
    else
    {
        newActionValue = data.Reward;
    }
    

    m_trainoutputCache1[0][action] = newActionValue;

    unsigned int trainCount = 3000;
    for (unsigned int i = 0; i < trainCount; i++)
    {
        float rate = 3.0f * (trainCount - i) / trainCount;
        m_pBrain->Train(m_trainInputCache1, m_trainoutputCache1, rate);
    }
    
}