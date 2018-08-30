#include "GomokuAi.h"

#include <cmath>
#include <cstdlib>
#include <vector>
#include <map>
using std::vector;
using std::map;

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

/// @brief 训练数据池
class CTrainDataPool
{
public:
    /// @brief 构造函数
    /// @param[in] maxSize 训练池最大数据个数
    CTrainDataPool(unsigned int maxSize)
    {
        m_dataMaxSize = maxSize;

        m_dataVec.resize(maxSize);
        m_dataUsedVec.resize(maxSize);
        for (unsigned int i = 0; i < maxSize; i++)
        {
            m_dataUsedVec[i] = false;
        }
        m_dataUsedSize = 0;
    }

    /// @brief 析构函数
    ~CTrainDataPool()
    {

    }

    /// @brief 获取数据池数据数量
    unsigned int Size()
    {
        return m_dataUsedSize;
    }

    /// @brief 在数据池中创建新数据
    /// @return 成功创建返回数据地址, 失败返回nullptr, 数据池已满会失败
    LTrainData* NewData()
    {
        if (m_dataUsedSize >= m_dataMaxSize)
            return nullptr;

        // 找到一个可用空间
        for (unsigned int i = 0; i < m_dataMaxSize; i++)
        {
            if (m_dataUsedVec[i] == false)
            {
                m_dataUsedSize += 1;

                return &(m_dataVec[i]);
            }
        }
        return nullptr;
    }

    /// @brief 从数据池中随机弹出一个数据
    /// @param[out] pData 存储弹出的数据
    /// @return 成功放入返回true, 失败返回false, 数据池为空会失败
    bool Pop(OUT LTrainData* pData)
    {
        if (m_dataUsedSize < 1)
            return false;
        if (pData == nullptr)
            return false;

        int randCount = RandInt(1, (int)m_dataUsedSize);

        int count = 0;
        for (unsigned int i = 0; i < m_dataMaxSize; i++)
        {
            if (m_dataUsedVec[i] == true)
            {
                count += 1;

                if (randCount == count)
                {
                    (*pData) = m_dataVec[i];
                    break;
                }
            }
        }

        return true;
    }

private:
    unsigned int m_dataMaxSize;                 // 数据池最大数据个数
    unsigned int m_dataUsedSize;                // 记录已使用的个数
    vector<LTrainData> m_dataVec;               // 数据池
    vector<bool> m_dataUsedVec;                 // 标记数据池中的对应数据是否被使用
    
};

LTrainDataPool::LTrainDataPool(unsigned int maxSize)
{
    m_pDataPool = new CTrainDataPool(maxSize);
}

LTrainDataPool::~LTrainDataPool()
{
    if (m_pDataPool != nullptr)
    {
        delete m_pDataPool;
        m_pDataPool = nullptr;
    }
}

unsigned int LTrainDataPool::Size()
{
    return m_pDataPool->Size();
}

LTrainData* LTrainDataPool::NewData()
{
    return m_pDataPool->NewData();
}

bool LTrainDataPool::Pop(OUT LTrainData* pData)
{
    return m_pDataPool->Pop(pData);
}