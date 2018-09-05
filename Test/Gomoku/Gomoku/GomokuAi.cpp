#include "GomokuAi.h"
#include "LNeuralNetwork.h"

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

/// @brief 打印矩阵
static void MatrixDebugPrint(IN const LNNMatrix& dataMatrix)
{
    DebugPrint("Matrix Row: %u  Col: %u\n", dataMatrix.RowLen, dataMatrix.ColumnLen);
    for (unsigned int i = 0; i < dataMatrix.RowLen; i++)
    {
        for (unsigned int j = 0; j < dataMatrix.ColumnLen; j++)
        {
            DebugPrint("%.5f  ", dataMatrix[i][j]);
        }
        DebugPrint("\n");
    }
    DebugPrint("\n");
}


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

/// @brief 五子棋Ai, 执白子
class CGomokuAi
{
public:
    /// @brief 构造函数
    /// @param[in] param Ai参数
    CGomokuAi(const LAiParam& param)
    {
        m_aiParam = param;

        LBPNetworkPogology pogology;
        pogology.InputNumber = CHESSMAN_NUM;
        pogology.OutputNumber = CHESSMAN_NUM;
        pogology.HiddenLayerNumber = param.BrainLayersNum;
        pogology.NeuronsOfHiddenLayer = param.LayerNeuronsNum;
        m_pBrain = new LBPNetwork(pogology);

        m_inputCache.Reset(1, CHESSMAN_NUM);
        m_trainInputCache1.Reset(1, CHESSMAN_NUM);
        m_trainInputCache2.Reset(1, CHESSMAN_NUM);

        m_actionVecCache.reserve(CHESSMAN_NUM);
    }

    /// @brief 析构函数
    ~CGomokuAi()
    {
        if (m_pBrain != nullptr)
        {
            delete m_pBrain;
            m_pBrain = nullptr;
        }
    }

    /// @brief 落子
    /// @param[in] chessBoard 当前棋局
    /// @param[in] e 随机执行动作的概率[0-1](不思考, 随机执行)
    /// @param[out] pPos 存储落子位置
    void Action(IN const LChessBoard& chessBoard, IN double e, OUT LChessPos* pPos)
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
            m_actionVecCache.clear();
            for (unsigned int row = 0; row < chessBoard.RowLen; row++)
            {
                for (unsigned int col = 0; col < chessBoard.ColumnLen; col++)
                {
                    if (chessBoard[row][col] == SPOT_NONE)
                        m_actionVecCache.push_back(row * CHESS_BOARD_COLUMN + col);
                }
            }

            unsigned int actionIdx = 0;
            if (m_actionVecCache.size() > 1)
            {
                int i = RandInt(0, int(m_actionVecCache.size() - 1));
                actionIdx = i;
            }

            unsigned int action = m_actionVecCache[actionIdx];

            pPos->Row = action / CHESS_BOARD_COLUMN;
            pPos->Col = action % CHESS_BOARD_COLUMN;

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
        double maxAction = GAME_LOSE_SCORE;
        
        for (unsigned int i = 0; i < m_outputCache.ColumnLen; i++)
        {
            if (m_outputCache[0][i] > maxAction)
            {
                m_actionVecCache.clear();
                m_actionVecCache.push_back(i);
                maxAction = m_outputCache[0][i];
            }
            else if (m_outputCache[0][i] == maxAction)
            {
                m_actionVecCache.push_back(i);
            }
        }

        unsigned int actionIdx = 0;
        if (m_actionVecCache.size() > 1)
        {
            int i = RandInt(0, int(m_actionVecCache.size() - 1));
            actionIdx = i;
        }

        unsigned int action = m_actionVecCache[actionIdx];

        pPos->Row = action / CHESS_BOARD_COLUMN;
        pPos->Col = action % CHESS_BOARD_COLUMN;
    }

    /// @brief 训练
    /// @param[in] datas 训练数据
    void Train(IN const vector<LTrainData>& datas)
    {
        m_trainInputCache1.Reset((unsigned int)datas.size(), CHESSMAN_NUM);

        for (unsigned int i = 0; i < datas.size(); i++)
        {
            const LTrainData& data = datas[i];

            for (unsigned int row = 0; row < data.State.RowLen; row++)
            {
                for (unsigned int col = 0; col < data.State.ColumnLen; col++)
                {
                    unsigned int idx = row * CHESS_BOARD_COLUMN + col;
                    m_trainInputCache1[i][idx] = data.State[row][col];
                }
            }
        }
        m_pBrain->Active(m_trainInputCache1, &m_trainOutputCache1);

        for (unsigned int i = 0; i < datas.size(); i++)
        {
            const LTrainData& data = datas[i];
            double newActionValue = 0.0;
            unsigned int action = data.Action.Row * CHESS_BOARD_COLUMN + data.Action.Col;

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

                m_pBrain->Active(m_trainInputCache2, &m_trainOutputCache2);

                double currentActionValue = m_trainOutputCache1[i][action];

                double nextActionValueMax = GAME_LOSE_SCORE;
                for (unsigned int col = 0; col < m_trainOutputCache2.ColumnLen; col++)
                {
                    if (m_trainOutputCache2[0][col] > nextActionValueMax)
                        nextActionValueMax = m_trainOutputCache2[0][col];
                }

                double difValue = data.Reward + m_aiParam.QLearningGamma * nextActionValueMax - currentActionValue;
                newActionValue = currentActionValue + m_aiParam.QLearningRate * difValue;
                if (newActionValue < GAME_LOSE_SCORE)
                    newActionValue = GAME_LOSE_SCORE;
                if (newActionValue > GAME_WIN_SCORE)
                    newActionValue = GAME_WIN_SCORE;
            }
            else
            {
                newActionValue = data.Reward;
            }

            m_trainOutputCache1[i][action] = newActionValue;
        }


        unsigned int trainCount = m_aiParam.BrainTrainCount;
        for (unsigned int i = 0; i < trainCount; i++)
        {
            double rate = (m_aiParam.BrainLearningRate * (trainCount - i)) / trainCount;
            m_pBrain->Train(m_trainInputCache1, m_trainOutputCache1, (float)rate);
        }

    }

    /// @brief 将五子棋Ai保存到文件中
    /// @param[in] pFilePath 文件路径
    void Save2File(IN char* pFilePath)
    {
        m_pBrain->Save2File(pFilePath);
    }

    /// @brief 从文件中加载五子棋Ai
    /// @param[in] pFilePath 文件路径
    void LoadFromFile(IN char* pFilePath)
    {
        if (m_pBrain != nullptr)
        {
            delete m_pBrain;
            m_pBrain = nullptr;
        }

        m_pBrain = new LBPNetwork(pFilePath);
    }

private:
    LBPNetwork* m_pBrain;                   // Ai的大脑
    LAiParam    m_aiParam;                  // Ai参数

    LNNMatrix m_inputCache;                 // 输入缓存, 提高程序执行效率
    LNNMatrix m_outputCache;                // 输出缓存, 提高程序执行效率
    LNNMatrix m_trainInputCache1;           // 输入缓存, 提高程序执行效率
    LNNMatrix m_trainOutputCache1;          // 输出缓存, 提高程序执行效率
    LNNMatrix m_trainInputCache2;           // 输入缓存, 提高程序执行效率
    LNNMatrix m_trainOutputCache2;          // 输出缓存, 提高程序执行效率

    vector<unsigned int> m_actionVecCache;  // 动作缓存, 提高程序执行效率

};

LGomokuAi::LGomokuAi(const LAiParam& param)
{
    m_pGomokuAi = new CGomokuAi(param);
}

LGomokuAi::~LGomokuAi()
{
    if (m_pGomokuAi != nullptr)
    {
        delete m_pGomokuAi;
        m_pGomokuAi = nullptr;
    }
}

void LGomokuAi::Action(IN const LChessBoard& chessBoard, IN double e, OUT LChessPos* pPos)
{
    m_pGomokuAi->Action(chessBoard, e, pPos);
}

void LGomokuAi::Train(IN const vector<LTrainData>& datas)
{
    m_pGomokuAi->Train(datas);
}

void LGomokuAi::Save2File(IN char* pFilePath)
{
    m_pGomokuAi->Save2File(pFilePath);
}

void LGomokuAi::LoadFromFile(IN char* pFilePath)
{
    m_pGomokuAi->LoadFromFile(pFilePath);
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

                m_dataUsedVec[i] = true;
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
                    m_dataUsedSize -= 1;
                    m_dataUsedVec[i] = false;
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