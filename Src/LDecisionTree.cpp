

#include "LDecisionTree.h"

#include <vector>
using std::vector;
#include <set>
using std::set;
#include <map>
using std::map;


/// @brief决策树节点
struct CDecisionTreeNode
{
    /// @brief 默认构造函数
    CDecisionTreeNode()
    {
        PTrueChildren = 0;
        PFalseChildren = 0;
    }

    /// @brief 构造分支结点
    CDecisionTreeNode(
        IN int column,
        IN double checkValue,
        IN CDecisionTreeNode* pTrueChildren,
        IN CDecisionTreeNode* pFalseChildren)
    {
        CheckColumn = column;
        CheckValue = checkValue;
        PTrueChildren = pTrueChildren;
        PFalseChildren = pFalseChildren;
    }

    /// @brief 构造叶子结点
    CDecisionTreeNode(IN const map<double, int>& labelMap)
    {
        PTrueChildren = 0;
        PFalseChildren = 0;

        LabelMap = labelMap;
    }

    unsigned int CheckColumn; ///< 需要检验的列索引
    double CheckValue; ///< 检验值, 为了使结果为true, 当前列必须匹配的值(如果是离散值则必须相等才为true, 如果是连续值则大于等于为true)

    CDecisionTreeNode* PTrueChildren; ///< 条件为true的分支结点
    CDecisionTreeNode* PFalseChildren; ///< 条件为false的分支结点

    map<double, int> LabelMap; ///< 标签表<标签值, 值数量>, 叶子结点的该属性才有意义
};


/// @brief 决策树分类器
class CDecisionTreeClassifier
{
public:
    /// @brief 构造函数
    CDecisionTreeClassifier()
    {

    }

    /// @brief 析构造函数
    ~CDecisionTreeClassifier()
    {

    }

    /// @brief 训练模型
    bool TrainModel(IN const LDTCMatrix& xMatrix, IN const LDTCMatrix& nVector, IN const LDTCMatrix& yVector)
    {
        // 检查参数
        if (xMatrix.RowLen < 1)
            return false;
        if (xMatrix.ColumnLen < 1)
            return false;
        if (nVector.RowLen != 1)
            return false;
        if (nVector.ColumnLen != xMatrix.ColumnLen)
            return false;
        if (yVector.ColumnLen != 1)
            return false;
        if (yVector.RowLen != xMatrix.RowLen)
            return false;
        for (unsigned int i = 0; i < nVector.ColumnLen; i++)
        {
            if (nVector[0][i] != FEATURE_DISCRETE && nVector[0][i] != FEATURE_CONTINUUM)
                return false;
        }

        // 复制样本数据
        m_xMatrix = xMatrix;
        m_nVector = nVector;
        m_yVector = yVector;

        return true;

    }

private:

    /// @brief 递归构造决策树
    /// @param[in] idxList 样本索引列表
    /// @return 决策树节点
    CDecisionTreeNode* RecursionBuildTree(IN const vector<unsigned int>& idxList)
    {
        // 如果当前熵为0.0则生成叶子结点
        double currentEntropy = this->Entropy(idxList);
        if (currentEntropy == 0.0f)
        {

            map<double, int> labelMap;
            this->CountLabel(idxList, labelMap);
            return new CDecisionTreeNode(labelMap);
        }

        float maxGain = 0.0f; // 最大信息增益
        unsigned int bestCheckCol = 0; // 最佳检查列
        double bestCheckValue; // 最佳检查值
        vector<unsigned int> bestTrueList;
        vector<unsigned int> bestFalseList;

        // 针对每个列
        for (unsigned int col = 0; col < m_xMatrix.ColumnLen; col++)
        {
            set<double> columnValueSet;

            // 当前列中生成一个由不同值构成的序列
            for (unsigned int i = 0; i < idxList.size(); i++)
            {
                unsigned int idx = idxList[i];

                columnValueSet.insert(m_xMatrix[idx][col]);
            }

            for (auto iter = columnValueSet.begin(); iter != columnValueSet.end(); iter++)
            {
                double checkValue = *iter;
                vector<unsigned int> trueList;
                vector<unsigned int> falseList;
                this->DevideSample(idxList, col, checkValue, trueList, falseList);

                double weight = (double)(trueList.size()) / (double)(idxList.size());
                float gain = currentEntropy - weight * this->Entropy(trueSet) - (1 - weight) * this->Entropy(falseSet);
                if (gain > bestGain && trueSet.Length != 0 && falseSet.Length != 0)
                {
                    bestGain = gain;
                    bestCheckCol = col;
                    bestCheckValue = checkValue;
                    bestTrueSet = trueSet;
                    bestFalseSet = falseSet;
                }
            }
        }
    }

    /// @brief 拆分样本集
    /// @param[in] idxList 样本索引列表
    /// @param[in] column 拆分依据的列
    /// @param[in] checkValue 拆分依据的列的检查值
    /// @param[out] trueList 检查结果为true的数据集
    /// @param[out] falseList 检查结果为false的数据集
    void DevideSample(
        IN const vector<unsigned int>& idxList,
        IN unsigned int column, 
        IN double checkValue,
        OUT vector<unsigned int>& trueList,
        OUT vector<unsigned int>& falseList)
    {
        if (m_nVector[0][column] == FEATURE_DISCRETE)
        {
            for (unsigned int i = 0; i < idxList.size(); i++)
            {
                unsigned int idx = idxList[i];

                if (m_xMatrix[idx][column] == checkValue)
                    trueList.push_back(idx);
                else
                    falseList.push_back(idx);

            }
            
        }
        
        if (m_nVector[0][column] == FEATURE_CONTINUUM)
        {
            for (unsigned int i = 0; i < idxList.size(); i++)
            {
                unsigned int idx = idxList[i];

                if (m_xMatrix[idx][column] >= checkValue)
                    trueList.push_back(idx);
                else
                    falseList.push_back(idx);

            }
        }
    }

    /// @brief 统计标签
    /// @param[in] idxList 样本索引列表
    /// @param[out] labelMap 标签表
    void CountLabel(IN const vector<unsigned int>& idxList, OUT map<double, int>& labelMap)
    {
        labelMap.clear();
        for (unsigned int i = 0; i < idxList.size(); i++)
        {
            unsigned int idx = idxList[i];

            ++labelMap[m_yVector[idx][0]];
        }
    }


    /// @brief 计算样本集类别的熵
    /// @param[in] idxList 样本索引列表
    /// @return 样本集类别熵
    double Entropy(IN const vector<unsigned int>& idxList)
    {
        double entropy = 0.0f;
        map<double, int> typeCountMap;
        for (unsigned int i = 0; i < idxList.size(); i++)
        {
            unsigned int idx = idxList[i];
            ++typeCountMap[m_yVector[idx][0]];
        }

        for (auto iter = typeCountMap.begin(); iter != typeCountMap.end(); iter++)
        {
            double prob = (double)(iter->second) / (double)(idxList.size());
            entropy -= prob * log(prob) / log(2.0);
        }

        return entropy;
    }


private:
    LDTCMatrix m_xMatrix; ///< 样本矩阵
    LDTCMatrix m_nVector; ///< 特征分布向量(行向量)
    LDTCMatrix m_yVector; ///< 标签向量(列向量)
    
};

LDecisionTreeClassifier::LDecisionTreeClassifier()
{
    m_pClassifier = nullptr;
    m_pClassifier = new CDecisionTreeClassifier();
}

LDecisionTreeClassifier::~LDecisionTreeClassifier()
{
    if (m_pClassifier != nullptr)
    {
        delete m_pClassifier;
        m_pClassifier = nullptr;
    }
}

bool LDecisionTreeClassifier::TrainModel(IN const LDTCMatrix& xMatrix, IN const LDTCMatrix& nVector, IN const LDTCMatrix& yVector)
{
    return m_pClassifier->TrainModel(xMatrix, nVector, yVector);
}

