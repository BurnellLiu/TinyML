

#include "LDecisionTree.h"

#include <vector>
using std::vector;
#include <string>
using std::string;
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
        PTrueChildren = nullptr;
        PFalseChildren = nullptr;
    }

    /// @brief 构造分支结点
    CDecisionTreeNode(
        IN int column,
        IN double checkValue,
        IN double featureDis,
        IN CDecisionTreeNode* pTrueChildren,
        IN CDecisionTreeNode* pFalseChildren)
    {
        CheckColumn = column;
        CheckValue = checkValue;
        FeatureDis = featureDis;

        PTrueChildren = pTrueChildren;
        PFalseChildren = pFalseChildren;
    }

    /// @brief 构造叶子结点
    CDecisionTreeNode(IN const map<double, int>& labelMap)
    {
        PTrueChildren = nullptr;
        PFalseChildren = nullptr;

        LabelMap = labelMap;
    }

    unsigned int CheckColumn;           ///< 需要检验的列索引
    double CheckValue;                  ///< 检验值, 为了使结果为true, 当前列必须匹配的值(如果是离散值则必须相等才为true, 如果是连续值则大于等于为true)
    double FeatureDis;                 ///< 特征分布, 可以为DT_FEATURE_DISCRETE或DT_FEATURE_CONTINUUM

    CDecisionTreeNode* PTrueChildren;  ///< 条件为true的分支结点
    CDecisionTreeNode* PFalseChildren; ///< 条件为false的分支结点

    map<double, int> LabelMap;         ///< 标签表<标签值, 值数量>, 叶子结点的该属性才有意义
};


/// @brief 决策树分类器
class CDecisionTreeClassifier
{
public:
    /// @brief 构造函数
    CDecisionTreeClassifier()
    {
        m_pXMatrix = nullptr;
        m_pYVector = nullptr;
        m_pNVector = nullptr;

        m_pRootNode = nullptr;
    }

    /// @brief 析构造函数
    ~CDecisionTreeClassifier()
    {
        if (m_pRootNode != nullptr)
        {
            this->RecursionDeleteTree(m_pRootNode);
            m_pRootNode = nullptr;
        }
    }

    /// @brief 训练模型
    bool TrainModel(IN const LDTMatrix& xMatrix, IN const LDTMatrix& nVector, IN const LDTMatrix& yVector)
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
            if (nVector[0][i] != DT_FEATURE_DISCRETE && nVector[0][i] != DT_FEATURE_CONTINUUM)
                return false;
        }

        // 如果已经训练过, 则删除树
        if (m_pRootNode != nullptr)
        {
            this->RecursionDeleteTree(m_pRootNode);
            m_pRootNode = nullptr;
        }

        m_pXMatrix = &xMatrix;
        m_pYVector = &yVector;
        m_pNVector = &nVector;
        m_featureNum = xMatrix.ColumnLen;
        
        // 提取样本标签
        vector<unsigned int> xIdxList;
        xIdxList.reserve(m_pXMatrix->RowLen);
        for (unsigned int i = 0; i < m_pXMatrix->RowLen; i++)
        {
            xIdxList.push_back(i);
        }

        // 递归建树
        m_pRootNode = RecursionBuildTree(xIdxList);

        m_pXMatrix = nullptr;
        m_pYVector = nullptr;
        m_pNVector = nullptr;

        return true;

    }

    /// @brief 进行剪枝操作(即合并叶子节点)
    void Prune(IN double minGain)
    {
        if (nullptr == m_pRootNode)
            return;
        if (minGain < 0.0)
            return;

        this->RecursionPrune(m_pRootNode, minGain);

    }

    /// @brief 使用训练好的模型预测数据
    bool Predict(IN const LDTMatrix& xMatrix, OUT LDTMatrix& yVector) const
    {
        // 检查参数
        if (nullptr == m_pRootNode)
            return false;
        if (xMatrix.RowLen < 1)
            return false;
        if (xMatrix.ColumnLen != m_featureNum)
            return false;

        yVector.Reset(xMatrix.RowLen, 1, 0.0);

        for (unsigned int i = 0; i < xMatrix.RowLen; i++)
        {
            this->RecursionPredicty(m_pRootNode, xMatrix, i, yVector);
        }

        return true;
        
    }

    /// @brief 计算模型得分
    double Score(IN const LDTMatrix& xMatrix, IN const LDTMatrix& yVector) const
    {
        // 检查参数
        if (nullptr == m_pRootNode)
            return -1.0;
        if (xMatrix.RowLen < 1)
            return -1.0;
        if (xMatrix.ColumnLen != m_featureNum)
            return -1.0;
        if (yVector.ColumnLen != 1)
            return -1.0;
        if (yVector.RowLen != xMatrix.RowLen)
            return -1.0;

        LDTMatrix predictVector;
        this->Predict(xMatrix, predictVector);
        if (predictVector.RowLen != yVector.RowLen)
            return -1.0;

        double trueCount = 0.0;
        for (unsigned int row = 0; row < yVector.RowLen; row++)
        {
            if (predictVector[row][0] == yVector[row][0])
                trueCount += 1.0;
        }

        return trueCount / (double)yVector.RowLen;
    }

    /// @brief 打印树, 用于调试
    void PrintTree()
    {
        printf("DecisionTreeClassifier: \n");
        this->RecursionPrintTree(m_pRootNode, "  ");
        printf("\n");
    }

private:
    /// @brief 递归打印树
    void RecursionPrintTree(IN const CDecisionTreeNode* pNode, IN string space)
    {
        if (pNode == 0)
            return;

        if (pNode->LabelMap.size() > 0)
        {
            printf("{");
            for (auto iter = pNode->LabelMap.begin(); iter != pNode->LabelMap.end(); iter++)
            {
                printf(" %.2f : %d;", iter->first, iter->second);
            }
            printf("}\n"); 
            return;
        }

        printf(" %d : %.2f ?\n", pNode->CheckColumn, pNode->CheckValue);

        printf("%sTrue->  ", space.c_str());
        this->RecursionPrintTree(pNode->PTrueChildren, space + "  ");
        printf("%sFalse->  ", space.c_str());
        this->RecursionPrintTree(pNode->PFalseChildren, space + "  ");
    }

    /// @brief 递归剪枝
    void RecursionPrune(IN CDecisionTreeNode* pNode, IN double minGain)
    {
        if (pNode == 0)
            return;

        // 叶子结点不用剪枝
        if (pNode->LabelMap.size() != 0)
            return;

        CDecisionTreeNode* pTrueChildren = pNode->PTrueChildren;
        CDecisionTreeNode* pFalseChildren = pNode->PFalseChildren;

        // true分支不是叶子节点
        if (pTrueChildren->LabelMap.size() == 0)
            this->RecursionPrune(pTrueChildren, minGain);

        // false分支不是叶子节点
        if (pFalseChildren->LabelMap.size() == 0)
            this->RecursionPrune(pFalseChildren, minGain);

        // 两个分支都是叶子节点, 则判断是否合并
        if (pTrueChildren->LabelMap.size() != 0 &&
            pFalseChildren->LabelMap.size() != 0)
        {
            int trueCount = 0;
            int falseCount = 0;
            map<double, int> totalMap;
            for (auto iter = pTrueChildren->LabelMap.begin(); iter != pTrueChildren->LabelMap.end(); iter++)
            {
                totalMap[iter->first] += iter->second;
                trueCount += iter->second;
            }

            for (auto iter = pFalseChildren->LabelMap.begin(); iter != pFalseChildren->LabelMap.end(); iter++)
            {
                totalMap[iter->first] += iter->second;
                falseCount += iter->second;
            }

            int totalCount = trueCount + falseCount;
            double gain = this->Entropy(totalMap) - 
                (double)(trueCount) / (double)(totalCount) * Entropy(pTrueChildren->LabelMap) - 
                (double)(falseCount) / (double)(totalCount) * Entropy(pFalseChildren->LabelMap);

            if (gain < minGain)
            {
                pNode->LabelMap = totalMap;
                delete pNode->PTrueChildren;
                pNode->PTrueChildren = nullptr;
                delete pNode->PFalseChildren;
                pNode->PFalseChildren = nullptr;
            }
        }
    }

    /// @brief 递归预测数据
    void RecursionPredicty(
        IN CDecisionTreeNode* pNode, 
        IN const LDTMatrix& xMatrix, 
        IN unsigned int idx, 
        OUT LDTMatrix& yVector) const
    {
        if (pNode == nullptr)
            return;

        // 叶子结点, 则计算各个标签的概率
        if (pNode->LabelMap.size() != 0)
        {
            int totalCount = 0;
            double maxProb = 0.0;
            double label;
            for (auto iter = pNode->LabelMap.begin(); iter != pNode->LabelMap.end(); iter++)
                totalCount += iter->second;
            for (auto iter = pNode->LabelMap.begin(); iter != pNode->LabelMap.end(); iter++)
            {
                double prob = (double)(iter->second) / (double)(totalCount);
                if (prob > maxProb)
                {
                    maxProb = prob;
                    label = iter->first;
                }
            }

            yVector[idx][0] = label;

            return;
        }

        // 分支结点
        double currentValue = xMatrix[idx][pNode->CheckColumn];
        double checkVallue = pNode->CheckValue;
        if (pNode->FeatureDis == DT_FEATURE_DISCRETE)
        {
            if (currentValue == checkVallue)
                this->RecursionPredicty(pNode->PTrueChildren, xMatrix, idx, yVector);
            else
                this->RecursionPredicty(pNode->PFalseChildren, xMatrix, idx, yVector);
        }
        else if (pNode->FeatureDis == DT_FEATURE_CONTINUUM)
        {
            if (currentValue >= checkVallue)
                this->RecursionPredicty(pNode->PTrueChildren, xMatrix, idx, yVector);
            else
                this->RecursionPredicty(pNode->PFalseChildren, xMatrix, idx, yVector);
        }


    }

    /// @brief 递归删除决策树
    void RecursionDeleteTree(CDecisionTreeNode* pNode)
    {
        if (pNode == 0)
            return;

        if (pNode->PFalseChildren != 0)
            this->RecursionDeleteTree(pNode->PFalseChildren);
        if (pNode->PTrueChildren != 0)
            this->RecursionDeleteTree(pNode->PTrueChildren);

        delete pNode;
    }

    /// @brief 递归构造决策树
    /// @param[in] xIdxList 样本索引列表
    /// @return 决策树节点
    CDecisionTreeNode* RecursionBuildTree(IN const vector<unsigned int>& xIdxList)
    {
        // 如果当前熵为0.0则生成叶子结点
        double currentEntropy = this->Entropy(xIdxList);
        if (currentEntropy == 0.0f)
        {

            map<double, int> labelMap;
            this->CountLabel(xIdxList, labelMap);
            return new CDecisionTreeNode(labelMap);
        }

        double maxGain = 0.0;                // 最大信息增益
        unsigned int bestCheckCol = 0;       // 最佳检查列
        double bestCheckValue;               // 最佳检查值
        double featurDis;                    // 特征分布
        vector<unsigned int> xBestTrueList;  // 最佳true分支样本索引列表
        vector<unsigned int> xBestFalseList; // 最佳false分支样本索引列表

        // 针对每个列
        for (unsigned int col = 0; col < m_pXMatrix->ColumnLen; col++)
        {
            set<double> columnValueSet; // 列中不重复的值集合

            // 当前列中生成一个由不同值构成的序列
            for (unsigned int i = 0; i < xIdxList.size(); i++)
            {
                unsigned int idx = xIdxList[i];

                columnValueSet.insert((*m_pXMatrix)[idx][col]);
            }

            // 针对一个列中的每个不同值
            for (auto iter = columnValueSet.begin(); iter != columnValueSet.end(); iter++)
            {
                double checkValue = *iter;       // 检查值
                vector<unsigned int> xTrueList;  // true分支样本索引列表
                vector<unsigned int> xFalseList; // false分支样本索引列表
                this->DevideSample(xIdxList, col, checkValue, xTrueList, xFalseList);

                double weight = (double)(xTrueList.size()) / (double)(xIdxList.size());

                // 计算信息增益
                double gain = currentEntropy - weight * this->Entropy(xTrueList) - (1 - weight) * this->Entropy(xFalseList);
                if (gain > maxGain)
                {
                    maxGain = gain;
                    bestCheckCol = col;
                    bestCheckValue = checkValue;
                    featurDis = (*m_pNVector)[0][col];
                    xBestTrueList = xTrueList;
                    xBestFalseList = xFalseList;
                }
            }
        }

        CDecisionTreeNode* pTrueChildren = this->RecursionBuildTree(xBestTrueList);
        CDecisionTreeNode* pFalseChildren = this->RecursionBuildTree(xBestFalseList);
        return new CDecisionTreeNode(bestCheckCol, bestCheckValue, featurDis, pTrueChildren, pFalseChildren);

    }

    /// @brief 拆分样本集
    /// @param[in] xIdxList 样本索引列表
    /// @param[in] column 拆分依据的列
    /// @param[in] checkValue 拆分依据的列的检查值
    /// @param[out] xTrueList 检查结果为true的样本索引列表
    /// @param[out] xFalseList 检查结果为false的样本索引列表
    void DevideSample(
        IN const vector<unsigned int>& xIdxList,
        IN unsigned int column, 
        IN double checkValue,
        OUT vector<unsigned int>& xTrueList,
        OUT vector<unsigned int>& xFalseList)
    {
        xTrueList.clear();
        xFalseList.clear();

        if ((*m_pNVector)[0][column] == DT_FEATURE_DISCRETE)
        {
            for (unsigned int i = 0; i < xIdxList.size(); i++)
            {
                unsigned int idx = xIdxList[i];

                if ((*m_pXMatrix)[idx][column] == checkValue)
                    xTrueList.push_back(idx);
                else
                    xFalseList.push_back(idx);

            }
            
        }
        
        if ((*m_pNVector)[0][column] == DT_FEATURE_CONTINUUM)
        {
            for (unsigned int i = 0; i < xIdxList.size(); i++)
            {
                unsigned int idx = xIdxList[i];

                if ((*m_pXMatrix)[idx][column] >= checkValue)
                    xTrueList.push_back(idx);
                else
                    xFalseList.push_back(idx);

            }
        }
    }

    /// @brief 统计标签
    /// @param[in] xIdxList 样本索引列表
    /// @param[out] labelMap 标签表
    void CountLabel(IN const vector<unsigned int>& xIdxList, OUT map<double, int>& labelMap)
    {
        labelMap.clear();

        for (unsigned int i = 0; i < xIdxList.size(); i++)
        {
            unsigned int idx = xIdxList[i];

            ++labelMap[(*m_pYVector)[idx][0]];
        }
    }


    /// @brief 计算样本集类别的熵
    /// 对于任意一个随机变量 X, 它的熵定义如下:
    /// 变量的不确定性越大, 熵也就越大, 把它搞清楚所需要的信息量也就越大
    /// @param[in] xIdxList 样本索引列表
    /// @return 样本集类别熵
    double Entropy(IN const vector<unsigned int>& xIdxList)
    {
        double entropy = 0.0f;
        map<double, int> typeCountMap;
        for (unsigned int i = 0; i < xIdxList.size(); i++)
        {
            unsigned int idx = xIdxList[i];
            ++typeCountMap[(*m_pYVector)[idx][0]];
        }

        for (auto iter = typeCountMap.begin(); iter != typeCountMap.end(); iter++)
        {
            double prob = (double)(iter->second) / (double)(xIdxList.size());
            entropy -= prob * log(prob) / log(2.0);
        }

        return entropy;
    }

    /// @brief 根据标签表计算熵
    double Entropy(IN const map<double, int>& labelMap)
    {
        double entropy = 0.0f;

        int totalCount = 0;
        for (auto iter = labelMap.begin(); iter != labelMap.end(); iter++)
        {
            totalCount += iter->second;
        }

        for (auto iter = labelMap.begin(); iter != labelMap.end(); iter++)
        {
            double prob = (double)(iter->second) / (double)(totalCount);
            entropy -= prob * log(prob) / log(2.0);
        }

        return entropy;
    }


private:
    const LDTMatrix* m_pXMatrix;   ///< 样本矩阵, 训练时所用临时变量
    const LDTMatrix* m_pYVector;   ///< 标签向量(列向量), 训练时所用临时变量
    const LDTMatrix* m_pNVector;   ///< 特征分布向量(行向量), 训练时所用临时变量

    unsigned int m_featureNum;      ///< 特征数

    CDecisionTreeNode* m_pRootNode; ///< 决策树根结点
    
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

bool LDecisionTreeClassifier::TrainModel(IN const LDTMatrix& xMatrix, IN const LDTMatrix& nVector, IN const LDTMatrix& yVector)
{
    return m_pClassifier->TrainModel(xMatrix, nVector, yVector);
}

void LDecisionTreeClassifier::Prune(IN double minGain)
{
    m_pClassifier->Prune(minGain);
}

bool LDecisionTreeClassifier::Predict(IN const LDTMatrix& xMatrix, OUT LDTMatrix& yVector) const
{
    return m_pClassifier->Predict(xMatrix, yVector);
}

double LDecisionTreeClassifier::Score(IN const LDTMatrix& xMatrix, IN const LDTMatrix& yVector) const
{
    return m_pClassifier->Score(xMatrix, yVector);
}



void LDecisionTreeClassifier::PrintTree()
{
    m_pClassifier->PrintTree();
}


/// @brief 回归树节点
struct CDTRegressionNode
{
    unsigned int CheckColumn;           ///< 需要检验的列索引, 叶子结点该值无意义
    double CheckValue;                  ///< 检验值, 为了使结果为true, 当前列必须匹配的值(如果是离散值则必须相等才为true, 如果是连续值则大于等于为true), 叶子结点该值无意义
    double FeatureDis;                  ///< 特征分布, 可以为DT_FEATURE_DISCRETE或DT_FEATURE_CONTINUUM, 叶子结点该值无意义

    CDTRegressionNode* PTrueChildren;   ///< 条件为true的分支结点, 叶子结点该值为nullptr
    CDTRegressionNode* PFalseChildren;  ///< 条件为false的分支结点, 叶子结点该值为nullptr

    double Mean;                        ///< 该结点代表的均值
    double LossValue;                   ///< 该结点的损失值
};

/// @brief 复制回归树
/// @param[in] pNode 需要复制的树结点
/// @return 复制后的新结点
static CDTRegressionNode* RegressionTreeCopy(IN const CDTRegressionNode* pNode)
{
    if (pNode == nullptr)
        return nullptr;

    CDTRegressionNode* pNewNode = new CDTRegressionNode();
    (*pNewNode) = (*pNode);

    pNewNode->PTrueChildren = nullptr;
    pNewNode->PFalseChildren = nullptr;

    pNewNode->PTrueChildren = RegressionTreeCopy(pNode->PTrueChildren);
    pNewNode->PFalseChildren = RegressionTreeCopy(pNode->PFalseChildren);

    return pNewNode;

}

/// @brief 回归树损失值
/// @param[in] pNode 需要计算损失值的结点
/// @param[out] lossValue 存储损失值
/// @param[out] leafCount 存储叶子结点数量
static void RegressionTreeLossValue(IN CDTRegressionNode* pNode, OUT double& lossValue, OUT unsigned int& leafCount)
{
    if (pNode == nullptr)
        return;

    if (pNode->PTrueChildren != nullptr)
        RegressionTreeLossValue(pNode->PTrueChildren, lossValue, leafCount);
    if (pNode->PFalseChildren != nullptr)
        RegressionTreeLossValue(pNode->PFalseChildren, lossValue, leafCount);

    if (pNode->PTrueChildren == nullptr &&
        pNode->PFalseChildren == nullptr)
    {
        lossValue += pNode->LossValue;
        leafCount += 1;
    }

}

/// @brief 后序遍历内部结点
/// @param[in] pNode 需要遍历的结点
/// @param[out] nodeList 存储内部结点
static void RegressionTreeInsideNodes(IN CDTRegressionNode* pNode, OUT vector<CDTRegressionNode*>& nodeList)
{
    if (pNode == nullptr)
        return;

    if (pNode->PTrueChildren != nullptr)
        RegressionTreeInsideNodes(pNode->PTrueChildren, nodeList);
    if (pNode->PFalseChildren != nullptr)
        RegressionTreeInsideNodes(pNode->PFalseChildren, nodeList);

    if (pNode->PTrueChildren != nullptr &&
        pNode->PFalseChildren != nullptr)
        nodeList.push_back(pNode);

}

/// @brief 回归树预测
/// @param[in] pNode 预测结点
/// @param[in] xMatrix 需要预测的样本矩阵
/// @param[in] idx 需要预测的样本索引
/// @param[out] yVector 存储预测结果
static void RegressionTreePredicty(
    IN CDTRegressionNode* pNode,
    IN const LDTMatrix& xMatrix,
    IN unsigned int idx,
    OUT LDTMatrix& yVector)
{
    if (pNode == nullptr)
        return;

    // 叶子结点
    if (pNode->PTrueChildren == nullptr &&
        pNode->PFalseChildren == nullptr)
    {
        yVector[idx][0] = pNode->Mean;
        return;
    }

    // 分支结点
    double currentValue = xMatrix[idx][pNode->CheckColumn];
    double checkVallue = pNode->CheckValue;
    if (pNode->FeatureDis == DT_FEATURE_DISCRETE)
    {
        if (currentValue == checkVallue)
            RegressionTreePredicty(pNode->PTrueChildren, xMatrix, idx, yVector);
        else
            RegressionTreePredicty(pNode->PFalseChildren, xMatrix, idx, yVector);
    }
    else if (pNode->FeatureDis == DT_FEATURE_CONTINUUM)
    {
        if (currentValue >= checkVallue)
            RegressionTreePredicty(pNode->PTrueChildren, xMatrix, idx, yVector);
        else
            RegressionTreePredicty(pNode->PFalseChildren, xMatrix, idx, yVector);
    }
}

/// @brief 删除回归树
/// @param[in] pNode 需要删除的结点
static void RegressionTreeDelete(IN CDTRegressionNode* pNode)
{
    if (pNode == nullptr)
        return;

    if (pNode->PTrueChildren != nullptr)
        RegressionTreeDelete(pNode->PTrueChildren);
    if (pNode->PFalseChildren != nullptr)
        RegressionTreeDelete(pNode->PFalseChildren);


    delete pNode;
}

/// @brief 打印回归树
static void RegressionTreePrint(IN const CDTRegressionNode* pNode, IN string space)
{
    if (pNode == nullptr)
        return;

    if (pNode->PTrueChildren == nullptr &&
        pNode->PFalseChildren == nullptr)
    {
        printf("{Mean: %.2f LossValue: %.2f}\n", pNode->Mean, pNode->LossValue);
        return;
    }

    printf(" %d : %.2f ?\n", pNode->CheckColumn, pNode->CheckValue);

    printf("%sTrue->  ", space.c_str());
    RegressionTreePrint(pNode->PTrueChildren, space + "  ");
    printf("%sFalse->  ", space.c_str());
    RegressionTreePrint(pNode->PFalseChildren, space + "  ");
}


/// @brief 回归树
class CDecisionTreeRegression
{
public:
    /// @brief 构造函数
    CDecisionTreeRegression()
    {
        m_pXMatrix = nullptr;
        m_pNVector = nullptr;
        m_pYVector = nullptr;

        m_pRootNode = nullptr;
    }

    /// @brief 析构函数
    ~CDecisionTreeRegression()
    {
        if (m_pRootNode != nullptr)
        {
            RegressionTreeDelete(m_pRootNode);
            m_pRootNode = nullptr;
        }
    }

    /// @brief 训练模型
    bool TrainModel(IN const LDTMatrix& xMatrix, IN const LDTMatrix& nVector, IN const LDTMatrix& yVector)
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
            if (nVector[0][i] != DT_FEATURE_DISCRETE && nVector[0][i] != DT_FEATURE_CONTINUUM)
                return false;
        }

        // 如果已经训练过, 则删除树
        if (m_pRootNode != nullptr)
        {
            RegressionTreeDelete(m_pRootNode);
            m_pRootNode = nullptr;
        }

        // 将样本集拆分为训练集和验证集, 30%作为验证集
        unsigned int verifySampleCount = (unsigned int)(xMatrix.RowLen * 0.3);
        LDTMatrix verifyXMatrix;
        LDTMatrix trainXMatrix;
        xMatrix.SubMatrix(0, verifySampleCount, 0, xMatrix.ColumnLen, verifyXMatrix);
        xMatrix.SubMatrix(verifySampleCount, xMatrix.RowLen - verifySampleCount, 0, xMatrix.ColumnLen, trainXMatrix);
        LDTMatrix verifyYVector;
        LDTMatrix trainYVector;
        yVector.SubMatrix(0, verifySampleCount, 0, yVector.ColumnLen, verifyYVector);
        yVector.SubMatrix(verifySampleCount, yVector.RowLen - verifySampleCount, 0, yVector.ColumnLen, trainYVector);

        m_pXMatrix = &trainXMatrix;
        m_pYVector = &trainYVector;
        m_pNVector = &nVector;
        m_featureNum = xMatrix.ColumnLen;

        // 提取样本标签
        vector<unsigned int> xIdxList;
        xIdxList.reserve(m_pXMatrix->RowLen);
        for (unsigned int i = 0; i < m_pXMatrix->RowLen; i++)
        {
            xIdxList.push_back(i);
        }

        // 建一个尽量大的树
        double lossValue = 0.0;
        double mean = 0.0;
        this->CalculateLossValue(xIdxList, mean, lossValue);
        CDTRegressionNode* pTree = RecursionBuildTree(xIdxList, mean, lossValue);

        // 剪枝后的树列表
        vector<CDTRegressionNode*> treeList;

        while (true)
        {
            // 复制树并且保存起来
            CDTRegressionNode* pNewTree = RegressionTreeCopy(pTree);
            treeList.push_back(pNewTree);

            // 如果树被剪枝只剩根结点则退出
            if (pTree->PTrueChildren == nullptr &&
                pTree->PFalseChildren == nullptr)
            {
                break;
            }

            // 获取树所有内部结点
            vector<CDTRegressionNode*> nodeList;
            RegressionTreeInsideNodes(pTree, nodeList);

            // 针对每个结点计算对该结点进行剪枝后整个树损失函数减少的程度
            // 选择树损失函数减少的最小的结点进行剪枝
            double minAlpha = -1.0;
            CDTRegressionNode* pPruneNode = nullptr;
            for (auto iter = nodeList.begin(); iter != nodeList.end(); iter++)
            {
                lossValue = 0.0;
                unsigned int leafCount = 0;
                RegressionTreeLossValue(*iter, lossValue, leafCount);
                double alpha = ((*iter)->LossValue - lossValue) / (leafCount - 1);
                
                if (minAlpha == -1.0)
                {
                    minAlpha = alpha;
                    pPruneNode = *iter;
                    continue;
                }

                if (alpha < minAlpha)
                {
                    minAlpha = alpha;
                    pPruneNode = *iter;
                }
            }

            // 进行剪枝
            RegressionTreeDelete(pPruneNode->PTrueChildren);
            RegressionTreeDelete(pPruneNode->PFalseChildren);
            pPruneNode->PTrueChildren = nullptr;
            pPruneNode->PFalseChildren = nullptr;
        }

        // 使用验证集在被剪枝的树中选择一个最优的
        double minErrorSum = -1.0;
        CDTRegressionNode* pBestTree = nullptr;
        for (auto iter = treeList.begin(); iter != treeList.end(); iter++)
        {
            double errorSum = 0.0;
            LDTMatrix predictY(verifyYVector.RowLen, 1, 0.0);
            for (unsigned int i = 0; i < verifyXMatrix.RowLen; i++)
            {
                RegressionTreePredicty(*iter, verifyXMatrix, i, predictY);
                double dif = verifyYVector[i][0] - predictY[i][0];
                errorSum += dif * dif;
            }

            if (minErrorSum == -1.0)
            {
                minErrorSum = errorSum;
                pBestTree = *iter;
                continue;
            }

            if (errorSum < minErrorSum)
            {
                // 删除被抛弃的树
                RegressionTreeDelete(pBestTree);
                pBestTree = nullptr;

                minErrorSum = errorSum;
                pBestTree = *iter;
            }
            else
            {
                // 删除被抛弃的树
                RegressionTreeDelete(*iter);
            }

            *iter = nullptr;
            
        }

        m_pRootNode = pBestTree;

        m_pXMatrix = nullptr;
        m_pYVector = nullptr;
        m_pNVector = nullptr;

        return true;
    }

    /// @brief 使用训练好的模型预测数据
    bool Predict(IN const LDTMatrix& xMatrix, OUT LDTMatrix& yVector) const
    {
        // 检查参数
        if (nullptr == m_pRootNode)
            return false;
        if (xMatrix.RowLen < 1)
            return false;
        if (xMatrix.ColumnLen != m_featureNum)
            return false;

        yVector.Reset(xMatrix.RowLen, 1, 0.0);

        for (unsigned int i = 0; i < xMatrix.RowLen; i++)
        {
            RegressionTreePredicty(m_pRootNode, xMatrix, i, yVector);
        }

        return true;
    }

    /// @brief 计算模型得分
    double Score(IN const LDTMatrix& xMatrix, IN const LDTMatrix& yVector) const
    {
        LDTMatrix predictY;
        bool bRet = this->Predict(xMatrix, predictY);
        if (!bRet)
            return 2.0;


        double sumY = 0.0;
        for (unsigned int i = 0; i < yVector.RowLen; i++)
        {
            sumY += yVector[i][0];
        }
        double meanY = sumY / (double)yVector.RowLen;

        double lossValue = 0.0;
        double denominator = 0.0;
        for (unsigned int i = 0; i < yVector.RowLen; i++)
        {
            double dis = yVector[i][0] - predictY[i][0];
            lossValue += dis * dis;

            dis = yVector[i][0] - meanY;
            denominator += dis * dis;
        }

        double score = 1.0 - lossValue / denominator;

        return score;
    }

    /// @brief 打印树, 用于调试
    void PrintTree() const
    {
        printf("Regression Tree: \n");
        RegressionTreePrint(m_pRootNode, "  ");
        printf("\n");
    }

private:
    /// @brief 递归构造决策树
    /// @param[in] xIdxList 样本索引列表
    /// @return 决策树节点
    CDTRegressionNode* RecursionBuildTree(IN const vector<unsigned int>& xIdxList, double mean, double lossValue)
    {
        CDTRegressionNode* pNode = new CDTRegressionNode();

        // 存储当前结点的均值和损失值
        pNode->Mean = mean;
        pNode->LossValue = lossValue;

        // 如果当前损失值为0.0则生成叶子结点
        if (lossValue == 0.0f)
        {
            pNode->PTrueChildren = nullptr;
            pNode->PFalseChildren = nullptr;
            return pNode;
        }

        double minLossValue = lossValue;     // 最小损失值
        unsigned int bestCheckCol = 0;       // 最佳检查列
        double bestCheckValue;               // 最佳检查值
        double bestColFeaturDis;             // 最佳列特征分布
        vector<unsigned int> xBestTrueList;  // 最佳true分支样本索引列表
        vector<unsigned int> xBestFalseList; // 最佳false分支样本索引列表
        double bestTrueMean;
        double bestTrueLossValue;
        double bestFalseMean;
        double bestFalseLossValue;

         // 针对每个列
        for (unsigned int col = 0; col < m_pXMatrix->ColumnLen; col++)
        {
            set<double> columnValueSet; // 列中不重复的值集合

             // 当前列中生成一个由不同值构成的序列
            for (unsigned int i = 0; i < xIdxList.size(); i++)
            {
                unsigned int idx = xIdxList[i];

                columnValueSet.insert((*m_pXMatrix)[idx][col]);
            }

            // 针对一个列中的每个不同值
            for (auto iter = columnValueSet.begin(); iter != columnValueSet.end(); iter++)
            {
                double checkValue = *iter;       // 检查值
                vector<unsigned int> xTrueList;  // true分支样本索引列表
                vector<unsigned int> xFalseList; // false分支样本索引列表
                this->DevideSample(xIdxList, col, checkValue, xTrueList, xFalseList);
                if (xTrueList.size() == 0 ||
                    xFalseList.size() == 0)
                    continue;

                double trueMean;
                double trueLossValue;
                double falseMean;
                double falseLossValue;
                this->CalculateLossValue(xTrueList, trueMean, trueLossValue);
                this->CalculateLossValue(xFalseList, falseMean, falseLossValue);
                double sumLossValue = trueLossValue + falseLossValue;

                if (sumLossValue < minLossValue)
                {
                    minLossValue = sumLossValue;
                    bestCheckCol = col;
                    bestCheckValue = checkValue;
                    bestColFeaturDis = (*m_pNVector)[0][col];
                    xBestTrueList = xTrueList;
                    xBestFalseList = xFalseList;

                    bestTrueMean = trueMean;
                    bestTrueLossValue = trueLossValue;
                    bestFalseMean = falseMean;
                    bestFalseLossValue = falseLossValue;
                }
            }
        }

        CDTRegressionNode* pTrueChildren = this->RecursionBuildTree(xBestTrueList, bestTrueMean, bestTrueLossValue);
        CDTRegressionNode* pFalseChildren = this->RecursionBuildTree(xBestFalseList, bestFalseMean, bestFalseLossValue);
        pNode->PTrueChildren = pTrueChildren;
        pNode->PFalseChildren = pFalseChildren;
        pNode->CheckColumn = bestCheckCol;
        pNode->CheckValue = bestCheckValue;
        pNode->FeatureDis = bestColFeaturDis;

        return pNode;

    }

    /// @brief 拆分样本集
    /// @param[in] xIdxList 样本索引列表
    /// @param[in] column 拆分依据的列
    /// @param[in] checkValue 拆分依据的列的检查值
    /// @param[out] xTrueList 检查结果为true的样本索引列表
    /// @param[out] xFalseList 检查结果为false的样本索引列表
    void DevideSample(
        IN const vector<unsigned int>& xIdxList,
        IN unsigned int column,
        IN double checkValue,
        OUT vector<unsigned int>& xTrueList,
        OUT vector<unsigned int>& xFalseList)
    {
        xTrueList.clear();
        xFalseList.clear();

        if ((*m_pNVector)[0][column] == DT_FEATURE_DISCRETE)
        {
            for (unsigned int i = 0; i < xIdxList.size(); i++)
            {
                unsigned int idx = xIdxList[i];

                if ((*m_pXMatrix)[idx][column] == checkValue)
                    xTrueList.push_back(idx);
                else
                    xFalseList.push_back(idx);

            }

        }

        if ((*m_pNVector)[0][column] == DT_FEATURE_CONTINUUM)
        {
            for (unsigned int i = 0; i < xIdxList.size(); i++)
            {
                unsigned int idx = xIdxList[i];

                if ((*m_pXMatrix)[idx][column] >= checkValue)
                    xTrueList.push_back(idx);
                else
                    xFalseList.push_back(idx);

            }
        }
    }

    /// @brief 计算样本损失值(二乘法)
    /// @param[in] xIdxList 样本索引列表
    /// @return 样本损失值
    void CalculateLossValue(IN const vector<unsigned int>& xIdxList, OUT double& mean, OUT double& lossValue) const
    {
        mean = 0.0;
        lossValue = 0.0;

        for (auto iter = xIdxList.begin(); iter != xIdxList.end(); iter++)
        {
            unsigned int idx = *iter;
            mean += (*m_pYVector)[idx][0];
        }

        mean = mean / (double)xIdxList.size();

        for (auto iter = xIdxList.begin(); iter != xIdxList.end(); iter++)
        {
            unsigned int idx = *iter;
            double dif = (*m_pYVector)[idx][0] - mean;
            lossValue += dif * dif;
        }
    }

private:

    const LDTMatrix* m_pXMatrix;   ///< 样本矩阵, 训练时所用临时变量
    const LDTMatrix* m_pYVector;   ///< 标签向量(列向量), 训练时所用临时变量
    const LDTMatrix* m_pNVector;   ///< 特征分布向量(行向量), 训练时所用临时变量

    unsigned int m_featureNum;      ///< 特征数

    CDTRegressionNode* m_pRootNode; ///< 决策树根结点

};

LDecisionTreeRegression::LDecisionTreeRegression()
{
    m_pRegressor = nullptr;
    m_pRegressor = new CDecisionTreeRegression();
}

LDecisionTreeRegression::~LDecisionTreeRegression()
{
    if (m_pRegressor != nullptr)
    {
        delete m_pRegressor;
        m_pRegressor = nullptr;
    }
}

bool LDecisionTreeRegression::TrainModel(IN const LDTMatrix& xMatrix, IN const LDTMatrix& nVector, IN const LDTMatrix& yVector)
{
    return m_pRegressor->TrainModel(xMatrix, nVector, yVector);
}

bool LDecisionTreeRegression::Predict(IN const LDTMatrix& xMatrix, OUT LDTMatrix& yVector) const
{
    return m_pRegressor->Predict(xMatrix, yVector);
}

double LDecisionTreeRegression::Score(IN const LDTMatrix& xMatrix, IN const LDTMatrix& yVector) const
{
    return m_pRegressor->Score(xMatrix, yVector);
}

void LDecisionTreeRegression::PrintTree() const
{
    return m_pRegressor->PrintTree();
}

