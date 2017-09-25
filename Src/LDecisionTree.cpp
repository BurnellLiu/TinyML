

#include "LDecisionTree.h"

#include <cmath>
#include <cstdio>

#include <set>
using std::set;

/// @brief 计算数据的熵
/// 
/// 对于任意一个随机变量 X, 它的熵定义如下:
/// 变量的不确定性越大, 熵也就越大, 把它搞清楚所需要的信息量也就越大
/// @param[in] dataList 数据列表
/// @return 数据的熵
template<typename T>
static float CalculateEntropy(IN const LArray<T>& dataList)
{
    float entropy = 0.0f;
    map<T, int> typeCountMap;
    for (int i = 0; i < dataList.Length; i++)
    {
        ++typeCountMap[dataList.Data[i]];
    }

    for (auto iter = typeCountMap.begin(); iter != typeCountMap.end(); iter++)
    {
        float prob = (float)(iter->second)/(float)(dataList.Length);
        entropy -= prob * log(prob)/log(2.0f);
    }

    return entropy;
}

static float CalculateEntropy(IN const map<LVariant, int>& dataMap)
{
    float entropy = 0.0f;

    int totalCount = 0;
    for (auto iter = dataMap.begin(); iter != dataMap.end(); iter++)
    {
        totalCount += iter->second;
    }

    for (auto iter = dataMap.begin(); iter != dataMap.end(); iter++)
    {
        float prob = (float)(iter->second)/(float)(totalCount);
        entropy -= prob * log(prob)/log(2.0f);
    }

    return entropy;
}


/// @brief决策树节点
struct LDecisionTreeNode
{
    LDecisionTreeNode()
    {
        m_checkColumn = -1;
        m_pTrueChildren = 0;
        m_pFalseChildren = 0;
    }

    LDecisionTreeNode(
        IN int column, 
        IN const LVariant& checkValue, 
        IN LDecisionTreeNode* pTrueChildren, 
        IN LDecisionTreeNode* pFalseChildren)
    {
        m_checkColumn = column;
        m_checkValue = checkValue;
        m_pTrueChildren = pTrueChildren;
        m_pFalseChildren = pFalseChildren;
    }

    LDecisionTreeNode(IN const map<LVariant, int>& resultMap)
    {
        m_resultMap = resultMap;
    }

    int m_checkColumn; ///< 需要检验的列索引
    LVariant m_checkValue; ///< 为了使结果为true, 当前列必须匹配的值(如果变体是字符串则必须相等才为true, 如果变体是浮点数则大于等于为true)

    LDecisionTreeNode* m_pTrueChildren; ///< 条件为true的分支节点
    LDecisionTreeNode* m_pFalseChildren; ///< 条件为false的分支节点

    map<LVariant, int> m_resultMap; ///< 当前分支的结果, 除了叶节点外, 在其他节点上该值都没有
};

LVariant::LVariant()
{
    m_pValueInt = 0;
    m_pValueStr = 0;
    m_type = UNKNOWN;
}

LVariant::LVariant(IN int value)
{
    
    m_pValueInt = 0;
    m_pValueStr = 0;

    m_pValueInt = new int;
    (*m_pValueInt) = value;
    m_type = INT;
    
}

LVariant::LVariant(IN const string& value)
{
    m_pValueInt = 0;
    m_pValueStr = 0;

    m_pValueStr = new string;
    (*m_pValueStr) = value;
    m_type = STRING;
}

LVariant::LVariant(IN const LVariant& rhs)
{
    this->m_type = rhs.m_type;
    this->m_pValueInt = 0;
    this->m_pValueStr = 0;

    if (rhs.m_pValueInt != 0)
    {
        this->m_pValueInt = new int(*rhs.m_pValueInt);
    }

    if (rhs.m_pValueStr != 0)
    {
        this->m_pValueStr = new string(*rhs.m_pValueStr);
    }
}

LVariant::~LVariant()
{
    if (m_pValueInt != 0)
    {
        delete m_pValueInt;
        m_pValueInt = 0;
    }

    if (m_pValueStr != 0)
    {
        delete m_pValueStr;
        m_pValueStr = 0;
    }

    m_type = UNKNOWN;
}

LVariant& LVariant::operator = (IN const LVariant& rhs)
{
    if (this->m_pValueInt != 0)
    {
        delete m_pValueInt;
        this->m_pValueInt = 0;
    }

    if (this->m_pValueStr != 0)
    {
        delete m_pValueStr;
        this->m_pValueStr = 0;
    }

    this->m_type = rhs.m_type;

    if (rhs.m_pValueInt != 0)
    {
        this->m_pValueInt = new int(*rhs.m_pValueInt);
    }

    if (rhs.m_pValueStr != 0)
    {
        this->m_pValueStr = new string(*rhs.m_pValueStr);
    }

    return *this;
}

bool LVariant::operator < (IN const LVariant& rhs) const
{
    if (this->GetType() == rhs.GetType() &&
        this->GetType() == INT)
    {
        return this->GetIntValue() < rhs.GetIntValue();
    }

    if (this->GetType() == rhs.GetType() &&
        this->GetType() == STRING)
    {
        return this->GetStringValue() < rhs.GetStringValue();
    }

    return false;
}

LVariant::VALUE_TYPE LVariant::GetType() const
{
    return m_type;
}

int LVariant::GetIntValue() const
{
    return *m_pValueInt;
}

void LVariant::SetIntValue(IN int value)
{
    if (m_pValueInt != 0)
    {
        delete m_pValueInt;
        m_pValueInt = 0;
    }

    m_pValueInt = new int;
    (*m_pValueInt) = value;

    m_type = INT;
}

string LVariant::GetStringValue() const
{
    return *m_pValueStr;
}

void LVariant::SetStringValue(IN const string& value)
{
    if (m_pValueStr != 0)
    {
        delete m_pValueStr;
        m_pValueStr = 0;
    }
    m_pValueStr = new string(value);

    m_type = STRING;
}

LDecisionTree::LDecisionTree()
{
    m_pRootNode = 0;
}

LDecisionTree::~LDecisionTree()
{
    if (m_pRootNode != 0)
    {
        this->RecursionDeleteTree(m_pRootNode);
        m_pRootNode = 0;
    }
}

bool LDecisionTree::BuildTree(IN const LDTDataSet& dataSet)
{
    if (m_pRootNode != 0)
    {
        this->RecursionDeleteTree(m_pRootNode);
        m_pRootNode = 0;
    }

    bool bRet = CheckDataSet(dataSet);

    if (bRet)
    {
        m_pRootNode = this->RecursionBuildTree(dataSet);
        return true;
    }

    return false;

    
}

void LDecisionTree::Prune(IN float minGain)
{
    this->RecursionPrune(m_pRootNode, minGain);
}

bool LDecisionTree::Classify(IN const LDTDataList& dataList, OUT map<LVariant, float>& resultMap)
{
    resultMap.clear();

    map<LVariant, int> countResultMap;
    bool bRet =this->RecursionClassify(m_pRootNode, dataList, countResultMap);

    if (!bRet)
        return false;

    int totalCount = 0;

    for (auto iter = countResultMap.begin(); iter != countResultMap.end(); iter++)
    {
        totalCount += iter->second;
    }

    for (auto iter = countResultMap.begin(); iter != countResultMap.end(); iter++)
    {
        resultMap[iter->first] = (float)(iter->second)/(float)(totalCount);
    }

    return true;

}

void LDecisionTree::PrintTree()
{
    this->RecursionPrintTree(m_pRootNode, "  ");
}

LDecisionTreeNode* LDecisionTree::RecursionBuildTree(IN const LDTDataSet& dataSet)
{
    if (dataSet.Length < 1)
        return 0;

    float currentEntropy = this->Entropy(dataSet);
    if (currentEntropy == 0.0f)
    {

        map<LVariant, int> resultMap;
        this->CountResult(dataSet, resultMap);
        return new LDecisionTreeNode(resultMap);
    }

    float bestGain = 0.0f;
    int bestCheckCol = -1;
    LVariant bestCheckValue;
    LDTDataSet bestTrueSet;
    LDTDataSet bestFalseSet;


    for (int col = 0; col < dataSet[0].Length-1; col++)
    {
        set<int> columnValueListInt;
        set<string> columnValueListStr;

        // 当前列中生成一个由不同值构成的序列
        for (int row = 0; row < dataSet.Length; row++)
        {
            if (dataSet[row][col].GetType() == LVariant::INT)
            {
                columnValueListInt.insert(dataSet[row][col].GetIntValue());
            }

            if (dataSet[row][col].GetType() == LVariant::STRING)
            {
                columnValueListStr.insert(dataSet[row][col].GetStringValue());
            }
        }

        // 数据格式错误, 同一列中存在两种数据类型
        if (columnValueListStr.size() != 0 && columnValueListInt.size() != 0)
        {
            return 0;
        }

        if (columnValueListInt.size() != 0 && columnValueListInt.size() != 1)
        {
            for (auto iter = columnValueListInt.begin(); iter != columnValueListInt.end(); iter++)
            {
                LVariant checkValue(*iter);
                LDTDataSet trueSet;
                LDTDataSet falseSet;
                this->DevideSet(dataSet, col, checkValue, trueSet, falseSet);

                float weight = (float)(trueSet.Length)/(float)(dataSet.Length);
                float gain = currentEntropy - weight * this->Entropy(trueSet) - (1-weight) * this->Entropy(falseSet);
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

        if (columnValueListStr.size() != 0 && columnValueListStr.size() != 1)
        {
            for (auto iter = columnValueListStr.begin(); iter != columnValueListStr.end(); iter++)
            {
                LVariant checkValue(*iter);
                LDTDataSet trueSet;
                LDTDataSet falseSet;
                this->DevideSet(dataSet, col, checkValue, trueSet, falseSet);

                float weight = (float)(trueSet.Length)/(float)(dataSet.Length);
                float gain = currentEntropy - weight * this->Entropy(trueSet) - (1-weight) * this->Entropy(falseSet);
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

    if (bestGain > 0.0f)
    {
        LDecisionTreeNode* pTrueChildren = this->RecursionBuildTree(bestTrueSet);
        LDecisionTreeNode* pFalseChildren = this->RecursionBuildTree(bestFalseSet);
        return new LDecisionTreeNode(bestCheckCol, bestCheckValue, pTrueChildren, pFalseChildren);
    }
    else
    {
        map<LVariant, int> resultMap;
        this->CountResult(dataSet, resultMap);
        return new LDecisionTreeNode(resultMap);
    }

    return 0;
}

void LDecisionTree::RecursionPrune(IN LDecisionTreeNode* pNode, IN float minGain)
{
    if (pNode == 0)
        return;

    if (pNode->m_resultMap.size() != 0)
        return;

    // true分支不是叶子节点
    if (pNode->m_pTrueChildren->m_resultMap.size() == 0)
        this->RecursionPrune(pNode->m_pTrueChildren, minGain);
    
    // false分支不是叶子节点
    if (pNode->m_pFalseChildren->m_resultMap.size() == 0)
        this->RecursionPrune(pNode->m_pFalseChildren, minGain);

    // 两个分支都是叶子节点, 则判断是否合并
    if (pNode->m_pTrueChildren->m_resultMap.size() != 0 &&
        pNode->m_pFalseChildren->m_resultMap.size() != 0)
    {
        int trueCount = 0;
        int falseCount = 0;
        map<LVariant, int> totalMap;
        for (auto iter = pNode->m_pTrueChildren->m_resultMap.begin(); iter != pNode->m_pTrueChildren->m_resultMap.end(); iter++)
        {
            totalMap[iter->first] += iter->second;
            trueCount += iter->second;
        }

        for (auto iter = pNode->m_pFalseChildren->m_resultMap.begin(); iter != pNode->m_pFalseChildren->m_resultMap.end(); iter++)
        {
            totalMap[iter->first] += iter->second;
            falseCount += iter->second;
        }

        int totalCount = trueCount + falseCount;
        float gain = CalculateEntropy(totalMap) - (float)(trueCount)/(float)(totalCount) * CalculateEntropy(pNode->m_pTrueChildren->m_resultMap)
            - (float)(falseCount)/(float)(totalCount) * CalculateEntropy(pNode->m_pFalseChildren->m_resultMap);

        if (gain < minGain)
        {
            pNode->m_checkColumn = -1;
            pNode->m_resultMap = totalMap;
            delete pNode->m_pTrueChildren;
            pNode->m_pTrueChildren = 0;
            delete pNode->m_pFalseChildren;
            pNode->m_pFalseChildren = 0;
        }
    }
}

bool LDecisionTree::RecursionClassify(IN LDecisionTreeNode* pNode, IN const LDTDataList& dataList, OUT map<LVariant, int>& resultMap)
{
    if (pNode == 0)
        return false;

    if (pNode->m_resultMap.size() != 0)
    {
        resultMap = pNode->m_resultMap;

        return true;
    }

    if (pNode->m_checkColumn >= dataList.Length)
        return false;


    if (dataList[pNode->m_checkColumn].GetType() == pNode->m_checkValue.GetType() &&
        pNode->m_checkValue.GetType() == LVariant::INT)
    {
        if (dataList[pNode->m_checkColumn].GetIntValue() >= pNode->m_checkValue.GetIntValue())
            return this->RecursionClassify(pNode->m_pTrueChildren, dataList, resultMap);
        else
            return this->RecursionClassify(pNode->m_pFalseChildren, dataList, resultMap);
    }

    if (dataList[pNode->m_checkColumn].GetType() == pNode->m_checkValue.GetType() &&
        pNode->m_checkValue.GetType() == LVariant::STRING)
    {
        if (dataList[pNode->m_checkColumn].GetStringValue() == pNode->m_checkValue.GetStringValue())
            return this->RecursionClassify(pNode->m_pTrueChildren, dataList, resultMap);
        else
            return this->RecursionClassify(pNode->m_pFalseChildren, dataList, resultMap);
    }

    // 缺失数据处理
    if (dataList[pNode->m_checkColumn].GetType() == LVariant::UNKNOWN)
    {
        map<LVariant, int> trueResultMap;
        map<LVariant, int> falseResultMap;
        bool bTrueRet = this->RecursionClassify(pNode->m_pTrueChildren, dataList, trueResultMap);
        bool bFalseRet = this->RecursionClassify(pNode->m_pFalseChildren, dataList, falseResultMap);
        if (!bTrueRet || !bFalseRet)
            return false;

        for (auto iter = trueResultMap.begin(); iter != trueResultMap.end(); iter++)
        {
            resultMap[iter->first] += iter->second;
        }
        for (auto iter = falseResultMap.begin(); iter != falseResultMap.end(); iter++)
        {
            resultMap[iter->first] += iter->second;
        }

        return true;

    }

    return false;
}

void LDecisionTree::RecursionDeleteTree(LDecisionTreeNode* pNode)
{
    if (pNode == 0)
        return;

    if (pNode->m_pFalseChildren != 0)
        this->RecursionDeleteTree(pNode->m_pFalseChildren);
    if (pNode->m_pTrueChildren != 0)
        this->RecursionDeleteTree(pNode->m_pTrueChildren);

    delete pNode;
}



void LDecisionTree::RecursionPrintTree(IN const LDecisionTreeNode* pNode, IN string space)
{
    if (pNode == 0)
        return;


    if (pNode->m_resultMap.size() > 0)
    {
        auto fristItem = pNode->m_resultMap.begin();
        if (fristItem->first.GetType() == LVariant::STRING)
        {
            printf("{");
            for (auto iter = pNode->m_resultMap.begin(); iter != pNode->m_resultMap.end(); iter++)
            {
                printf(" %s : %d;", iter->first.GetStringValue().c_str(), iter->second);
            }
            printf("}\n");
        }

        if (fristItem->first.GetType() == LVariant::INT)
        {
            printf("{");
            for (auto iter = pNode->m_resultMap.begin(); iter != pNode->m_resultMap.end(); iter++)
            {
                printf(" %d : %d;", iter->first.GetIntValue(), iter->second);
            }
            printf("}\n");
        }

        return;
    }

    if (pNode->m_checkValue.GetType() == LVariant::STRING)
    {
        printf(" %d : %s ?\n", pNode->m_checkColumn, pNode->m_checkValue.GetStringValue().c_str());
    }

    if (pNode->m_checkValue.GetType() == LVariant::INT)
    {
        printf(" %d : %d ?\n", pNode->m_checkColumn, pNode->m_checkValue.GetIntValue());
    }

    printf("%sTrue->  ", space.c_str());
    this->RecursionPrintTree(pNode->m_pTrueChildren, space + "  ");
    printf("%sFalse->  ", space.c_str());
    this->RecursionPrintTree(pNode->m_pFalseChildren, space + "  ");
}

bool LDecisionTree::CheckDataSet(IN const LDTDataSet& dataSet)
{
    int rows = dataSet.Length;

    // 数据不是多行
    if (rows < 1)
        return false;

    // 数据行不是由观测变量和结果值组成
    int cols = dataSet[0].Length;
    if (cols < 2)
        return false;

    // 各个数据行长度不同
    for (int i = 0; i < rows; i++)
    {
        if (dataSet[i].Length != cols)
            return false;
    }

    // 同一列的观测变量同时存在int和string
    for (int col = 0; col < cols; col++)
    {
        bool intFound = false;
        bool strFound = false;
        for (int row = 0; row < rows; row++)
        {
            if (dataSet[row][col].GetType() == LVariant::STRING)
                strFound = true;
            if (dataSet[row][col].GetType() == LVariant::INT)
                intFound = true;
            if (dataSet[row][col].GetType() == LVariant::UNKNOWN)
                return false;
        }

        if (intFound && strFound)
            return false;
    }

    return true;
}

void LDecisionTree::DevideSet(
    IN const LDTDataSet& dataSet, 
    IN int column, 
    IN LVariant& checkValue, 
    OUT LDTDataSet& trueSet, 
    OUT LDTDataSet& falseSet)
{
    if (dataSet.Length <= 1)
        return;

    if (column < 0 || column >= dataSet[0].Length)
        return;

    if (checkValue.GetType() == LVariant::UNKNOWN)
        return;

    int trueSetLength = 0;
    int falseSetLength = 0;

    LVariant::VALUE_TYPE type = checkValue.GetType();

    if (type == LVariant::INT)
    {
        for (int i = 0; i < dataSet.Length; i++)
        {
            if (dataSet.Data[i].Data[column].GetType() != LVariant::INT &&
                dataSet.Data[i].Data[column].GetType() != LVariant::UNKNOWN)
                return;

            if (dataSet.Data[i].Data[column].GetType() == LVariant::UNKNOWN)
            {
                falseSetLength++;
            }
            if (dataSet.Data[i].Data[column].GetType() == LVariant::INT)
            {
                if (dataSet.Data[i].Data[column].GetIntValue() >= checkValue.GetIntValue())
                    trueSetLength++;
                else
                    falseSetLength++;
            }
        }
    }
  
    if (type == LVariant::STRING)
    {
        for (int i = 0; i < dataSet.Length; i++)
        {
            if (dataSet.Data[i].Data[column].GetType() != LVariant::STRING &&
                dataSet.Data[i].Data[column].GetType() != LVariant::UNKNOWN)
                return;

            if (dataSet.Data[i].Data[column].GetType() == LVariant::UNKNOWN)
            {
                falseSetLength++;
            }

            if (dataSet.Data[i].Data[column].GetType() == LVariant::STRING)
            {
                if (dataSet.Data[i].Data[column].GetStringValue() == checkValue.GetStringValue())
                    trueSetLength++;
                else
                    falseSetLength++;
            }
           
        }
    }

    trueSet.Reset(trueSetLength);
    falseSet.Reset(falseSetLength);
    int trueSetIndex = 0;
    int falseSetIndex = 0;

    if (type == LVariant::INT)
    {
        for (int i = 0; i < dataSet.Length; i++)
        {
            if (dataSet.Data[i].Data[column].GetType() == LVariant::UNKNOWN)
            {
                falseSet.Data[falseSetIndex] = dataSet.Data[i];
                falseSetIndex++;
                continue;
            }

            if (dataSet.Data[i].Data[column].GetIntValue() >= checkValue.GetIntValue())
            {
                trueSet.Data[trueSetIndex] = dataSet.Data[i];
                trueSetIndex++;
            }
            else
            {
                falseSet.Data[falseSetIndex] = dataSet.Data[i];
                falseSetIndex++;
            }
        }
    }

    if (type == LVariant::STRING)
    {
        for (int i = 0; i < dataSet.Length; i++)
        {

            if (dataSet.Data[i].Data[column].GetType() == LVariant::UNKNOWN)
            {
                falseSet.Data[falseSetIndex] = dataSet.Data[i];
                falseSetIndex++;
                continue;
            }

            if (dataSet.Data[i].Data[column].GetStringValue() == checkValue.GetStringValue())
            {
                trueSet.Data[trueSetIndex] = dataSet.Data[i];
                trueSetIndex++;
            }
            else
            {
                falseSet.Data[falseSetIndex] = dataSet.Data[i];
                falseSetIndex++;
            }
        }
    }

}



void LDecisionTree::CountResult(IN const LDTDataSet& dataSet, OUT map<LVariant, int>& resultMap)
{
    if (dataSet.Length < 1)
        return;

    if (dataSet[0].Length < 1)
        return;


    resultMap.clear();

    int col = dataSet[0].Length - 1;
    for (int row = 0; row < dataSet.Length; row++)
    {
        ++resultMap[dataSet[row][col]];
    }
}

float LDecisionTree::Entropy(IN const LDTDataSet& dataSet)
{
    LArray<string> strDataList;
    LArray<int> floatDataList;

    float entropy = 0.0f;

    if (dataSet.Length < 1)
        return entropy;

    if (dataSet[0].Length < 1)
        return entropy;

    int column = dataSet[0].Length-1;
    if (dataSet[0][column].GetType() == LVariant::INT)
    {
        floatDataList.Reset(dataSet.Length);

        for (int i = 0; i < dataSet.Length; i++)
        {
            floatDataList[i] = dataSet[i][column].GetIntValue();
        }

        entropy = CalculateEntropy(floatDataList);
    }

    if (dataSet[0][column].GetType() == LVariant::STRING)
    {
        strDataList.Reset(dataSet.Length);

        for (int i = 0; i < dataSet.Length; i++)
        {
            strDataList[i] = dataSet[i][column].GetStringValue();
        }

        entropy = CalculateEntropy(strDataList);
    }

    return entropy;
}

