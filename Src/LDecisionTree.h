/// @file LDecision.h
/// @brief  决策树头文件
/// 
/// Detail:该文件定义了实现决策树的实现算法, 如分类与回归树算法(CART, Classification And Regression Tree)
/// @author Burnell_Liu  
/// @version   
/// @date 5:6:2015

#ifndef _LDECISION_H_
#define _LDECISION_H_

#include <string>
using std::string;
#include <map>
using std::map;

#include "LDataStruct/LArray.h"


#ifndef IN
#define IN
#endif

#ifndef INOUT
#define INOUT
#endif

#ifndef OUT
#define OUT
#endif

/// @brief 变体
class LVariant
{
public:
    enum VALUE_TYPE
    {
        UNKNOWN = 0, // 未知
        INT, ///< 数值整型
        STRING ///< 字符串型
    };
public:
    LVariant();
    LVariant(IN int value);
    LVariant(IN const string& value);
    LVariant(IN const LVariant& rhs);
    ~LVariant();

    LVariant& operator = (IN const LVariant& rhs);

    bool operator < (IN const LVariant& rhs) const;

    /// @brief 获取变体中存储的数据类型
    ///  
    /// @return 数据类型
    VALUE_TYPE GetType() const;

    /// @brief 设置整形数值
    ///  
    /// @param[in] value 整形数值
    void SetIntValue(IN int value);

    /// @brief 获取整形数值
    ///  
    /// @return 整形值
    int GetIntValue() const;

    /// @brief 设置字符串值
    ///  
    /// @param[in] value 字符串值
    void SetStringValue(IN const string& value);

    /// @brief 获取字符串值
    ///  
    /// @return 字符串值
    string GetStringValue() const;

private:
    int* m_pValueInt; ///< 整形数值
    string* m_pValueStr; ///< 字符串值
    VALUE_TYPE m_type; ///< 数值类型
};


typedef LArray<LVariant> LDTDataList;
typedef LArray<LDTDataList> LDTDataSet;

struct LDecisionTreeNode;

/// @brief 决策树
class LDecisionTree
{
public:
    LDecisionTree();
    ~LDecisionTree();

    /// @brief 构造决策树
    ///  
    /// 数据集由多行数据组成, 每一行数据都包含一组观测变量和一个结果值
    /// 观测变量和结果值可以是string或int
    /// 同一列的值要么全为string要么全为int, 不可以有unknown数据
    /// @param[in] dataSet 训练数据集
    /// @return 构造失败返回false, 训练数据集格式不正确会失败
    bool BuildTree(IN const LDTDataSet& dataSet);

    /// @brief 进行剪枝操作(即合并叶子节点)
    ///
    /// 如果两个叶子节点合并后增加的熵小于指定的最小值, 则进行合并操作
    /// @param[in] minGain 最小增加值
    void Prune(IN float minGain);

    /// @brief 对新数据进行分类
    ///  
    /// @param[in] dataList 数据列表
    /// @param[out] result 分类结果
    /// @return 成功分类返回true, 失败返回false
    bool Classify(IN const LDTDataList& dataList, OUT map<LVariant, float>& resultMap);

    /// @brief 打印决策树(用于调试)
    void PrintTree();

private:
    /// @brief 递归构造决策树
    ///  
    /// @param[in] dataSet 数据集
    /// @return 决策树节点
    LDecisionTreeNode* RecursionBuildTree(IN const LDTDataSet& dataSet);

    /// @brief 递归进行剪枝操作(即合并叶子节点)
    ///
    /// 如果两个叶子节点合并后增加的熵小于指定的最小值, 则进行合并操作
    /// @param[in] pNode 决策树节点
    /// @param[in] minGain 最小增加值
    void RecursionPrune(IN LDecisionTreeNode* pNode, IN float minGain);

    /// @brief 对新数据进行递归分类
    ///  
    /// @param[in] pNode 决策树节点
    /// @param[in] dataList 数据列表
    /// @param[out] result 分类结果
    /// @return 成功分类返回true, 失败返回false
    bool RecursionClassify(IN LDecisionTreeNode* pNode, IN const LDTDataList& dataList, OUT map<LVariant, int>& resultMap);

    /// @brief 递归删除决策树
    ///  
    /// @param[in] pNode 决策树节点
    void RecursionDeleteTree(IN LDecisionTreeNode* pNode);

    /// @brief 递归打印树
    ///  
    /// @param[in] pNode 决策树节点
    /// @param[in] space 空格
    void RecursionPrintTree(IN const LDecisionTreeNode* pNode, IN string space);

    /// @brief 检查数据集是否符合要求
    ///  
    /// @param[in] dataSet 数据集
    /// @return 符合要求返回true, 不符合要求返回false
    bool CheckDataSet(IN const LDTDataSet& dataSet);

    /// @brief 拆分数据集
    ///  
    /// @param[in] dataSet 需要拆分的数据集
    /// @param[in] column 拆分依据的列
    /// @param[in] checkValue 拆分依据的列的检查值
    /// @param[out] trueSet 检查结果为true的数据集
    /// @param[out] falseSet 检查结果为false的数据集
    void DevideSet(
        IN const LDTDataSet& dataSet, 
        IN int column, 
        IN LVariant& checkValue, 
        OUT LDTDataSet& trueSet, 
        OUT LDTDataSet& falseSet);

    /// @brief 统计结果列
    ///  
    /// @param[in] dataSet 数据集合
    /// @param[out] resultMap 结果字典
    void CountResult(IN const LDTDataSet& dataSet, OUT map<LVariant, int>& resultMap);

    /// @brief 计算数据集分类的熵
    ///  
    /// 每行数据列的最后一项为分类值
    /// @param[in] dataSet 数据集
    /// @return 数据集分类熵
    float Entropy(IN const LDTDataSet& dataSet);

private:
    LDecisionTreeNode* m_pRootNode; ///< 决策树根节点
};





#endif