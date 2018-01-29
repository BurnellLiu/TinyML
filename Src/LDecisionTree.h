/// @file LDecision.h
/// @brief  决策树头文件
/// 
/// Detail:该文件定义了实现决策树的实现算法, 如分类与回归树算法(CART, Classification And Regression Tree)
/// @author Jie_Liu  
/// @version   
/// @date 2018/01/29

#ifndef _LDECISIONTREE_H_
#define _LDECISIONTREE_H_

#include "LMatrix.h"



typedef LMatrix<double> LDTCMatrix; ///< 决策数分类器矩阵

#define FEATURE_DISCRETE  0.0  ///< 特征值为离散分布
#define FEATURE_CONTINUUM 1.0  ///< 特征值为连续分布

class CDecisionTreeClassifier;

/// @brief 决策树分类器
class LDecisionTreeClassifier
{
public:
    /// @brief 构造函数
    LDecisionTreeClassifier();

    /// @brief 析构造函数
    ~LDecisionTreeClassifier();

    /// @brief 训练模型
    /// 每使用一次该方法, 则生成一个新的模型
    /// @param[in] xMatrix 样本矩阵, 每一行代表一个样本, 每一列代表样本的一个特征
    /// @param[in] nVector 样本特征分布向量(行向量), 每一列代表一个特征的分布, 值只能为FEATURE_DISCRETE和FEATURE_CONTINUUM
    /// @param[in] yVector 样本标签向量(列向量), 每一行代表一个样本, 标记值应为离散值, 不同的值代表不同的类别
    /// @return 成功返回true, 失败返回false(参数错误的情况下会返回失败)
    bool TrainModel(IN const LDTCMatrix& xMatrix, IN const LDTCMatrix& nVector, IN const LDTCMatrix& yVector);

    /// @brief 打印树, 用于调试
    void PrintTree();

private:
    CDecisionTreeClassifier* m_pClassifier; ///< 决策树分类器实现对象
};



#endif