/// @file LDecisionTree.h
/// @brief  决策树头文件
/// 
/// Detail:该文件声明了决策树分类器
/// @author Jie_Liu  
/// @version   
/// @date 2018/01/29

#ifndef _LDECISIONTREE_H_
#define _LDECISIONTREE_H_

#include "LMatrix.h"


typedef LMatrix<double> LDTCMatrix;     ///< 决策数分类器矩阵

#ifndef DTC_FEATURE_DISCRETE
#define DTC_FEATURE_DISCRETE  0.0       ///< 特征值为离散分布
#endif

#ifndef DTC_FEATURE_CONTINUUM
#define DTC_FEATURE_CONTINUUM 1.0       ///< 特征值为连续分布
#endif


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
    /// @param[in] yVector 样本标签向量(列向量), 每一行代表一个样本, 标签值应为离散值, 不同的值代表不同的类别
    /// @return 成功返回true, 失败返回false(参数错误的情况下会返回失败)
    bool TrainModel(IN const LDTCMatrix& xMatrix, IN const LDTCMatrix& nVector, IN const LDTCMatrix& yVector);

    /// @brief 使用训练好的模型预测数据
    /// @param[in] xMatrix 需要预测的样本矩阵
    /// @param[out] yVector 存储预测的标签向量(列向量)
    /// @return 成功返回true, 失败返回false(模型未训练或参数错误的情况下会返回失败)
    bool Predict(IN const LDTCMatrix& xMatrix, OUT LDTCMatrix& yVector) const;

    /// @brief 进行剪枝操作(即合并叶子节点)
    /// 剪枝可以防止模型过拟合
    /// 如果两个叶子节点合并后增加的熵小于指定的最小值, 则进行合并操作
    /// @param[in] minGain 最小增加值
    void Prune(IN double minGain);

    /// @brief 打印树, 用于调试
    void PrintTree();

private:
    CDecisionTreeClassifier* m_pClassifier; ///< 决策树分类器实现对象
};



#endif