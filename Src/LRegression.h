/// @file LRegression.h
/// @brief 本文件中声明了一些回归算法
/// 线性回归, 逻辑回归, Softmax回归
/// Detail:
/// 线性回归:
/// 优点: 结果易于理解, 计算上不复杂
/// 缺点: 对非线性数据拟合不好
/// 线性回归的改进算法有局部加权回归
/// 
/// 逻辑回归:
/// 优点: 计算代价不高, 易于理解和实现
/// 缺点: 容易欠拟合, 分类精度可能不高
/// 
/// Softmax回归:
/// 该回归是逻辑回归在多分类问题上的推广
///
/// 梯度下降算法: 每次训练使用所有样本集, 如果样本集很大, 则导致内存开销大, 并且训练耗时长, 优点是收敛快
/// 随机梯度下降算法: 每次训练使用样本集中的一个样本, 缺点是收敛慢
/// 批量梯度下降算法: 综合以上两种
/// @author Burnell_Liu Email:burnell_liu@outlook.com
/// @version   
/// @date 2017/09/29

/*  使用线性回归示例代码如下:

// 定义训练样本
float trainX[4] =
{
    2.0f,
    4.0f,
    6.0f,
    8.0f
};
LRegressionMatrix xMatrix(4, 1, trainX);

// 定义训练样本输出
float trainY[4] =
{
    1.0f,
    2.0f,
    3.0f,
    4.0f
};
LRegressionMatrix yMatrix(4, 1, trainY);


// 定义线性回归对象
LLinearRegression linearReg;

// 训练模型
// 计算每一次训练后的损失值
for (unsigned int i = 0; i < 10; i++)
{
    linearReg.TrainModel(xMatrix, yMatrix, 0.01f);
    float loss = linearReg.LossValue(xMatrix, yMatrix);
    printf("Train Time: %u  ", i);
    printf("Loss Value: %f\n", loss);
}

// 进行预测
LRegressionMatrix yVector;
linearReg.Predict(xMatrix, yVector);

printf("Predict Value: ");
for (unsigned int i = 0; i < yVector.RowLen; i++)
{
    printf("%.5f  ", yVector[i][0]);
}
printf("\n");

*/


/*  使用逻辑回归示例代码如下:

// 训练样本数据
float trainSample[36] =
{
    1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f,
    1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f,
    1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 1.0f, 1.0f, 1.0f, 0.0f,
    0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 1.0f, 1.0f, 1.0f, 0.0f
};
LRegressionMatrix X(6, 6, trainSample);

// 训练样本标签数据
float trainLabel[12] =
{
    1.0f,
    1.0f,
    1.0f,
    0.0f,
    0.0f,
    0.0f
};
LRegressionMatrix Y(6, 1, trainLabel);

// 定义逻辑回归
LLogisticRegression logisticReg(6, 6);

// 训练500次
for (unsigned int i = 0; i < 500; i++)
{
    logisticReg.TrainModel(X, Y, 0.1f);
}

// 测试样本
float testSample[12] =
{
    1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 1.0f, 1.0f, 1.0f, 0.0f
};
LRegressionMatrix testX(2, 6, testSample);

// 对测试样本进行预测
LRegressionMatrix testY;
logisticReg.Predict(testX, testY);
*/


/*  使用Softmax回归示例代码如下:

// 训练样本数据
float trainSample[36] = 
{
    1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f,
    1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f,
    1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 1.0f, 1.0f, 1.0f, 0.0f,
    0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 1.0f, 1.0f, 1.0f, 0.0f
};
LRegressionMatrix X(6, 6, trainSample);

// 训练样本标签数据
float trainLabel[12] = 
{
    1.0f, 0.0f,
    1.0f, 0.0f,
    1.0f, 0.0f,
    0.0f, 1.0f,
    0.0f, 1.0f,
    0.0f, 1.0f
};
LRegressionMatrix Y(6, 2, trainLabel);

// 定义Softmax回归
LSoftmaxRegression softmaxReg(6, 6, 2);

// 训练500次
for (unsigned int i = 0; i < 500; i++)
{
    softmaxReg.TrainModel(X, Y, 0.1f);   
}

// 测试样本
float testSample[12] = 
{
    1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 1.0f, 1.0f, 1.0f, 0.0f
};
LRegressionMatrix testX(2, 6, testSample);

// 对测试样本进行预测
LRegressionMatrix testY;
softmaxReg.Predict(testX, testY);
*/

#ifndef _LREGRESSION_H_
#define _LREGRESSION_H_


#include "LMatrix.h"

typedef LMatrix<float> LRegressionMatrix;

class CLinearRegression;

/// @brief 线性回归类
class LLinearRegression
{
public:
    /// @brief 构造函数
    LLinearRegression();

    /// @brief 析构函数
    ~LLinearRegression();

    /// @brief 训练模型
    /// 如果一次训练的样本数量为1, 则为随机梯度下降
    /// 如果一次训练的样本数量为M(样本总数), 则为梯度下降
    /// 如果一次训练的样本数量为m(1 < m < M), 则为批量梯度下降
    /// @param[in] xMatrix 样本矩阵, 每一行代表一个样本, 每一列代表样本的一个特征
    /// @param[in] yVector(列向量) 样本输出向量, 每一行代表一个样本
    /// @param[in] alpha 学习速度, 该值必须大于0.0f
    /// @return 成功返回true, 失败返回false(参数错误的情况下会返回失败)
    bool TrainModel(IN const LRegressionMatrix& xMatrix, IN const LRegressionMatrix& yVector, IN float alpha);

    /// @brief 使用训练好的模型预测数据
    /// @param[in] xMatrix 需要预测的样本矩阵
    /// @param[out] yVector 存储预测的结果向量(列向量)
    /// @return 成功返回true, 失败返回false(模型未训练或参数错误的情况下会返回失败)
    bool Predict(IN const LRegressionMatrix& xMatrix, OUT LRegressionMatrix& yVector) const;

    /// @brief 计算损失值, 损失值为大于等于0的数, 损失值越小模型越好
    /// @param[in] xMatrix 样本矩阵, 每一行代表一个样本, 每一列代表样本的一个特征
    /// @param[in] yVector(列向量) 样本输出向量, 每一行代表一个样本
    /// @return 成功返回损失值, 失败返回-1.0f(参数错误的情况下会返回失败)
    float LossValue(IN const LRegressionMatrix& xMatrix, IN const LRegressionMatrix& yVector) const;

private:
    CLinearRegression* m_pLinearRegression; ///< 线性回归实现对象
};

/// @brief 回归中的ZERO分类
#ifndef REGRESSION_ZERO
#define REGRESSION_ZERO 0.0f
#endif

/// @brief 回归中的ONE分类
#ifndef REGRESSION_ONE
#define REGRESSION_ONE 1.0f
#endif

class CLogisticRegression;

/// @brief 逻辑回归(分类)
class LLogisticRegression
{
public:
    /// @brief 构造函数
    LLogisticRegression();

    /// @brief 析构函数
    ~LLogisticRegression();

    /// @brief 训练模型
    /// 如果一次训练的样本数量为1, 则为随机梯度下降
    /// 如果一次训练的样本数量为M(样本总数), 则为梯度下降
    /// 如果一次训练的样本数量为m(1 < m < M), 则为批量梯度下降
    /// @param[in] xMatrix 样本矩阵, 每一行代表一个样本, 每一列代表样本的一个特征
    /// @param[in] yVector(列向量) 样本标记向量, 每一行代表一个样本, 值只能为REGRESSION_ONE或REGRESSION_ZERO 
    /// @param[in] alpha 学习速度, 该值必须大于0.0f
    /// @return 成功返回true, 失败返回false(参数错误的情况下会返回失败)
    bool TrainModel(IN const LRegressionMatrix& xMatrix, IN const LRegressionMatrix& yVector, IN float alpha);

    /// @brief 使用训练好的模型预测数据
    /// @param[in] xMatrix 需要预测的样本矩阵
    /// @param[out] yVector 存储预测的结果向量(列向量), 值为REGRESSION_ONE标记的概率
    /// @return 成功返回true, 失败返回false(模型未训练或参数错误的情况下会返回失败)
    bool Predict(IN const LRegressionMatrix& xMatrix, OUT LRegressionMatrix& yVector) const;

    /// @brief 计算似然值, 似然值为0.0~1.0之间的数, 似然值值越大模型越好
    /// @param[in] xMatrix 样本矩阵, 每一行代表一个样本, 每一列代表样本的一个特征
    /// @param[in] yVector(列向量) 样本输出向量, 每一行代表一个样本, 值只能为REGRESSION_ONE或REGRESSION_ZERO
    /// @return 成功返回似然值, 失败返回-1.0f(参数错误的情况下会返回失败)
    float LikelihoodValue(IN const LRegressionMatrix& xMatrix, IN const LRegressionMatrix& yVector);

private:
    CLogisticRegression* m_pLogisticRegression; ///< 逻辑回归实现类
};


class CSoftmaxRegression;

/// @brief Softmax回归(多分类)
class LSoftmaxRegression
{
public:
    /// @brief 构造函数
    /// @param[in] m 训练样本总个数, 不能小于2
    /// @param[in] n 样本特征值个数, 不能小于1
    /// @param[in] k 样本类别个数, 不能小于2
    LSoftmaxRegression(IN unsigned int m, IN unsigned int n, IN unsigned int k);

    /// @brief 析构函数
    ~LSoftmaxRegression();

    /// @brief 训练模型
    /// 如果一次训练的样本数量为1, 则为随机梯度下降
    /// 如果一次训练的样本数量为M(样本总数), 则为梯度下降
    /// 如果一次训练的样本数量为m(1 < m < M), 则为批量梯度下降
    /// @param[in] xMatrix 样本矩阵, 每一行代表一个样本, 每一列代表样本的一个特征
    /// @param[in] yMatrix 类标记矩阵, 每一行代表一个样本, 每一列代表样本的一个类别, 
    /// 如果样本属于该类别则标记为REGRESSION_ONE, 不属于则标记为REGRESSION_ZERO
    /// @param[in] alpha 学习速度, 该值必须大于0.0f
    /// @return 成功返回true, 失败返回false(参数错误的情况下会返回失败)
    bool TrainModel(IN const LRegressionMatrix& xMatrix, IN const LRegressionMatrix& yMatrix, IN float alpha);

    /// @brief 使用训练好的模型预测数据
    /// @param[in] xMatrix 需要预测的样本矩阵
    /// @param[out] yMatrix 存储预测的结果矩阵, 每一行代表一个样本, 每一列代表在该类别下的概率
    /// @return 成功返回true, 失败返回false(参数错误的情况下会返回失败)
    bool Predict(IN const LRegressionMatrix& xMatrix, OUT LRegressionMatrix& yMatrix) const;

private:
    CSoftmaxRegression* m_pSoftmaxRegression; ///< Softmax回归实现对象
};

#endif