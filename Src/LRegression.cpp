
#include "LRegression.h"

#include <cmath>


namespace Regression
{
    /// @brief 样本矩阵中增加常数项, 添加在最后一列, 值为1.0f
    /// @param[in] sampleMatrix 样本矩阵
    /// @param[in] newSampleMatrix 添加常数项后的样本矩阵
    void SamplexAddConstant(IN const LRegressionMatrix& sampleMatrix, OUT LRegressionMatrix& newSampleMatrix)
    {
        // 每个样本中最后一项增加常数项的特征值:1.0
        newSampleMatrix.Reset(sampleMatrix.RowLen, sampleMatrix.ColumnLen + 1);
        for (unsigned int row = 0; row < sampleMatrix.RowLen; row++)
        {
            for (unsigned int col = 0; col < sampleMatrix.ColumnLen; col++)
            {
                newSampleMatrix[row][col] = sampleMatrix[row][col];
            }
            newSampleMatrix[row][sampleMatrix.ColumnLen] = 1.0f; 
        }
    }
}

/// @brief 线性回归实现类
/// 线性函数为 h(x)  =  X * W
/// W为特征权重的列向量, X为特征的行向量
class CLinearRegression
{
public:
    CLinearRegression()
    {
        m_N = 0;
    }

    ~CLinearRegression()
    {

    }

    /// @brief 训练模型
    bool TrainModel(IN const LRegressionMatrix& xMatrix, IN const LRegressionMatrix& yVector, IN float alpha)
    {
        // 第一次训练, 记录下特征值数量, 并且初始化权重向量为0.0
        if (m_N == 0)
        {
            m_N = xMatrix.ColumnLen;
            m_wVector.Reset(m_N + 1, 1, 0.0f);
        }


        // 检查参数
        if (m_N < 1)
            return false;
        if (xMatrix.RowLen < 1)
            return false;
        if (xMatrix.ColumnLen != m_N)
            return false;
        if (yVector.ColumnLen != 1)
            return false;
        if (yVector.RowLen != xMatrix.RowLen)
            return false;
        if (alpha <= 0.0f)
            return false;

        LRegressionMatrix X;
        Regression::SamplexAddConstant(xMatrix, X);

        const LRegressionMatrix& Y = yVector;
        LRegressionMatrix& W = m_wVector;

        LRegressionMatrix XT = X.T();

        LRegressionMatrix XW;
        LRegressionMatrix DW;

        /*
        h(x) = X * W
        wj = wj - α * ∑((h(x)-y) * xj)
        */
        LRegressionMatrix::MUL(X, W, XW);
        LRegressionMatrix::SUB(XW, Y, XW);
        LRegressionMatrix::MUL(XT, XW, DW);
        LRegressionMatrix::SCALARMUL(DW, -1.0f * alpha, DW);
        LRegressionMatrix::ADD(W, DW, W);

        return true;
    }

    /// @brief 使用训练好的模型预测数据
    bool Predict(IN const LRegressionMatrix& xMatrix, OUT LRegressionMatrix& yVector) const
    {
        // 检查参数
        // 特征值小于1说明模型还没有训练
        if (m_N < 1)
            return false;

        if (xMatrix.RowLen < 1)
            return false;
        if (xMatrix.ColumnLen != m_N)
            return false;

        LRegressionMatrix X;
        Regression::SamplexAddConstant(xMatrix, X);

        LRegressionMatrix::MUL(X, m_wVector, yVector);

        return true;
    }

    /// @brief 计算损失值
    float LossValue(IN const LRegressionMatrix& xMatrix, IN const LRegressionMatrix& yVector) const
    {
        // 检查参数
        // 特征值小于1说明模型还没有训练
        if (m_N < 1)
            return -1.0f;
        if (xMatrix.RowLen < 1)
            return -1.0f;
        if (xMatrix.RowLen != yVector.RowLen)
            return -1.0f;
        if (xMatrix.ColumnLen != m_N)
            return -1.0f;
        if (yVector.ColumnLen != 1)
            return -1.0f;

        LRegressionMatrix X;
        LRegressionMatrix Y;
        Regression::SamplexAddConstant(xMatrix, X);

        LRegressionMatrix::MUL(X, m_wVector, Y);
        LRegressionMatrix::SUB(Y, yVector, Y);
        LRegressionMatrix YT;
        LRegressionMatrix::T(Y, YT);

        LRegressionMatrix LOSS;
        LRegressionMatrix::MUL(YT, Y, LOSS);

        return LOSS[0][0]/2.0f;
    }

private:
    unsigned int m_N; ///< 样本特征值个数
    LRegressionMatrix m_wVector; ///<权重矩阵(列向量)
};

LLinearRegression::LLinearRegression()
    : m_pLinearRegression(0)
{
    m_pLinearRegression = new CLinearRegression();
}

LLinearRegression::~LLinearRegression()
{
    if (m_pLinearRegression != 0)
    {
        delete m_pLinearRegression;
        m_pLinearRegression = 0;
    }
}

bool LLinearRegression::TrainModel(IN const LRegressionMatrix& xMatrix, IN const LRegressionMatrix& yVector, IN float alpha)
{
    return m_pLinearRegression->TrainModel(xMatrix, yVector, alpha);
}

bool LLinearRegression::Predict(IN const LRegressionMatrix& xMatrix, OUT LRegressionMatrix& yVector) const
{
    return m_pLinearRegression->Predict(xMatrix, yVector);
}

float LLinearRegression::LossValue(IN const LRegressionMatrix& xMatrix, IN const LRegressionMatrix& yVector) const
{
    return m_pLinearRegression->LossValue(xMatrix, yVector);
}

/// @brief 逻辑回归(分类)实现类
/// 逻辑函数为 h(x)  =  1/(1 + e^(X * W)) 
/// W为特征权重的列向量, X为特征的行向量
/// 原始问题中的目标向量中的值只能为0.0f或1.0f
/// P(1) = h(x), P(0) = 1-h(x)
class CLogisticRegression
{
public:
    /// @brief 构造函数
    CLogisticRegression()
    {
        m_N = 0;
        
    }

    /// @brief 析构函数
    ~CLogisticRegression()
    {

    }

    /// @brief 训练模型
    bool TrainModel(IN const LRegressionMatrix& xMatrix, IN const LRegressionMatrix& yVector, IN float alpha)
    {
        // 第一次训练, 记录下特征值数量, 并且初始化权重向量为0.0
        if (m_N == 0)
        {
            m_N = xMatrix.ColumnLen;
            m_wVector.Reset(m_N + 1, 1, 0.0f);
        }

        // 检查参数
        if (m_N < 1)
            return false;

        if (xMatrix.RowLen < 1)
            return false;
        if (xMatrix.ColumnLen != m_N)
            return false;

        if (yVector.ColumnLen != 1)
            return false;
        if (yVector.RowLen != xMatrix.RowLen)
            return false;

        if (alpha <= 0.0f)
            return false;

        for (unsigned int i = 0; i < yVector.RowLen; i++)
        {
            if (yVector[i][0] != REGRESSION_ONE &&
                yVector[i][0] != REGRESSION_ZERO)
                return false;
        }

        LRegressionMatrix X;
        Regression::SamplexAddConstant(xMatrix, X);

        const LRegressionMatrix& Y = yVector;

        LRegressionMatrix& W = m_wVector;
        LRegressionMatrix XT = X.T();

        /*
        如果h(x)  =  1/(1 + e^(X * W)) 则
        wj = wj - α * ∑((y - h(x)) * xj)
        如果h(x)  =  1/(1 + e^(-X * W)) 则
        wj = wj + α * ∑((y - h(x)) * xj)
        */

        LRegressionMatrix XW(X.RowLen, 1);
        LRegressionMatrix DW;

        LRegressionMatrix::MUL(X, W, XW);
        for (unsigned int m = 0; m < XW.RowLen; m++)
        {
            this->Sigmoid(XW[m][0], XW[m][0]);
        }

        LRegressionMatrix::SUB(Y, XW, XW);
        LRegressionMatrix::MUL(XT, XW, DW);

        LRegressionMatrix::SCALARMUL(DW, -1.0f * alpha, DW);
        LRegressionMatrix::ADD(W, DW, W);

        return true;
    }

    /// @brief 使用训练好的模型预测数据
    bool Predict(IN const LRegressionMatrix& xMatrix, OUT LRegressionMatrix& yVector) const
    {
        // 检查参数
        if (m_N < 1)
            return false;

        if (xMatrix.RowLen < 1)
            return false;
        if (xMatrix.ColumnLen != m_N)
            return false;

        LRegressionMatrix X;
        Regression::SamplexAddConstant(xMatrix, X);

        yVector.Reset(X.RowLen, 1, 0.0f);
        LRegressionMatrix::MUL(X, m_wVector, yVector);

        for (unsigned int m = 0; m < yVector.RowLen; m++)
        {
            this->Sigmoid(yVector[m][0], yVector[m][0]);
        }

        return true;

    }

    /// @brief 计算似然值, 似然值为0.0~1.0之间的数, 似然值值越大模型越好
    float LikelihoodValue(IN const LRegressionMatrix& xMatrix, IN const LRegressionMatrix& yVector) const
    {
        // 检查参数
        // 特征值小于1说明模型还没有训练
        if (m_N < 1)
            return -1.0f;
        if (xMatrix.RowLen < 1)
            return -1.0f;
        if (xMatrix.RowLen != yVector.RowLen)
            return -1.0f;
        if (xMatrix.ColumnLen != m_N)
            return -1.0f;
        if (yVector.ColumnLen != 1)
            return -1.0f;

        LRegressionMatrix predictY;
        this->Predict(xMatrix, predictY);

        float likelihood = 1.0f;
        for (unsigned int i = 0; i < yVector.RowLen; i++)
        {
            if (yVector[i][0] == REGRESSION_ONE)
                likelihood *= predictY[i][0];
            else if (yVector[i][0] == REGRESSION_ZERO)
                likelihood *= (1.0f - predictY[i][0]);
            else
                return -1.0f;
        }

        return likelihood;
    }

private:
    /// @brief S型函数
    /// @param[in] input 输入值
    /// @param[out] output 存储输出值
    void Sigmoid(float input, float& output) const
    {
        output = 1.0f/(1.0f + exp(input));
    }

private:
    unsigned int m_N; ///< 样本特征值个数
    LRegressionMatrix m_wVector; ///<权重矩阵(列向量)
};

LLogisticRegression::LLogisticRegression()
    : m_pLogisticRegression(0)
{
    m_pLogisticRegression = new CLogisticRegression(); 
}

LLogisticRegression::~LLogisticRegression()
{
    if (m_pLogisticRegression != 0)
    {
        delete m_pLogisticRegression;
        m_pLogisticRegression = 0;
    }
}

bool LLogisticRegression::TrainModel(IN const LRegressionMatrix& xMatrix, IN const LRegressionMatrix& yVector, IN float alpha)
{
    return m_pLogisticRegression->TrainModel(xMatrix, yVector, alpha);
}

bool LLogisticRegression::Predict(IN const LRegressionMatrix& xMatrix, OUT LRegressionMatrix& yVector) const
{
    return m_pLogisticRegression->Predict(xMatrix, yVector);
}

float LLogisticRegression::LikelihoodValue(IN const LRegressionMatrix& xMatrix, IN const LRegressionMatrix& yVector) const
{
    return m_pLogisticRegression->LikelihoodValue(xMatrix, yVector);
}


class CSoftmaxRegression
{
public:
    CSoftmaxRegression(IN unsigned int m, IN unsigned int n, IN unsigned int k)
    {
        m_M = m;
        m_N = n;
        m_K = k;

        m_wMatrix.Reset(n + 1, k, 0.0f);
    }

    ~CSoftmaxRegression()
    {

    }

    /// @brief 训练模型
    bool TrainModel(IN const LRegressionMatrix& xMatrix, IN const LRegressionMatrix& yMatrix, IN float alpha)
    {
        // 检查参数
        if (m_M < 2 || m_N < 1 || m_K < 2)
            return false;

        if (xMatrix.RowLen < 1)
            return false;
        if (xMatrix.ColumnLen != m_N)
            return false;
        if (yMatrix.RowLen != xMatrix.RowLen)
            return false;
        if (yMatrix.ColumnLen != m_K)
            return false;

        if (alpha <= 0.0f)
            return false;

        // 增加常数项后的样本矩阵
        LRegressionMatrix X;
        Regression::SamplexAddConstant(xMatrix, X);

        // 权重矩阵
        LRegressionMatrix& W = m_wMatrix;

        // 概率矩阵
        LRegressionMatrix P(X.RowLen, m_K, 0.0f);

        // 计算概率矩阵
        this->SampleProbK(X, W, P);

        LRegressionMatrix::SUB(yMatrix, P, P);

        // 权重向量(列向量)
        LRegressionMatrix dwVec(m_N + 1, 1, 0.0f);
        for (unsigned int k = 0; k < m_K; k++)
        {
            dwVec.Reset(m_N + 1, 1, 0.0f);
            for (unsigned int row = 0; row < X.RowLen; row++)
            {
                for (unsigned int col = 0; col < X.ColumnLen; col++)
                {
                    dwVec[col][0] += X[row][col] * P[row][k];
                }
            }

            LRegressionMatrix::SCALARDIV(dwVec, (float)m_M, dwVec);

            LRegressionMatrix::SCALARMUL(dwVec, alpha, dwVec);

            for (unsigned int row = 0; row < m_wMatrix.RowLen; row++)
            {
                m_wMatrix[row][k] += dwVec[row][0];
            }
        }

        return true;
    }

    /// @brief 使用训练好的模型预测数据
    bool Predict(IN const LRegressionMatrix& xMatrix, OUT LRegressionMatrix& yMatrix) const
    {
        // 检查参数
        if (m_M < 2 || m_N < 1 || m_K < 2)
            return false;

        if (xMatrix.RowLen < 1)
            return false;
        if (xMatrix.ColumnLen != m_N)
            return false;

        yMatrix.Reset(xMatrix.RowLen, m_K, 0.0f);

        LRegressionMatrix X;
        Regression::SamplexAddConstant(xMatrix, X);

        this->SampleProbK(X, m_wMatrix, yMatrix);

        return true;
    }

private:
    /// @brief 计算样本属于K个分类的各个概率
    /// @param[in] sampleMatrix 样本矩阵, m * n
    /// @param[in] weightMatrix 权重矩阵, n * k, 每一列为一个分类权重
    /// @param[out] probMatrix 概率矩阵, 存储每个样本属于不同分类的概率
    /// @return 成功返回true, 失败返回false(参数错误的情况下会返回失败)
    void SampleProbK(
        IN const LRegressionMatrix& sampleMatrix, 
        IN const LRegressionMatrix& weightMatrix, 
        OUT LRegressionMatrix& probMatrix) const
    {
        LRegressionMatrix::MUL(sampleMatrix, weightMatrix, probMatrix);

        for (unsigned int row = 0; row < probMatrix.RowLen; row++)
        {
            for (unsigned int col = 0; col < probMatrix.ColumnLen; col++)
            {
                probMatrix[row][col] = exp(probMatrix[row][col]);
            }
        }

        for (unsigned int row = 0; row < probMatrix.RowLen; row++)
        {
            float sum = 0.0f;
            for (unsigned int col = 0; col < probMatrix.ColumnLen; col++)
            {
                sum += probMatrix[row][col];
            }

            for (unsigned int col = 0; col < probMatrix.ColumnLen; col++)
            {
                probMatrix[row][col] = probMatrix[row][col]/sum;
            }
        }
    }
private:
    unsigned int m_M; ///< 样本总个数
    unsigned int m_N; ///< 样本特征值个数
    unsigned int m_K; ///< 样本类别个数

    LRegressionMatrix m_wMatrix; ///<权重矩阵, 每一列则为一个分类的权重向量
};

LSoftmaxRegression::LSoftmaxRegression(IN unsigned int m, IN unsigned int n, IN unsigned int k)
    :m_pSoftmaxRegression(0)
{
    m_pSoftmaxRegression = new CSoftmaxRegression(m, n ,k);
}

LSoftmaxRegression::~LSoftmaxRegression()
{
    if (m_pSoftmaxRegression != 0)
    {
        delete m_pSoftmaxRegression;
        m_pSoftmaxRegression = 0;
    }
}


bool LSoftmaxRegression::TrainModel(IN const LRegressionMatrix& xMatrix, IN const LRegressionMatrix& yMatrix, IN float alpha)
{
    return m_pSoftmaxRegression->TrainModel(xMatrix, yMatrix, alpha);
}

bool LSoftmaxRegression::Predict(IN const LRegressionMatrix& xMatrix, OUT LRegressionMatrix& yMatrix) const
{
    return m_pSoftmaxRegression->Predict(xMatrix, yMatrix);
}
