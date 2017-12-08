

#include "../../../Src/LRegression.h"

#include <cstdio>
#include <cstdlib>

void TestLinearRegression()
{
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
}

void TestLogisticRegression()
{
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
    LLogisticRegression logisticReg;

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
    printf("Predict Value: ");
    for (unsigned int i = 0; i < testY.RowLen; i++)
    {
        printf("%.5f  ", testY[i][0]);
    }
    printf("\n");
}

int main()
{
    
    TestLogisticRegression();

    system("pause");

    return 0;
}

