

#include "../../../Src/LRegression.h"

#include <cstdio>
#include <cstdlib>


int main()
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

    system("pause");

    return 0;
}

