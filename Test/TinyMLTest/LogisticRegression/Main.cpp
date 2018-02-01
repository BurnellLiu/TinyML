
#include "../../../Src/LRegression.h"
#include "../../../Src/LCSVIo.h"
#include "../../../Src/LPreProcess.h"

#include <cstdio>
#include <cstdlib>
#include <ctime>

/// @brief 打印矩阵
void MatrixPrint(IN const LDataMatrix& dataMatrix)
{
    printf("Matrix Row: %u  Col: %u\n", dataMatrix.RowLen, dataMatrix.ColumnLen);
    for (unsigned int i = 0; i < dataMatrix.RowLen; i++)
    {
        for (unsigned int j = 0; j < dataMatrix.ColumnLen; j++)
        {
            printf("%.2f  ", dataMatrix[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

/// @brief 测试逻辑回归
void TestLogisticRegression()
{
    // 加载乳腺癌数据集
    LCSVParser csvParser(L"../../../DataSet/breast_cancer.csv");
    csvParser.SetSkipHeader(true);
    LDataMatrix dataMatrix;
    csvParser.LoadAllData(dataMatrix);

    // 数据进行缩放
    LUIntMatrix colVec(1, 30);
    for (unsigned int col = 0; col > colVec.ColumnLen; col++)
    {
        colVec[0][col] = col;
    }
    LMinMaxScaler scaler(0, 1.0);
    scaler.FitTransform(colVec, dataMatrix);

    // 打乱数据集
    DoubleMatrixShuffle(0, dataMatrix);

    // 将数据集拆分为训练集和测试集, 测试集占总集合的20%
    unsigned int testSize = (unsigned int)(dataMatrix.RowLen * 0.2);
    LDataMatrix trainData;
    LDataMatrix testData;
    dataMatrix.SubMatrix(0, testSize, 0, dataMatrix.ColumnLen, testData);
    dataMatrix.SubMatrix(testSize, dataMatrix.RowLen - testSize, 0, dataMatrix.ColumnLen, trainData);

    // 将训练集拆分为训练样本集合和标签集
    LRegressionMatrix trainXMatrix;
    LRegressionMatrix trainYVector;
    trainData.SubMatrix(0, trainData.RowLen, 0, trainData.ColumnLen - 1, trainXMatrix);
    trainData.SubMatrix(0, trainData.RowLen, trainData.ColumnLen - 1, 1, trainYVector);

    // 将测试集拆分为测试样本集合和标签集
    LRegressionMatrix testXMatrix;
    LRegressionMatrix testYVector;
    testData.SubMatrix(0, testData.RowLen, 0, testData.ColumnLen - 1, testXMatrix);
    testData.SubMatrix(0, testData.RowLen, testData.ColumnLen - 1, 1, testYVector);

    printf("Logistic Regression Model Train:\n");
    // 使用训练集训练模型
    LLogisticRegression clf;
    for (int i = 0; i < 1000; i++)
    {
        clf.TrainModel(trainXMatrix, trainYVector, 0.1);

        // 使用测试集计算模型得分
        double score = clf.Score(testXMatrix, testYVector);
        printf("Time: %u Score: %.4f\n", i, score);
    }

}

int main()
{
    TestLogisticRegression();

    system("pause");

    return 0;
}