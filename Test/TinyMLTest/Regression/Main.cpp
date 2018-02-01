

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

void TestLinearRegression()
{
    // 加载糖尿病数据集
    LCSVParser dataCSVParser(L"../../../DataSet/diabetes_data.csv");
    dataCSVParser.SetDelimiter(L' ');
    LDataMatrix xMatrix;
    dataCSVParser.LoadAllData(xMatrix);
    MatrixPrint(xMatrix);

    LCSVParser targetCSVParser(L"../../../DataSet/diabetes_target.csv");
    LDataMatrix yVector;
    targetCSVParser.LoadAllData(yVector);


    // 定义线性回归对象
    LLinearRegression linearReg;

    // 训练模型
    // 计算每一次训练后的损失值
    for (unsigned int i = 0; i < 500; i++)
    {
        linearReg.TrainModel(xMatrix, yVector, 0.003f);
        double loss = linearReg.LossValue(xMatrix, yVector);
        printf("Train Time: %u  ", i);
        printf("Loss Value: %.2f\n", loss);
    }

    
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

    // 使用训练集训练模型
    LLogisticRegression clf;
    for (int i = 0; i < 1000; i++)
    {
        clf.TrainModel(trainXMatrix, trainYVector, 0.1);
    }

    // 使用测试集计算模型得分
    double score = clf.Score(testXMatrix, testYVector);
    printf("Logistic Regression Model Score: %.2f\n", score);

}

/// @brief 测试SoftMax回归
void TestSoftmaxRegression()
{
    // 加载鸢尾花数据集
    LCSVParser csvParser(L"../../../DataSet/iris.csv");
    csvParser.SetSkipHeader(true);
    LDataMatrix dataMatrix;
    csvParser.LoadAllData(dataMatrix);

    // 数据进行缩放
    LUIntMatrix colVec(1, 4);
    colVec[0][0] = 0;
    colVec[0][1] = 1;
    colVec[0][2] = 2;
    colVec[0][3] = 3;
    LMinMaxScaler scaler(0.0, 1.0);
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
    // 将训练标签集因子化
    LRegressionMatrix trainYMatrix(trainYVector.RowLen, 3, REGRESSION_ZERO);
    for (unsigned int i = 0; i < trainYVector.RowLen; i++)
    {
        unsigned int label = (int)trainYVector[i][0];
        trainYMatrix[i][label] = REGRESSION_ONE;
    }

    // 将测试集拆分为测试样本集合和标签集
    LRegressionMatrix testXMatrix;
    LRegressionMatrix testYVector;
    testData.SubMatrix(0, testData.RowLen, 0, testData.ColumnLen - 1, testXMatrix);
    testData.SubMatrix(0, testData.RowLen, testData.ColumnLen - 1, 1, testYVector);
    // 将测试标签集因子化
    LRegressionMatrix testYMatrix(testYVector.RowLen, 3, REGRESSION_ZERO);
    for (unsigned int i = 0; i < testYVector.RowLen; i++)
    {
        unsigned int label = (int)testYVector[i][0];
        testYMatrix[i][label] = REGRESSION_ONE;
    }

    // 使用训练集训练模型
    LSoftmaxRegression clf;
    for (int i = 0; i < 150; i++)
    {
        clf.TrainModel(trainXMatrix, trainYMatrix, 0.1);
    }

    // 使用测试集计算模型得分
    double score = clf.Score(testXMatrix, testYMatrix);
    printf("Softmax Regression Model Score: %.2f\n", score);

}

int main()
{
    TestLinearRegression();
    // TestLogisticRegression();
    // TestSoftmaxRegression();

    system("pause");

    return 0;
}

