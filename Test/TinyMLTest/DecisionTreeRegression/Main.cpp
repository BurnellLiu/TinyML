#include "../../../Src/LDecisionTree.h"
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

/// @brief 测试树回归
void TestDecisionTreeRegression()
{
    // 加载糖尿病数据集
    LCSVParser dataCSVParser(L"../../../DataSet/diabetes_data.csv");
    dataCSVParser.SetDelimiter(L' ');
    LDataMatrix xMatrix;
    dataCSVParser.LoadAllData(xMatrix);
    LCSVParser targetCSVParser(L"../../../DataSet/diabetes_target.csv");
    LDataMatrix yVector;
    targetCSVParser.LoadAllData(yVector);

    // 定义特征值分布向量
    double featureN[10] =
    {
        DT_FEATURE_CONTINUUM,
        DT_FEATURE_CONTINUUM,
        DT_FEATURE_CONTINUUM,
        DT_FEATURE_CONTINUUM,
        DT_FEATURE_CONTINUUM,
        DT_FEATURE_CONTINUUM,
        DT_FEATURE_CONTINUUM,
        DT_FEATURE_CONTINUUM,
        DT_FEATURE_CONTINUUM,
        DT_FEATURE_CONTINUUM
    };
    LDTMatrix nVector(1, 10, featureN);


    // 定义线性回归对象
    LDecisionTreeRegression dtReg;

    printf("Decision Tree Regression Model Train:\n");
    dtReg.TrainModel(xMatrix, nVector, yVector);
    double score = dtReg.Score(xMatrix, yVector);
    printf("Model Score: %.2f\n", score);
}


int main()
{
    TestDecisionTreeRegression();

    system("pause");

    return 0;
}
