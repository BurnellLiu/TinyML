#include "../../../Src/LDecisionTree.h"

#include <cstdio>
#include <windows.h>

int main()
{
    /*
    "slashdot"    1.0
    "google"      2.0
    "digg"        3.0
    "kiwitobes"   4.0
    "(direct)"    5.0

    "USA"         1.0
    "France"      2.0
    "UK"          3.0
    "New Zealand" 4.0

    "yes"         1.0
    "no"          2.0
    */
    double trainX[64] = 
    {
        1.0, 1.0, 1.0, 18.0,
        2.0, 2.0, 1.0, 23.0,
        3.0, 1.0, 1.0, 24.0,
        4.0, 2.0, 1.0, 23.0,
        2.0, 3.0, 2.0, 21.0,
        5.0, 4.0, 2.0, 12.0,
        5.0, 3.0, 2.0, 21.0,
        2.0, 1.0, 2.0, 24.0,
        1.0, 2.0, 1.0, 19.0,
        3.0, 1.0, 2.0, 18.0,
        2.0, 3.0, 2.0, 18.0,
        4.0, 3.0, 2.0, 19.0,
        3.0, 4.0, 1.0, 12.0,
        1.0, 3.0, 2.0, 21.0,
        2.0, 3.0, 1.0, 18.0,
        4.0, 2.0, 1.0, 19.0
    };

    double trainN[4] = {FEATURE_DISCRETE, FEATURE_DISCRETE, FEATURE_DISCRETE, FEATURE_CONTINUUM};

    /*
    "None"       1.0
    "Premium"    2.0
    "Basic"      3.0
    */
    double trainY[16] =
    {
        1.0,
        2.0,
        3.0,
        3.0,
        2.0,
        1.0,
        3.0,
        2.0,
        1.0,
        1.0,
        1.0,
        1.0,
        3.0,
        1.0,
        3.0,
        3.0
    };
    LDTCMatrix xMatrix(16, 4, trainX);
    LDTCMatrix nVector(1, 4, trainN);
    LDTCMatrix yVector(16, 1, trainY);

    LDecisionTreeClassifier clf;
    clf.TrainModel(xMatrix, nVector, yVector);

    clf.PrintTree();

    system("pause");

    return 0;
}