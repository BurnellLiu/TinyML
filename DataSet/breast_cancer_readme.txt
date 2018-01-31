威斯康辛州乳腺癌（诊断）数据集

实例数量:	
569

特征数量:	
30(数值型)

属性信息:
radius               半径（从中心到边缘上点的距离的平均值）
texture              纹理（灰度值的标准偏差）
perimeter            周长
area                 区域
smoothness           平滑度（半径长度的局部变化）
compactness          紧凑度（周长 ^ 2 /面积 - 1.0）
concavity            凹面（轮廓的凹部的严重性）
concave points       凹点（轮廓的凹部的数量）
symmetry             对称性
fractal dimension    分形维数（海岸线近似 - 1）

类别:
WDBC-Malignant 恶性
WDBC-Benign 良性

对每个图像计算这些属性的平均值，标准误差，以及“最差”（因为是肿瘤）或最大值（最大的前三个值的平均值）
得到30个特征。例如，字段 3 是平均半径，字段 13 是半径的标准误差，字段 23 是最差半径。

特征信息:		 
radius (mean)
texture (mean)
perimeter (mean)
area (mean)
smoothness (mean)
compactness (mean)
concavity (mean)
concave points (mean)
symmetry (mean)
fractal dimension (mean)
radius (standard error)
texture (standard error)
perimeter (standard error)
area (standard error)
smoothness (standard error)
compactness (standard error)
concavity (standard error)
concave points (standard error)
symmetry (standard error)
fractal dimension (standard error)
radius (worst)
texture (worst)
perimeter (worst)
area (worst)
smoothness (worst)
compactness (worst)
concavity (worst)
concave points (worst)
symmetry (worst)
fractal dimension (worst)


类别分布: 212 - 恶性, 357 - 良性
