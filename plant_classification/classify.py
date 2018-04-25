# coding=utf-8
from __future__ import print_function
import rgbhistogram
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
import numpy as np
import argparse
import glob
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True, help="path to the image dataset")
ap.add_argument("-m", "--masks", required=True, help="path to the image masks")
args = vars(ap.parse_args())

imagePaths = sorted(glob.glob(args["images"] + "/*.png"))
maskPaths = sorted(glob.glob(args["masks"] + "/*.png"))

data = []
target = []

#构造描述器，用直方图描述作为特征
desc = rgbhistogram.RGBHistogram([8, 8, 8])

"""
a = [1,2,3]
b = [4,5,6]
并行遍历
zip(a,b)
[(1, 4), (2, 5), (3, 6)]
构造特征列表和对应的标注列表
"""
for (imagePath, maskPath) in zip(imagePaths, maskPaths):
    image = cv2.imread(imagePath)
    mask = cv2.imread(maskPath)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    #获取特征值
    features = desc.describs(image, mask)

    #特征列表
    data.append(features)
    #标注列表
    target.append(imagePath.split("_")[-2])

#去重
targetNames = np.unique(target)

"""
#LabelEncoder 是对不连续的数字或者文本进行编号

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

le.fit([1,5,67,100])

le.transform([1,1,100,67,5])

输出： array([0,0,3,2,1])
"""
le = LabelEncoder()
target = le.fit_transform(target)
print("target: {}".format(target))
"""
sklearn.model_selection.train_test_split随机划分训练集和测试集
官网文档：http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html#sklearn.model_selection.train_test_split
一般形式：
train_test_split是交叉验证中常用的函数，功能是从样本中随机的按比例选取train data和testdata，形式为：
X_train,X_test, y_train, y_test =
cross_validation.train_test_split(train_data,train_target,test_size=0.4, random_state=0)
参数解释：
train_data：所要划分的样本特征集
train_target：所要划分的样本结果
test_size：样本占比，如果是整数的话就是样本的数量
random_state：是随机数的种子。
随机数种子：其实就是该组随机数的编号，在需要重复试验的时候，保证得到一组一样的随机数。比如你每次都填1，其他参数一样的情况下你得到的随机数组是一样的。但填0或不填，每次都会不一样。
随机数的产生取决于种子，随机数和种子之间的关系遵从以下两个规则：
种子不同，产生不同的随机数；种子相同，即使实例不同也产生相同的随机数。
"""
(trainData, testData, trainTarget, testTarget) = train_test_split(data, target, test_size=0.3, random_state=42)

"""
随机森林分类器
n_estimators：森林里（决策）树的数目。
random_state：是随机数的种子。
随机数种子：其实就是该组随机数的编号，在需要重复试验的时候，保证得到一组一样的随机数。比如你每次都填1，其他参数一样的情况下你得到的随机数组是一样的。但填0或不填，每次都会不一样。
随机数的产生取决于种子，随机数和种子之间的关系遵从以下两个规则：
种子不同，产生不同的随机数；种子相同，即使实例不同也产生相同的随机数。
"""
model = RandomForestClassifier(n_estimators=25, random_state=84)
model.fit(trainData, trainTarget)

"""
classification_report简介

sklearn中的classification_report函数用于显示主要分类指标的文本报告．在报告中显示每个类的精确度，召回率，F1值等信息。 
主要参数: 
y_true：1维数组，或标签指示器数组/稀疏矩阵，目标值。 
y_pred：1维数组，或标签指示器数组/稀疏矩阵，分类器返回的估计值。 
labels：array，shape = [n_labels]，报表中包含的标签索引的可选列表。 
target_names：字符串列表，与标签匹配的可选显示名称（相同顺序）。 
sample_weight：类似于shape = [n_samples]的数组，可选项，样本权重。 
digits：int，输出浮点值的位数．

   precision    recall  f1-score   support

     crocus       0.92      1.00      0.96        12
      daisy       0.88      0.93      0.90        15
      pansy       1.00      0.85      0.92        20
  sunflower       0.96      1.00      0.98        24

avg / total       0.95      0.94      0.94        71

其中列表左边的一列为分类的标签名，右边support列为每个标签的出现次数．avg / total行为各列的均值（support列为总和）． 
precision recall f1-score三列分别为各个类别的精确度/召回率及 F1
值．
"""
print(classification_report(testTarget, model.predict(testData), target_names=targetNames))

"""
随机显示10幅花的照片，并辨别。

np.random.choice: 可以从一个int数字或1维array里随机选取内容，并将选取结果放入n维array中返回。
numpy.random.choice(a, size=None, replace=True, p=None)

np.arange: 创建等差数组.

model.predict([features]) 返回 [2], 
le.inverse_transform([2]) 返回 ['pansy'] 对之前的fit_tansform 编号进行反查，返回原始target对应文本
"""
for i in np.random.choice(np.arange(0, len(imagePaths)), 10):
    imagePath = imagePaths[i]
    maskPath = imagePaths[i]

    image = cv2.imread(imagePath)
    mask = cv2.imread(maskPath)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    features = desc.describs(image, mask)

    flower = le.inverse_transform(model.predict([features]))[0]
    print(imagePath)
    print("I think this flower is a {}".format(flower.upper()))
    cv2.imshow("image", image)
    cv2.waitKey(0)
