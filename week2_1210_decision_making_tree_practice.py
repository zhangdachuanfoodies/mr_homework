
# coding: utf-8

# ### Decision-making Tree

# In[12]:

from numpy import *
from math import log
import operator

# 计算香农熵  [计算划分前香农熵，和划分后的每个集合的香农熵的和，相减求得信息增益]
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    ShannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]/numEntries)
        ShannonEnt -= prob * log(prob, 2)
    return ShannonEnt


# In[14]:

# 测试香农熵计算
def createDataSet():
    dataSet = [[1, 1, 'yes'],
              [1, 1, 'yes'],
              [1, 0, 'no'],
              [0, 1, 'no'],
              [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels

dataSet, labels = createDataSet()
calcShannonEnt(dataSet)


# In[18]:

# 增加一个分类
def createDataSet2():
    dataSet = [[1, 1, 'maybe'],
              [1, 1, 'yes'],
              [1, 0, 'no'],
              [0, 1, 'no'],
              [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels

dataSet, labels = createDataSet2()
calcShannonEnt(dataSet)


# In[26]:

# 划分数据集
def splitDataSet(dataSet, axis, value):
    """
    input:待划分数据集，划分数据集的特征，需要返回的特征值
    output:包含该特征和值，去掉该特征后的数据集
    """
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[: axis]   # 这个特征前的特征
#             print(reducedFeatVec)
            reducedFeatVec.extend(featVec[axis+1: ])  # 这个特征后的特征
            retDataSet.append(reducedFeatVec)
    return retDataSet



# In[51]:

# 测试效果
dataSet, labels = createDataSet()
splitDataSet(dataSet, 1, 1)  # 索引为1的特征被除去了


# In[44]:

# 选择最好的数据划分方式
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0; bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]  # 这一整列，这一列特征
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:  
            # 对第i个特征的每个唯一属性值划分一次数据集，计算数据集的新熵值
            # 会划分为len(value)个数据集
            subDataSet = splitDataSet(dataSet, i, value)  
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
#         print(infoGain)
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature
        


# In[36]:

chooseBestFeatureToSplit(dataSet)


# In[37]:

# 多数表决
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCont = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]
    


# In[54]:

# 创建树
def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    
    # 类别完全相同时，则停止划分
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    
    # 遍历完所有特征时，返回出现次数最多的类别
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del (labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]  # 对于每个子集,subLabels互不影响
        myTree[bestFeatLabel][value] = createTree(
            splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree

myTree = createTree(dataSet, labels)
myTree


# In[60]:

# 使用决策树的分类函数
def classify(inputTree, featLabels, testVec):
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel


dataSet, labels = createDataSet()
splitDataSet(dataSet, 1, 1)  # 索引为1的特征被除去了
classify(myTree, labels, [1,0])


#python2.x dict.keys() 返回list类型，可直接使用索引获取其元素
#python3.x dict.keys() 返回dict_keys类型，其性质类似集合(set)而不是列表(list)，因此不能使用索引获取其元素
# type(myTree.keys())
#dict_keys
 
#python3.x解决办法
# list(myTree.keys())[0]


# In[69]:

# 决策树的储存
def storeTree(inputTree, filename):
    import pickle
    fw = open(filename, "wb")
    pickle.dump(inputTree, fw)
    
def grabTree(filename):
    import pickle
    fr = open(filename, 'rb')
    return pickle.load(fr)
myTree = {'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}}
storeTree(myTree, 'week2_1210_classifierstorage.txt')
grabTree('week2_1210_classifierstorage.txt')


# In[74]:

fr = open('./example_data/Ch03/lenses.txt')
lenses = [inst.strip().split('\t') for inst in fr.readlines()]
lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
lensesTree = createTree(lenses, lensesLabels)
lensesTree


# In[ ]:



